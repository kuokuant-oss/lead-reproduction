"""Checkpointed formal Tree runner for Steam/Hotwater-only 50k contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lead import ROOT, load_m3_frame  # noqa: E402
from lead.m5_context import feature_names  # noqa: E402
from run_m3_figure_observations import (  # noqa: E402
    MODEL_ORDER,
    build_frozen_models,
    frozen_model_contract,
    predict_probability,
)
from run_m5_tree_ensemble_matched_context import (  # noqa: E402
    build_features_keeping_index,
    feature_columns,
)


CONDITIONS = ("steam_only", "steam_hw_all", "steam_hw_anomaly", "steam_hw_normal")
ROWS = 10_137_155


def digest(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(values, dtype="<i8")).tobytes()
    ).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def repo_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_joblib(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    joblib.dump(value, tmp)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def preflight(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "m5_eh_50k_steam_hotwater_preflight_v1" or set(
        value.get("manifests", ())
    ) != set(CONDITIONS):
        raise ValueError("missing or wrong Steam/Hotwater preflight")
    for name in CONDITIONS:
        item = value["manifests"][name]
        raw = np.asarray(item["raw_index"], dtype="int64")
        if (
            len(raw) != 50_000
            or len(np.unique(raw)) != 50_000
            or item["summary"]["label_counts"] != {"normal": 25_000, "anomaly": 25_000}
            or digest(raw) != item["summary"]["raw_index_sha256"]
        ):
            raise ValueError(f"{name}: invalid preflight manifest")
    return value


def verify_source(value: dict[str, Any], m3: Path) -> None:
    observed = {name: file_digest(m3 / name) for name in value["source_sha256"]}
    if observed != value["source_sha256"]:
        raise ValueError("M3 source digest mismatch")


def heartbeat(root: Path, **extra: Any) -> None:
    atomic_json(
        root / "heartbeat.json",
        {"schema": "m5_eh_heartbeat_v1", "timestamp": time.time(), **extra},
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--m3-root", type=Path, default=ROOT / "data" / "raw" / "m3")
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("dry-run", "formal"), default="dry-run")
    p.add_argument("--confirm", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--predict-batch-rows", type=int, default=100_000)
    return p.parse_args()


def dry_run(value: dict[str, Any], args: argparse.Namespace) -> None:
    verify_source(value, args.m3_root)
    if args.predict_batch_rows <= 0 or args.predict_batch_rows > 100_000:
        raise ValueError("dry run rejects unbounded scoring batch")
    with np.load(args.canonical, allow_pickle=False) as a002:
        raw = a002["raw_index"]
    if (
        len(raw) != ROWS
        or len(np.unique(raw)) != ROWS
        or len(feature_names("F4")) != 137
    ):
        raise AssertionError("canonical/F4 identity gate failed")
    print(
        json.dumps(
            {
                "mode": "dry-run",
                "fit": 0,
                "predict": 0,
                "conditions": list(CONDITIONS),
                "component_fits": 16,
                "canonical_rows": len(raw),
                "repository_commit": repo_commit(),
            },
            sort_keys=True,
        )
    )


def validate_rows(value: dict[str, Any], frame: Any) -> dict[str, np.ndarray]:
    result = {
        name: np.asarray(value["manifests"][name]["raw_index"], dtype="int64")
        for name in CONDITIONS
    }
    for name, raw in result.items():
        rows = frame.loc[raw]
        if (rows["building_id"].to_numpy() % 2).any() or not np.isin(
            rows["meter"], (2, 3)
        ).all():
            raise AssertionError(f"{name}: building or meter isolation failed")
        if not np.array_equal(
            rows["anomaly"].to_numpy(dtype="int8"),
            np.tile(np.array([1, 0], dtype="int8"), 25_000),
        ):
            raise AssertionError(f"{name}: ordered 50:50 label gate failed")
        if name == "steam_only" and (rows["meter"] == 3).any():
            raise AssertionError("steam_only includes hotwater")
        if (
            name == "steam_hw_anomaly"
            and ((rows["meter"] == 3) & (rows["anomaly"] == 0)).any()
        ):
            raise AssertionError("steam_hw_anomaly includes hotwater normal")
        if (
            name == "steam_hw_normal"
            and ((rows["meter"] == 3) & (rows["anomaly"] == 1)).any()
        ):
            raise AssertionError("steam_hw_normal includes hotwater anomaly")
    return result


def fit_models(
    name: str,
    raw: np.ndarray,
    train_full: Any,
    frame: Any,
    columns: list[str],
    root: Path,
    base: dict[str, Any],
    resume: bool,
) -> tuple[StandardScaler, dict[str, Any], dict[str, Any]]:
    selected = train_full.loc[raw]
    if not np.array_equal(selected.index.to_numpy(dtype="int64"), raw):
        raise AssertionError(f"{name}: feature order drift")
    scaler_path = root / "scalers" / f"{name}.joblib"
    if resume and scaler_path.is_file():
        saved = joblib.load(scaler_path)
        if (
            saved.get("raw_index_sha256") != digest(raw)
            or saved.get("feature_names") != columns
        ):
            raise ValueError(f"{name}: scaler provenance mismatch")
        scaler = saved["scaler"]
    else:
        scaler = StandardScaler().fit(selected[columns].to_numpy(dtype="float32"))
        atomic_joblib(
            scaler_path,
            {
                "scaler": scaler,
                "raw_index_sha256": digest(raw),
                "feature_names": columns,
            },
        )
    provenance = base | {
        "condition": name,
        "raw_index_sha256": digest(raw),
        "scaler_sha256": file_digest(scaler_path),
    }
    x = scaler.transform(selected[columns].to_numpy(dtype="float32")).astype(
        "float32", copy=False
    )
    y = frame.loc[raw, "anomaly"].to_numpy(dtype="int8")
    models: dict[str, Any] = {}
    for component in MODEL_ORDER:
        path = root / "models" / name / f"{component}.joblib"
        if resume and path.is_file():
            saved = joblib.load(path)
            if saved.get("provenance") != provenance | {"component": component}:
                raise ValueError(f"{name}/{component}: model provenance mismatch")
            models[component] = saved["model"]
            continue
        started = time.perf_counter()
        model = build_frozen_models(42)[component]
        model.fit(
            np.nan_to_num(x, nan=0) if component == "hist_gradient_boosting" else x, y
        )
        atomic_joblib(
            path,
            {
                "model": model,
                "provenance": provenance | {"component": component},
                "fit_seconds": time.perf_counter() - started,
            },
        )
        models[component] = model
        heartbeat(
            root,
            phase="fit",
            active=f"{name}/{component}",
            completed_models=sum(1 for _ in (root / "models").glob("*/*.joblib")),
            expected_models=16,
        )
    return scaler, models, provenance


def score(
    name: str,
    models: dict[str, Any],
    scaler: StandardScaler,
    holdout: Any,
    raw: np.ndarray,
    columns: list[str],
    root: Path,
    batch: int,
    provenance: dict[str, Any],
) -> None:
    cell = root / "scores" / name
    spans = [
        (start, min(len(raw), start + batch)) for start in range(0, len(raw), batch)
    ]
    for start, end in spans:
        path = cell / "microbatches" / f"mb_{start:09d}_{end:09d}.npz"
        if path.is_file():
            with np.load(path) as saved:
                if set(saved.files) != set(MODEL_ORDER) or any(
                    len(saved[k]) != end - start for k in MODEL_ORDER
                ):
                    raise ValueError(f"{name}: corrupt score checkpoint")
            continue
        x = scaler.transform(
            holdout.loc[raw[start:end], columns].to_numpy(dtype="float32")
        )
        atomic_npz(
            path,
            **{
                k: predict_probability(k, models[k], x).astype("float32")
                for k in MODEL_ORDER
            },
        )
        heartbeat(
            root,
            phase="score",
            active=f"{name}/{start}:{end}",
            completed_models=16,
            expected_models=16,
        )
    values = {k: np.empty(len(raw), dtype="float32") for k in MODEL_ORDER}
    for start, end in spans:
        with np.load(cell / "microbatches" / f"mb_{start:09d}_{end:09d}.npz") as saved:
            for key in MODEL_ORDER:
                values[key][start:end] = saved[key]
    values["ensemble"] = np.mean(
        [values[k] for k in MODEL_ORDER], axis=0, dtype="float32"
    )
    if not all(np.isfinite(item).all() for item in values.values()):
        raise AssertionError(f"{name}: non-finite score")
    atomic_npz(cell / "scores.npz", raw_index=raw, **values)
    atomic_json(
        cell / "CELL_COMPLETE.json",
        {
            "provenance": provenance,
            "rows": len(raw),
            "raw_index_sha256": digest(raw),
            "score_fields": [*MODEL_ORDER, "ensemble"],
        },
    )


def formal(value: dict[str, Any], args: argparse.Namespace) -> None:
    if args.confirm != "開始":
        raise SystemExit("formal mode requires --confirm 開始")
    if args.m3_root.resolve() != (ROOT / "data" / "raw" / "m3").resolve():
        raise SystemExit("formal mode only permits repository M3 root")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit("formal output root is non-empty; use --resume")
    args.out.mkdir(parents=True, exist_ok=True)
    verify_source(value, args.m3_root)
    heartbeat(args.out, phase="initialising", completed_models=0, expected_models=16)
    frame = load_m3_frame(verbose=True)
    contexts = validate_rows(value, frame)
    with np.load(args.canonical, allow_pickle=False) as a002:
        holdout_raw = a002["raw_index"].astype("int64", copy=True)
    train_mask = frame["building_id"].to_numpy() % 2 == 0
    if not np.array_equal(
        np.sort(holdout_raw), frame.index[~train_mask].to_numpy(dtype="int64")
    ):
        raise AssertionError("A002 order is not exactly odd-building holdout")
    train_full = build_features_keeping_index(frame.loc[train_mask])
    columns = feature_columns(137, list(train_full.columns))
    if columns != feature_names("F4"):
        raise AssertionError("F4 feature order drift")
    base = {
        "preflight_sha256": file_digest(args.preflight),
        "source_sha256": value["source_sha256"],
        "feature_names_sha256": hashlib.sha256("\n".join(columns).encode()).hexdigest(),
        "model_contract": frozen_model_contract(42),
        "platform": platform.platform(),
        "repository_commit": repo_commit(),
    }
    holdout = None
    for name in CONDITIONS:
        scaler, models, provenance = fit_models(
            name,
            contexts[name],
            train_full,
            frame,
            columns,
            args.out,
            base,
            args.resume,
        )
        if holdout is None:
            holdout = build_features_keeping_index(frame.loc[~train_mask])
        score(
            name,
            models,
            scaler,
            holdout,
            holdout_raw,
            columns,
            args.out,
            args.predict_batch_rows,
            provenance,
        )
    atomic_json(
        args.out / "FORMAL_COMPLETE.json",
        {
            "expected_models": 16,
            "conditions": list(CONDITIONS),
            "holdout_rows": len(holdout_raw),
            "repository_commit": repo_commit(),
        },
    )


def main() -> int:
    args = parse_args()
    value = preflight(args.preflight)
    if args.mode == "dry-run":
        dry_run(value, args)
    else:
        formal(value, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
