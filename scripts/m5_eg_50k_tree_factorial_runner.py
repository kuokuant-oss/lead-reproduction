"""Checkpointed laptop-only Tree runner for the strict E0/E1 50k factorial.

Default mode is a non-scientific dry run.  Formal fitting requires both
``--mode formal`` and ``--confirm 開始``.  The four components are independent
fit checkpoints; scoring is checkpointed per cell and fixed holdout microbatch.
"""

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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

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


EXPECTED_CONTEXT_DIGEST = (
    "e9ffe0cffd2e0cf304d213a02e68f2d7ef092172efc0343e680f982a2d688cbe"
)
CELLS = ("11", "01", "10", "00")
FEATURES = 137


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


def repository_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_joblib(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    joblib.dump(payload, tmp)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_preflight(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "m5_eg_50k_tree_factorial_preflight_v2":
        raise ValueError("unsupported or incomplete preflight schema")
    if payload.get("e0_100k_digest") != EXPECTED_CONTEXT_DIGEST or not payload.get(
        "cell11_is_exact_e0_prefix"
    ):
        raise ValueError("E0/E1 exact-context identity gate failed")
    if set(payload.get("cells", ())) != set(CELLS):
        raise ValueError("preflight cell census is incomplete")
    for cell in CELLS:
        item = payload["cells"][cell]
        raw = np.asarray(item["raw_index"], dtype="int64")
        if (
            len(raw) != 50_000
            or len(np.unique(raw)) != 50_000
            or item["summary"]["label_counts"] != {"normal": 25_000, "anomaly": 25_000}
        ):
            raise ValueError(f"{cell}: invalid 50k row manifest")
        if digest(raw) != item["summary"]["raw_index_sha256"]:
            raise ValueError(f"{cell}: raw-index digest drift")
    return payload


def validate_source(preflight: dict[str, Any], m3_root: Path) -> None:
    expected = preflight.get("source_sha256", {})
    observed = {name: file_digest(m3_root / name) for name in expected}
    if observed != expected:
        raise ValueError("M3 source digest mismatch; refuse fit")


def heartbeat(out: Path, **extra: Any) -> None:
    atomic_json(
        out / "heartbeat.json",
        {"schema": "m5_eg_tree_heartbeat_v1", "timestamp": time.time(), **extra},
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--m3-root", type=Path, default=ROOT / "data" / "raw" / "m3")
    p.add_argument(
        "--canonical",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "legacy"
        / "m5_tabpfn_137_full_test_n8_predictions.npz",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("dry-run", "formal"), default="dry-run")
    p.add_argument("--confirm", default="")
    p.add_argument(
        "--resume",
        action="store_true",
        help="reuse only validated checkpoints in an existing formal root",
    )
    p.add_argument("--predict-batch-rows", type=int, default=100_000)
    return p.parse_args()


def dry_run(preflight: dict[str, Any], args: argparse.Namespace) -> None:
    validate_source(preflight, args.m3_root)
    if args.predict_batch_rows <= 0 or args.predict_batch_rows > 100_000:
        raise ValueError("dry run rejects unbounded scoring batch size")
    expected_features = feature_names("F4")
    if len(expected_features) != FEATURES or len(set(expected_features)) != FEATURES:
        raise AssertionError("F4 feature contract is not exactly 137 ordered columns")
    with np.load(args.canonical, allow_pickle=False) as canonical:
        raw = canonical["raw_index"]
        if len(raw) != 10_137_155 or len(np.unique(raw)) != len(raw):
            raise AssertionError("canonical A002 raw-index gate failed")
    print(
        json.dumps(
            {
                "mode": "dry-run",
                "fit": 0,
                "predict": 0,
                "cells": list(CELLS),
                "components": len(CELLS) * len(MODEL_ORDER),
                "predict_batch_rows": args.predict_batch_rows,
                "source_digest_gate": "passed",
                "canonical_rows": len(raw),
                "feature_count": FEATURES,
                "environment": {
                    "platform": platform.platform(),
                    "python": sys.version.split()[0],
                },
                "repository_commit": repository_commit(),
            },
            sort_keys=True,
        )
    )


def validate_cells_against_frame(
    preflight: dict[str, Any], frame: Any
) -> dict[str, np.ndarray]:
    result = {
        cell: np.asarray(preflight["cells"][cell]["raw_index"], dtype="int64")
        for cell in CELLS
    }
    for cell, raw in result.items():
        rows = frame.loc[raw]
        if (rows["building_id"].to_numpy() % 2).any() or not np.array_equal(
            rows["anomaly"].to_numpy(dtype="int8"),
            np.asarray([1 if i % 2 == 0 else 0 for i in range(len(raw))], dtype="int8"),
        ):
            raise AssertionError(
                f"{cell}: even-building or exact alternating-label gate failed"
            )
        for label, present in ((1, cell[0] == "1"), (0, cell[1] == "1")):
            if (
                bool(((rows["meter"] == 3) & (rows["anomaly"] == label)).any())
                != present
            ):
                raise AssertionError(f"{cell}: hotwater intervention gate failed")
    return result


def fit_or_load(
    cell: str,
    raw: np.ndarray,
    train_full: Any,
    frame: Any,
    columns: list[str],
    scaler: StandardScaler,
    out: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    selected = train_full.loc[raw]
    if not np.array_equal(selected.index.to_numpy(dtype="int64"), raw):
        raise AssertionError(f"{cell}: feature selection changed row order")
    x = scaler.transform(selected[columns].to_numpy(dtype="float32")).astype(
        "float32", copy=False
    )
    y = frame.loc[raw, "anomaly"].to_numpy(dtype="int8")
    models: dict[str, Any] = {}
    for name in MODEL_ORDER:
        target = out / "models" / f"cell{cell}" / f"{name}.joblib"
        if target.is_file():
            saved = joblib.load(target)
            if saved.get("provenance") != provenance | {
                "cell": cell,
                "component": name,
            }:
                raise ValueError(f"{cell}/{name}: cached component provenance mismatch")
            models[name] = saved["model"]
            continue
        started = time.perf_counter()
        model = build_frozen_models(42)[name]
        model.fit(np.nan_to_num(x, nan=0) if name == "hist_gradient_boosting" else x, y)
        atomic_joblib(
            target,
            {
                "model": model,
                "provenance": provenance | {"cell": cell, "component": name},
                "fit_seconds": time.perf_counter() - started,
            },
        )
        models[name] = model
        heartbeat(
            out,
            phase="fit",
            completed_models=sum((out / "models").glob("cell*/*.joblib")),
            expected_models=16,
            active=f"{cell}/{name}",
        )
    return models


def score_cell(
    cell: str,
    models: dict[str, Any],
    holdout_full: Any,
    holdout_raw: np.ndarray,
    columns: list[str],
    scaler: StandardScaler,
    out: Path,
    batch: int,
    provenance: dict[str, Any],
) -> None:
    root = out / "scores" / f"cell{cell}"
    spans = [
        (s, min(len(holdout_raw), s + batch)) for s in range(0, len(holdout_raw), batch)
    ]
    for start, end in spans:
        part = root / "microbatches" / f"mb_{start:09d}_{end:09d}.npz"
        if part.is_file():
            with np.load(part) as saved:
                if set(saved.files) != set(MODEL_ORDER) or any(
                    len(saved[name]) != end - start for name in MODEL_ORDER
                ):
                    raise ValueError(f"{cell}: corrupt microbatch checkpoint {part}")
            continue
        block = scaler.transform(
            holdout_full.loc[holdout_raw[start:end], columns].to_numpy(dtype="float32")
        )
        atomic_npz(
            part,
            **{
                name: predict_probability(name, models[name], block).astype("float32")
                for name in MODEL_ORDER
            },
        )
        heartbeat(
            out,
            phase="score",
            active=f"{cell}/{start}:{end}",
            completed_models=16,
            expected_models=16,
        )
    values = {name: np.empty(len(holdout_raw), dtype="float32") for name in MODEL_ORDER}
    for start, end in spans:
        with np.load(root / "microbatches" / f"mb_{start:09d}_{end:09d}.npz") as saved:
            for name in MODEL_ORDER:
                values[name][start:end] = saved[name]
    values["ensemble"] = np.mean(
        [values[name] for name in MODEL_ORDER], axis=0, dtype="float32"
    )
    if not all(np.isfinite(value).all() for value in values.values()):
        raise AssertionError(f"{cell}: non-finite score")
    atomic_npz(root / "scores.npz", raw_index=holdout_raw, **values)
    atomic_json(
        root / "CELL_COMPLETE.json",
        {
            "provenance": provenance | {"cell": cell},
            "rows": len(holdout_raw),
            "score_fields": [*MODEL_ORDER, "ensemble"],
            "raw_index_sha256": digest(holdout_raw),
        },
    )


def formal(preflight: dict[str, Any], args: argparse.Namespace) -> None:
    if args.confirm != "開始":
        raise SystemExit("formal mode requires --confirm 開始")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit(
            "formal output root is non-empty; pass --resume to reuse validated checkpoints"
        )
    if args.m3_root.resolve() != (ROOT / "data" / "raw" / "m3").resolve():
        raise SystemExit(
            "formal runner only permits the repository M3 root used by load_m3_frame"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    validate_source(preflight, args.m3_root)
    heartbeat(args.out, phase="initialising", completed_models=0, expected_models=16)
    frame = load_m3_frame(verbose=True)
    cells = validate_cells_against_frame(preflight, frame)
    train_mask = (frame["building_id"] % 2 == 0).to_numpy()
    with np.load(args.canonical, allow_pickle=False) as canonical:
        holdout_raw = canonical["raw_index"].astype("int64", copy=True)
    if len(holdout_raw) != 10_137_155:
        raise AssertionError("canonical holdout row count drift")
    if not np.array_equal(
        np.sort(holdout_raw), frame.index[~train_mask].to_numpy(dtype="int64")
    ):
        raise AssertionError(
            "A002 canonical holdout is not exactly the odd-building row set"
        )
    train_full = build_features_keeping_index(frame.loc[train_mask])
    columns = feature_columns(FEATURES, list(train_full.columns))
    if columns != feature_names("F4"):
        raise AssertionError("F4 column order drift")
    raw11 = cells["11"]
    scaler_path = args.out / "shared_scaler.joblib"
    if args.resume and scaler_path.is_file():
        scaler_payload = joblib.load(scaler_path)
        if (
            scaler_payload.get("fit_cell") != "11"
            or scaler_payload.get("raw_index_sha256") != digest(raw11)
            or scaler_payload.get("feature_names") != columns
        ):
            raise ValueError("cached shared scaler provenance mismatch")
        scaler = scaler_payload["scaler"]
    else:
        scaler = StandardScaler().fit(
            train_full.loc[raw11, columns].to_numpy(dtype="float32")
        )
        atomic_joblib(
            scaler_path,
            {
                "scaler": scaler,
                "fit_cell": "11",
                "raw_index_sha256": digest(raw11),
                "feature_names": columns,
            },
        )
    provenance = {
        "preflight_sha256": file_digest(args.preflight),
        "source_sha256": preflight["source_sha256"],
        "feature_names_sha256": hashlib.sha256("\n".join(columns).encode()).hexdigest(),
        "scaler_sha256": file_digest(scaler_path),
        "model_contract": frozen_model_contract(42),
        "platform": platform.platform(),
        "repository_commit": repository_commit(),
    }
    for cell in CELLS:
        models = fit_or_load(
            cell, cells[cell], train_full, frame, columns, scaler, args.out, provenance
        )
        if cell == CELLS[0]:
            holdout_full = build_features_keeping_index(frame.loc[~train_mask])
        score_cell(
            cell,
            models,
            holdout_full,
            holdout_raw,
            columns,
            scaler,
            args.out,
            args.predict_batch_rows,
            provenance,
        )
    atomic_json(
        args.out / "FORMAL_COMPLETE.json",
        {
            "expected_models": 16,
            "completed_cells": list(CELLS),
            "holdout_rows": len(holdout_raw),
            "provenance": provenance,
        },
    )


def main() -> int:
    args = parse_args()
    preflight = read_preflight(args.preflight)
    if args.mode == "dry-run":
        dry_run(preflight, args)
    else:
        formal(preflight, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
