"""Gated, resumable TabPFN runner for Steam/Hotwater 20K-prefix and 50K contexts.

20K contexts are deliberately defined as the first 20,000 rows of the matching
authoritative 50K vector.  This creates a nested, order-preserving derivative
context without a new sampler or any post-load resampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from lead import BASELINE_FEATURE_COLS, load_m3_frame  # noqa: E402
from run_m5_story_ae_probe import SHIFTS, add_value_change_features, feature_names  # noqa: E402

FROZEN_50K = {
    "steam_only": "4defe9e9498b302f308fcd902f140f5750195ff664df8f9945c8b5928abca65e",
    "steam_hw_normal": "a0b50958f395668b3be25a17dc6720c98dca98ef73b1d3d49817d84933f54da5",
    "steam_hw_anomaly": "9f4db0387c3d90220fdc22c85f4f536ad912169bfc2da4e3009eed8692c0b4c7",
    "steam_hw_all": "acc441899b7aa14bc18833a8a9f4bf1014107d9b7500fd828c359493900fdfe3",
}
SOURCE_SHA256 = {
    "train.csv": "2d75e0c4cfa93818647cf5272ef8d48f9cf8e9d3479dde357faae381d1abbcb3",
    "bad_meter_readings.csv": "e9846b746bc584f3f57f30f54d2077beb294e39407146ff6c9ad024f806bab93",
    "building_metadata.csv": "357bf585047359e9dfbaef2935453429f8ec19e5e80c08a8eb066789f28c4070",
    "weather_train.csv": "81022191f16dacc21494c15dac7975611cb39922fc7332e419a857cbb00cc125",
}
VALIDATION_SHA256 = "4f2002bfad4feba4ac3cf235ad724496bcd9845947650ce64367f44e0baa99f9"
CONDITIONS = tuple(FROZEN_50K)
LABELS = tuple(
    f"{budget}k_{condition}" for budget in (20, 50) for condition in CONDITIONS
)
FEATURES = 137
HOLDOUT_ROWS = 10_137_155
MICROBATCH_ROWS = 20_000
MODEL_SEED = 42
N_ESTIMATORS = 8
CHECKPOINT_SHA256 = "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_i64(values: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(values, dtype="<i8")).tobytes())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("wb") as fh:
        np.save(fh, value)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def repo_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def label_parts(label: str) -> tuple[int, str]:
    budget, condition = label.split("_", 1)
    rows = int(budget.removesuffix("k")) * 1_000
    if label not in LABELS:
        raise ValueError(f"unknown context label {label}")
    return rows, condition


def load_contexts(preflight: Path) -> dict[str, np.ndarray]:
    value = json.loads(preflight.read_text(encoding="utf-8"))
    manifests = value.get("manifests", {})
    if value.get("schema") != "m5_eh_50k_steam_hotwater_preflight_v1" or set(
        manifests
    ) != set(CONDITIONS):
        raise ValueError("wrong 50K Steam/Hotwater preflight")
    base: dict[str, np.ndarray] = {}
    for condition, expected in FROZEN_50K.items():
        raw = np.asarray(manifests[condition]["raw_index"], dtype="int64")
        if (
            len(raw) != 50_000
            or len(np.unique(raw)) != 50_000
            or sha256_i64(raw) != expected
        ):
            raise ValueError(f"{condition}: authoritative 50K raw_index gate failed")
        base[condition] = raw
    return {
        **{f"20k_{condition}": raw[:20_000].copy() for condition, raw in base.items()},
        **{f"50k_{condition}": raw.copy() for condition, raw in base.items()},
    }


def verify_sources(m3_root: Path) -> None:
    observed = {name: sha256_file(m3_root / name) for name in SOURCE_SHA256}
    if observed != SOURCE_SHA256:
        raise ValueError(f"M3 source digest mismatch: {observed}")


def validation_index(train_csv: Path) -> np.ndarray:
    train = pd.read_csv(
        train_csv, usecols=["building_id"], dtype={"building_id": "int16"}
    )
    raw = np.arange(len(train), dtype="int64")
    even = train["building_id"].to_numpy() % 2 == 0
    validation = (
        np.random.RandomState(20_042)
        .choice(raw[even], size=4_000, replace=False)
        .astype("int64")
    )
    if sha256_i64(np.sort(validation)) != VALIDATION_SHA256:
        raise ValueError("fixed validation reconstruction digest failed")
    return validation


def verify_context(
    label: str, raw: np.ndarray, frame: pd.DataFrame, validation: np.ndarray
) -> dict[str, Any]:
    expected_rows, condition = label_parts(label)
    rows = frame.loc[raw]
    meter = rows["meter"].to_numpy(dtype="int8")
    target = rows["anomaly"].to_numpy(dtype="int8")
    if len(raw) != expected_rows or len(np.unique(raw)) != expected_rows:
        raise ValueError(f"{label}: row cardinality gate failed")
    if not np.array_equal(rows.index.to_numpy(dtype="int64"), raw):
        raise ValueError(f"{label}: frame selection reordered raw_index")
    if (rows["building_id"].to_numpy() % 2).any() or np.intersect1d(
        raw, validation
    ).size:
        raise ValueError(f"{label}: split or validation exclusion gate failed")
    if not np.isin(meter, (2, 3)).all() or target.sum() != expected_rows // 2:
        raise ValueError(f"{label}: meter or balanced-label gate failed")
    if condition == "steam_only" and (meter == 3).any():
        raise ValueError(f"{label}: steam_only includes Hotwater")
    if condition == "steam_hw_normal" and ((meter == 3) & (target == 1)).any():
        raise ValueError(f"{label}: includes Hotwater anomaly")
    if condition == "steam_hw_anomaly" and ((meter == 3) & (target == 0)).any():
        raise ValueError(f"{label}: includes Hotwater normal")
    return {
        "rows": expected_rows,
        "raw_index_sha256": sha256_i64(raw),
        "positive": int(target.sum()),
        "negative": int((target == 0).sum()),
    }


def feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    even = frame["building_id"].to_numpy() % 2 == 0
    train = frame.loc[even].copy()
    train["_raw_index"] = train.index.to_numpy(dtype="int64")
    built = add_value_change_features(
        train, list(SHIFTS), value_change_regime="timestamp_merge"
    )
    built = built.set_index("_raw_index")
    columns = list(BASELINE_FEATURE_COLS) + [
        column for column in built.columns if column.startswith("lag_value_")
    ]
    if columns != list(feature_names("F4")) or len(columns) != FEATURES:
        raise ValueError("F4/137 feature identity gate failed")
    return built, columns


def checkpoint_ok(
    path: Path, shape: tuple[int, ...], provenance: dict[str, Any]
) -> bool:
    marker = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not marker.is_file():
        return False
    try:
        saved = np.load(path, mmap_mode="r")
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        return saved.shape == shape and metadata == provenance
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def save_checkpoint(path: Path, value: np.ndarray, provenance: dict[str, Any]) -> None:
    atomic_npy(path, value)
    atomic_json(path.with_suffix(path.suffix + ".json"), provenance)


def prepare(
    args: argparse.Namespace,
    contexts: dict[str, np.ndarray],
    context_info: dict[str, Any],
    frame: pd.DataFrame,
) -> None:
    built, columns = feature_frame(frame)
    for label, raw in contexts.items():
        info = context_info[label]
        root = args.out / "contexts" / label
        provenance = {
            "schema": "m5_steam_hotwater_tabpfn_context_v1",
            "label": label,
            "raw_index_sha256": info["raw_index_sha256"],
            "feature_names": columns,
            "repository_commit": repo_commit(),
        }
        x_path, y_path = root / "x.float32.npy", root / "y.int8.npy"
        if not checkpoint_ok(x_path, (len(raw), FEATURES), provenance):
            selected = built.loc[raw, columns]
            if not np.array_equal(selected.index.to_numpy(dtype="int64"), raw):
                raise ValueError(f"{label}: F4 selection reordered raw_index")
            save_checkpoint(
                x_path, selected.to_numpy(dtype="float32", copy=True), provenance
            )
        if not checkpoint_ok(y_path, (len(raw),), provenance):
            save_checkpoint(
                y_path,
                frame.loc[raw, "anomaly"].to_numpy(dtype="int8", copy=True),
                provenance,
            )
        scaler_path = root / "scaler.joblib"
        if not scaler_path.is_file():
            scaler = StandardScaler().fit(np.load(x_path, mmap_mode="r"))
            tmp = scaler_path.with_name(f".{scaler_path.name}.{os.getpid()}.tmp")
            joblib.dump({"provenance": provenance, "scaler": scaler}, tmp)
            os.replace(tmp, scaler_path)
        saved = joblib.load(scaler_path)
        if saved.get("provenance") != provenance:
            raise ValueError(f"{label}: scaler provenance mismatch")
    atomic_json(
        args.out / "PREPARE_COMPLETE.json",
        {"contexts": context_info, "repository_commit": repo_commit()},
    )


def score_one(
    args: argparse.Namespace, label: str, context_info: dict[str, Any]
) -> None:
    from run_m5_tabpfn_canonical_full_test import (
        create_real_model,
        verify_fitted_context,
    )

    root = args.out / "contexts" / label
    x = np.load(root / "x.float32.npy", mmap_mode="r")
    y = np.load(root / "y.int8.npy", mmap_mode="r")
    saved = joblib.load(root / "scaler.joblib")
    scaler = saved["scaler"]
    if x.shape != (context_info[label]["rows"], FEATURES) or y.shape != (len(x),):
        raise ValueError(f"{label}: context checkpoint shape mismatch")
    if sha256_file(args.model_path) != CHECKPOINT_SHA256:
        raise ValueError("TabPFN checkpoint digest mismatch")
    model = create_real_model(args.model_path, MODEL_SEED, N_ESTIMATORS)
    model.fit(scaler.transform(np.asarray(x)).astype("float32", copy=False), y)
    fitted = verify_fitted_context(model, len(x), N_ESTIMATORS)
    if (
        fitted.get("status") != "verified"
        or fitted.get("effective_estimators") != N_ESTIMATORS
    ):
        raise ValueError(f"{label}: fitted-context verification failed")
    holdout = np.load(
        args.feature_root / "e6_holdout_raw_f4_137.float32.npy", mmap_mode="r"
    )
    raw = np.load(args.feature_root / "e6_holdout_raw_index.npy", mmap_mode="r")
    if holdout.shape != (HOLDOUT_ROWS, FEATURES) or raw.shape != (HOLDOUT_ROWS,):
        raise ValueError("canonical raw holdout shape gate failed")
    out = args.out / "scores" / label
    parts = out / "microbatches"
    for batch, start in enumerate(range(0, HOLDOUT_ROWS, MICROBATCH_ROWS)):
        stop = min(start + MICROBATCH_ROWS, HOLDOUT_ROWS)
        part = parts / f"mb_{batch:04d}.npy"
        provenance = {
            "label": label,
            "start": start,
            "stop": stop,
            "raw_index_sha256": context_info[label]["raw_index_sha256"],
            "repository_commit": repo_commit(),
        }
        if checkpoint_ok(part, (stop - start,), provenance):
            continue
        prediction = model.predict_proba(
            scaler.transform(np.asarray(holdout[start:stop])).astype(
                "float32", copy=False
            )
        )[:, 1].astype("float32")
        if prediction.shape != (stop - start,) or not np.isfinite(prediction).all():
            raise ValueError(f"{label}: invalid microbatch {batch}")
        save_checkpoint(part, prediction, provenance)
        atomic_json(
            args.out / "heartbeat.json",
            {
                "active": label,
                "batch": batch,
                "total_batches": int(np.ceil(HOLDOUT_ROWS / MICROBATCH_ROWS)),
                "timestamp": time.time(),
            },
        )
    scores = np.concatenate(
        [
            np.load(parts / f"mb_{batch:04d}.npy")
            for batch in range(int(np.ceil(HOLDOUT_ROWS / MICROBATCH_ROWS)))
        ]
    )
    if scores.shape != (HOLDOUT_ROWS,) or not np.isfinite(scores).all():
        raise ValueError(f"{label}: checkpoint assembly failed")
    atomic_npy(out / "scores.float32.npy", scores)
    atomic_json(
        out / "CELL_COMPLETE.json",
        {
            "label": label,
            "rows": HOLDOUT_ROWS,
            "context": context_info[label],
            "scores_sha256": sha256_bytes(np.ascontiguousarray(scores).tobytes()),
            "repository_commit": repo_commit(),
        },
    )


def evaluate(
    args: argparse.Namespace, label: str, frame: pd.DataFrame
) -> dict[str, Any]:
    raw = np.load(args.feature_root / "e6_holdout_raw_index.npy", mmap_mode="r")
    scores = np.load(args.out / "scores" / label / "scores.float32.npy", mmap_mode="r")
    steam = frame.loc[raw, "meter"].to_numpy(dtype="int8") == 2
    y = frame.loc[raw, "anomaly"].to_numpy(dtype="int8")[steam]
    s = scores[steam].astype("float64")
    return {
        "steam_rows": int(steam.sum()),
        "steam_positives": int(y.sum()),
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--m3-root", type=Path, default=ROOT / "data" / "raw" / "m3")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("validate", "formal"), default="validate")
    parser.add_argument("--confirm", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contexts = load_contexts(args.preflight)
    verify_sources(args.m3_root)
    validation = validation_index(args.m3_root / "train.csv")
    frame = load_m3_frame(verbose=True)
    context_info = {
        label: verify_context(label, raw, frame, validation)
        for label, raw in contexts.items()
    }
    if args.mode == "validate":
        print(
            json.dumps(
                {"mode": "validate", "contexts": context_info, "fit": 0, "predict": 0},
                sort_keys=True,
            )
        )
        return 0
    if args.confirm != "M5_STEAM_HOTWATER_TABPFN_FORMAL":
        raise SystemExit("formal mode requires the explicit confirm token")
    args.out.mkdir(parents=True, exist_ok=True)
    prepare(args, contexts, context_info, frame)
    metrics = {}
    for label in LABELS:
        score_one(args, label, context_info)
        metrics[label] = evaluate(args, label, frame)
        atomic_json(args.out / "metrics.json", metrics)
    atomic_json(
        args.out / "FORMAL_COMPLETE.json",
        {
            "labels": list(LABELS),
            "metrics": metrics,
            "repository_commit": repo_commit(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
