"""Resumably rebuild the frozen M3 17- or 137-feature ensemble predictions.

Each fitted component model is a durable unit.  Its predictions are atomically
added to a feature-count checkpoint, so an interruption never discards a
completed model.  ``--mode validation`` is intentionally isolated from report
outputs and requires deterministic building caps; ``--mode formal`` is the
only mode that may write the report-input artefacts.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from lead import (
    BASELINE_FEATURE_COLS,
    PROC,
    RANDOM_STATE,
    ROOT,
    SHIFTS,
    add_value_change_features,
    downsample_indices,
    load_m3_frame,
)


MODEL_ORDER = ("lightgbm", "xgboost", "catboost", "hist_gradient_boosting")
EXPECTED_FULL_SPLIT = {
    "train_buildings": 725,
    "validation_buildings": 724,
    "train_rows": 10_078_945,
    "validation_rows": 10_137_155,
}
SCRIPT_SCHEMA = 1


def log(message: str) -> None:
    print(message, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write, reopen, and validate an NPZ before making it visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    with np.load(temporary) as payload:
        if set(payload.files) != set(arrays):
            raise RuntimeError("temporary prediction checkpoint schema mismatch")
        for name, expected in arrays.items():
            observed = np.asarray(payload[name])
            if observed.shape != expected.shape or observed.dtype != expected.dtype:
                raise RuntimeError(f"temporary prediction checkpoint invalid: {name}")
    temporary.replace(path)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def model_contract() -> dict[str, dict[str, Any]]:
    return {
        "lightgbm": {"n_estimators": 100, "verbose": -1, "random_state": RANDOM_STATE},
        "xgboost": {
            "n_estimators": 100,
            "eval_metric": "logloss",
            "verbosity": 0,
            "random_state": RANDOM_STATE,
        },
        "catboost": {
            "iterations": 1000,
            "verbose": False,
            "random_seed": RANDOM_STATE,
            "allow_writing_files": False,
        },
        "hist_gradient_boosting": {"max_iter": 100, "random_state": RANDOM_STATE},
    }


def build_model(name: str) -> Any:
    contract = model_contract()[name]
    if name == "lightgbm":
        return lgb.LGBMClassifier(**contract)
    if name == "xgboost":
        return xgb.XGBClassifier(**contract)
    if name == "catboost":
        return CatBoostClassifier(**contract)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**contract)
    raise KeyError(name)


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def capped_building_frame(frame: Any, *, train_cap: int, validation_cap: int) -> Any:
    """Make the non-scientific validation data cap deterministic by building id."""
    validation = frame["building_id"] % 2 == 1
    train_ids = np.sort(frame.loc[~validation, "building_id"].unique())[:train_cap]
    validation_ids = np.sort(frame.loc[validation, "building_id"].unique())[
        :validation_cap
    ]
    return frame.loc[
        frame["building_id"].isin(np.concatenate([train_ids, validation_ids]))
    ].copy()


def prepare(
    *,
    feature_count: int,
    mode: str,
    validation_train_buildings: int | None,
    validation_buildings: int | None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if feature_count not in (17, 137):
        raise ValueError("feature_count must be 17 or 137")
    if progress is not None:
        progress("loading_m3_frame")
    frame = load_m3_frame()
    if mode == "validation":
        if not validation_train_buildings or not validation_buildings:
            raise ValueError(
                "validation mode requires both deterministic building caps"
            )
        frame = capped_building_frame(
            frame,
            train_cap=validation_train_buildings,
            validation_cap=validation_buildings,
        )
    if progress is not None:
        progress("validating_split")
    validation_mask = (frame["building_id"] % 2 == 1).to_numpy()
    train_buildings = int(frame.loc[~validation_mask, "building_id"].nunique())
    val_buildings = int(frame.loc[validation_mask, "building_id"].nunique())
    split = {
        "train_buildings": train_buildings,
        "validation_buildings": val_buildings,
        "train_rows": int((~validation_mask).sum()),
        "validation_rows": int(validation_mask.sum()),
    }
    if mode == "formal" and split != EXPECTED_FULL_SPLIT:
        raise RuntimeError(f"frozen M3 split mismatch: {split}")
    log(f"Building timestamp-merge feature tables ({feature_count} features)")
    if progress is not None:
        progress("building_feature_tables")
    train = add_value_change_features(
        frame.loc[~validation_mask], list(SHIFTS), value_change_regime="timestamp_merge"
    )
    validation = add_value_change_features(
        frame.loc[validation_mask], list(SHIFTS), value_change_regime="timestamp_merge"
    )
    value_cols = [name for name in train if name.startswith("lag_value_")]
    features = (
        list(BASELINE_FEATURE_COLS)
        if feature_count == 17
        else [*BASELINE_FEATURE_COLS, *value_cols]
    )
    if feature_count == 137 and len(features) != 137:
        raise RuntimeError(f"frozen 137-feature contract mismatch: {len(features)}")
    fit_index = downsample_indices(train["anomaly"])
    if progress is not None:
        progress("scaling_feature_matrices")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train.loc[fit_index, features])
    x_validation = scaler.transform(validation[features])
    labels = validation["anomaly"].to_numpy(dtype="int8", copy=True)
    meter = validation["meter"].to_numpy(dtype="int8", copy=True)
    site_id = validation["site_id"].to_numpy(dtype="int16", copy=True)
    row_identity = np.rec.fromarrays(
        [
            validation["building_id"].to_numpy(dtype="int32", copy=True),
            meter,
            validation["timestamp"].astype("int64").to_numpy(copy=True),
        ],
        names="building_id,meter,timestamp_ns",
    )
    if progress is not None:
        progress("feature_matrices_ready")
    return {
        "x_train": x_train,
        "y_train": train.loc[fit_index, "anomaly"],
        "x_validation": x_validation,
        "labels": labels,
        "meter": meter,
        "site_id": site_id,
        "row_identity": row_identity,
        "split": split,
        "feature_names": features,
        "fit_rows": int(len(fit_index)),
    }


def provenance(
    *, feature_count: int, mode: str, split: dict[str, int]
) -> dict[str, Any]:
    raw = ROOT / "data" / "raw" / "m3"
    inputs = [
        raw / "train.csv",
        raw / "bad_meter_readings.csv",
        raw / "building_metadata.csv",
    ]
    return {
        "schema_version": SCRIPT_SCHEMA,
        "script_sha256": sha256_file(Path(__file__)),
        "git_commit": git_sha(),
        "feature_count": feature_count,
        "mode": mode,
        "split": split,
        "model_contract": model_contract(),
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in inputs
        },
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "lightgbm",
                "xgboost",
                "catboost",
                "scikit-learn",
                "numpy",
                "pandas",
            )
        },
    }


def assert_provenance(expected: dict[str, Any], observed: dict[str, Any]) -> None:
    if expected != observed:
        raise RuntimeError(
            "checkpoint provenance differs from this run; refusing resume"
        )


def validate_mode_options(args: argparse.Namespace) -> None:
    """Reject ambiguous validation limits before any expensive input work."""
    caps = (args.validation_train_buildings, args.validation_buildings)
    if args.mode == "validation":
        if any(value is None or value <= 0 for value in caps):
            raise ValueError(
                "validation mode requires positive deterministic building caps"
            )
    elif any(value is not None for value in caps):
        raise ValueError("formal mode must not apply validation building caps")


def require_finalizable(predictions: dict[str, np.ndarray]) -> None:
    missing = set(MODEL_ORDER) - set(predictions)
    extra = set(predictions) - set(MODEL_ORDER)
    if missing or extra:
        raise RuntimeError(
            f"refusing finalization with missing={sorted(missing)}, extra={sorted(extra)}"
        )


def prediction_arrays(
    prepared: dict[str, Any], predictions: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        "anomaly": prepared["labels"],
        "meter": prepared["meter"],
        "site_id": prepared["site_id"],
        "row_identity": prepared["row_identity"],
        **{name: values.astype("float32") for name, values in predictions.items()},
    }


def load_prediction_checkpoint(
    path: Path, prepared: dict[str, Any]
) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with np.load(path) as payload:
        required = {"anomaly", "meter", "site_id", "row_identity"}
        if not required.issubset(payload.files):
            raise RuntimeError("prediction checkpoint is incomplete")
        expected = prediction_arrays(prepared, {})
        for name in required:
            if fingerprint(np.asarray(payload[name])) != fingerprint(expected[name]):
                raise RuntimeError(f"prediction checkpoint identity mismatch: {name}")
        return {
            name: np.asarray(payload[name])
            for name in payload.files
            if name in MODEL_ORDER
        }


def write_status(
    path: Path,
    *,
    stage: str,
    completed: int,
    total: int,
    started: float,
    unit_seconds: list[float],
    mode: str,
) -> None:
    mean_seconds = float(np.mean(unit_seconds)) if unit_seconds else None
    atomic_json(
        path,
        {
            "experiment": "m3_feature_count_ensemble_rebuild",
            "mode": mode,
            "stage": stage,
            "completed_units": completed,
            "total_units": total,
            "elapsed_seconds": time.perf_counter() - started,
            "mean_completed_unit_seconds": mean_seconds,
            "estimated_remaining_seconds": None
            if mean_seconds is None
            else mean_seconds * (total - completed),
            "updated_at_utc": utc_now(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-count", type=int, choices=(17, 137), required=True)
    parser.add_argument("--mode", choices=("validation", "formal"), required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--validation-train-buildings", type=int)
    parser.add_argument("--validation-buildings", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_mode_options(args)
    feature_tag = f"m3_{args.feature_count}_feature_ensemble_predictions_50_50.npz"
    if args.mode == "formal":
        output = args.out or PROC / feature_tag
    else:
        output = args.out or ROOT / "data" / "validation" / feature_tag
    checkpoint_dir = args.checkpoint_dir or output.parent / f"{output.stem}.checkpoints"
    prediction_checkpoint = checkpoint_dir / "predictions.npz"
    result_checkpoint = checkpoint_dir / "result.json"
    status_path = checkpoint_dir / "status.json"
    started = time.perf_counter()
    write_status(
        status_path,
        stage="starting_preparation",
        completed=0,
        total=len(MODEL_ORDER),
        started=started,
        unit_seconds=[],
        mode=args.mode,
    )

    def preparation_progress(stage: str) -> None:
        write_status(
            status_path,
            stage=stage,
            completed=0,
            total=len(MODEL_ORDER),
            started=started,
            unit_seconds=[],
            mode=args.mode,
        )

    prepared = prepare(
        feature_count=args.feature_count,
        mode=args.mode,
        validation_train_buildings=args.validation_train_buildings,
        validation_buildings=args.validation_buildings,
        progress=preparation_progress,
    )
    run_provenance = provenance(
        feature_count=args.feature_count, mode=args.mode, split=prepared["split"]
    )
    prior: dict[str, Any] | None = None
    if args.resume and result_checkpoint.is_file():
        prior = json.loads(result_checkpoint.read_text(encoding="utf-8"))
        assert_provenance(run_provenance, prior["provenance"])
    predictions = (
        load_prediction_checkpoint(prediction_checkpoint, prepared)
        if args.resume
        else {}
    )
    unit_seconds = list((prior or {}).get("unit_seconds", []))
    write_status(
        status_path,
        stage="prepared",
        completed=len(predictions),
        total=len(MODEL_ORDER),
        started=started,
        unit_seconds=unit_seconds,
        mode=args.mode,
    )
    for name in MODEL_ORDER:
        if name in predictions:
            log(f"Reusing completed {name} prediction checkpoint")
            continue
        write_status(
            status_path,
            stage=f"fitting_{name}",
            completed=len(predictions),
            total=len(MODEL_ORDER),
            started=started,
            unit_seconds=unit_seconds,
            mode=args.mode,
        )
        log(f"Fitting {name} ({args.feature_count} features)")
        t0 = time.perf_counter()
        model = build_model(name)
        train_values = (
            np.nan_to_num(prepared["x_train"], nan=0.0)
            if name == "hist_gradient_boosting"
            else prepared["x_train"]
        )
        validation_values = (
            np.nan_to_num(prepared["x_validation"], nan=0.0)
            if name == "hist_gradient_boosting"
            else prepared["x_validation"]
        )
        model.fit(train_values, prepared["y_train"])
        predictions[name] = model.predict_proba(validation_values)[:, 1]
        elapsed = time.perf_counter() - t0
        unit_seconds.append(elapsed)
        atomic_npz(prediction_checkpoint, prediction_arrays(prepared, predictions))
        record = {
            "complete": False,
            "provenance": run_provenance,
            "completed_models": list(predictions),
            "unit_seconds": unit_seconds,
            "metrics": {
                model_name: metrics(prepared["labels"], score)
                for model_name, score in predictions.items()
            },
            "prediction_checkpoint_sha256": sha256_file(prediction_checkpoint),
        }
        atomic_json(result_checkpoint, record)
        write_status(
            status_path,
            stage=f"completed_{name}",
            completed=len(predictions),
            total=len(MODEL_ORDER),
            started=started,
            unit_seconds=unit_seconds,
            mode=args.mode,
        )
        log(f"Completed {name} in {elapsed:.1f}s")
    require_finalizable(predictions)
    predictions["ensemble"] = np.mean(
        [predictions[name] for name in MODEL_ORDER], axis=0
    )
    output_arrays = prediction_arrays(prepared, predictions)
    atomic_npz(output, output_arrays)
    final = {
        "complete": True,
        "provenance": run_provenance,
        "completed_models": list(MODEL_ORDER),
        "unit_seconds": unit_seconds,
        "unit_p95_seconds": float(np.percentile(unit_seconds, 95)),
        "metrics": {
            name: metrics(prepared["labels"], score)
            for name, score in predictions.items()
        },
        "output": str(output),
        "output_sha256": sha256_file(output),
        "prediction_checkpoint_sha256": sha256_file(prediction_checkpoint),
        "feature_names": prepared["feature_names"],
        "fit_rows": prepared["fit_rows"],
    }
    atomic_json(result_checkpoint, final)
    write_status(
        status_path,
        stage="complete",
        completed=len(MODEL_ORDER),
        total=len(MODEL_ORDER),
        started=started,
        unit_seconds=unit_seconds,
        mode=args.mode,
    )
    log(f"Saved {output}")


if __name__ == "__main__":
    main()
