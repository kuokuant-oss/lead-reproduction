"""Run the frozen M3 pipeline on a full 50/50 site-held-out split.

This is an additive experiment runner.  It preserves the canonical M3 data,
label, timestamp-merge feature, downsampling, scaling, model, ensemble, seed,
and threshold contracts.  The only experimental change is the split unit:
``site_id % 2 == 1`` is held out instead of ``building_id % 2 == 1``.

Unlike the bounded M5 comparison, this runner uses the complete M3-compatible
downsampled fit set and scores every held-out-site row.  It also records the
numeric inputs needed to render the same figure families as ``assets/m3``.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lead import (
    BASELINE_FEATURE_COLS,
    DOWNSAMPLE_SEEDS,
    PROC,
    RANDOM_STATE,
    ROOT,
    SHIFTS,
    add_value_change_features,
    downsample_indices,
    load_m3_frame,
    write_json_with_provenance,
)
from experiment_observability import host_environment, timing_protocol
from run_m3_figure_observations import (
    MODEL_ORDER,
    array_fingerprint,
    curve_summary,
    evaluation_summary,
    fit_frozen_models,
    frozen_model_contract,
    permutation_importance_observations,
    predict_probability,
    select_value_change_segment,
    stratified_sample_indices,
)


EXPERIMENT = "m3_full_site_transfer"
SITE_SPLIT_NAME = "site_id_mod2_50_50"
SITE_SPLIT_RULE = "validation sites are site_id % 2 == 1"
VALUE_CHANGE_REGIME = "timestamp_merge"
EXPECTED_SPLIT = {
    "train_sites": 8,
    "validation_sites": 8,
    "train_rows": 8_818_590,
    "validation_rows": 11_397_510,
}
PLOT_DATA_FAMILIES = (
    "confusion_matrix",
    "value_change_illustration",
    "roc_curves",
    "precision_recall_curves",
    "feature_engineering_17_vs_137",
    "model_permutation_importance",
    "ensemble_permutation_importance",
    "four_model_consensus_importance",
    "site_and_meter_slices",
)


def log(message: str) -> None:
    print(message, flush=True)


def resolve_output_path(path: Path) -> Path:
    """Resolve CLI output paths against the repository root."""
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def artifact_path(path: Path) -> str:
    """Return a repo-relative provenance path, or an absolute external path."""
    resolved = resolve_output_path(path)
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def site_transfer_mask(df: pd.DataFrame) -> np.ndarray:
    """Return the M5-compatible 50/50 held-out-site mask."""
    return (df["site_id"] % 2 == 1).to_numpy()


def site_split_summary(df: pd.DataFrame, mask_val: np.ndarray) -> dict[str, Any]:
    train_sites = set(int(value) for value in df.loc[~mask_val, "site_id"].unique())
    validation_sites = set(int(value) for value in df.loc[mask_val, "site_id"].unique())
    train_buildings = set(
        int(value) for value in df.loc[~mask_val, "building_id"].unique()
    )
    validation_buildings = set(
        int(value) for value in df.loc[mask_val, "building_id"].unique()
    )
    site_overlap = train_sites & validation_sites
    building_overlap = train_buildings & validation_buildings
    if site_overlap:
        raise AssertionError(f"site split overlap: {sorted(site_overlap)}")
    if building_overlap:
        raise AssertionError(
            f"site split has building overlap: {sorted(building_overlap)[:5]}"
        )
    summary = {
        "name": SITE_SPLIT_NAME,
        "rule": SITE_SPLIT_RULE,
        "unit_type": "site_id",
        "train_sites": len(train_sites),
        "validation_sites": len(validation_sites),
        "train_site_ids": sorted(train_sites),
        "validation_site_ids": sorted(validation_sites),
        "site_overlap": 0,
        "train_buildings": len(train_buildings),
        "validation_buildings": len(validation_buildings),
        "building_overlap": 0,
        "train_rows": int((~mask_val).sum()),
        "validation_rows": int(mask_val.sum()),
        "train_anomaly_rate": float(df.loc[~mask_val, "anomaly"].mean()),
        "validation_anomaly_rate": float(df.loc[mask_val, "anomaly"].mean()),
    }
    observed = {key: summary[key] for key in EXPECTED_SPLIT}
    if observed != EXPECTED_SPLIT:
        raise RuntimeError(
            f"Canonical M5-compatible site split drifted: {observed} "
            f"!= {EXPECTED_SPLIT}"
        )
    return summary


def safe_evaluation_summary(y_true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "n_rows": int(len(y_true)),
            "n_anomalies": int(y_true.sum()),
            "anomaly_rate": float(y_true.mean()) if len(y_true) else None,
            "roc_auc": None,
            "pr_auc": None,
            "threshold_0_5": None,
            "reason": "slice_requires_both_classes",
        }
    return {
        "n_rows": int(len(y_true)),
        "n_anomalies": int(y_true.sum()),
        "anomaly_rate": float(y_true.mean()),
        **evaluation_summary(y_true, pred),
    }


def grouped_metrics(
    group_values: np.ndarray,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value in sorted(np.unique(group_values).tolist()):
        mask = group_values == value
        output[str(value)] = {
            "group_value": int(value),
            "models": {
                name: safe_evaluation_summary(y_true[mask], pred[mask])
                for name, pred in predictions.items()
            },
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "m3_full_site_transfer.json",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=PROC / "m3_full_site_transfer_predictions.npz",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=PROC / "m3_full_site_transfer.checkpoint.json",
    )
    parser.add_argument("--permutation-sample-size", type=int, default=50_000)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="Recovery/debug option only; omit for the formal figure-data run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out = resolve_output_path(args.out)
    args.predictions_out = resolve_output_path(args.predictions_out)
    args.checkpoint_out = resolve_output_path(args.checkpoint_out)
    started = time.perf_counter()
    env = host_environment()

    log("Loading the frozen M3 frame")
    df = load_m3_frame()
    mask_val = site_transfer_mask(df)
    split = site_split_summary(df, mask_val)
    validation_raw_indices = np.flatnonzero(mask_val).astype("int64")

    log("Building train/held-out-site timestamp-merge feature tables")
    train_full = add_value_change_features(
        df.loc[~mask_val],
        list(SHIFTS),
        value_change_regime=VALUE_CHANGE_REGIME,
    )
    val_full = add_value_change_features(
        df.loc[mask_val],
        list(SHIFTS),
        value_change_regime=VALUE_CHANGE_REGIME,
    )
    value_cols = [column for column in train_full if column.startswith("lag_value_")]
    feature_cols = [*BASELINE_FEATURE_COLS, *value_cols]
    if len(value_cols) != 120 or len(feature_cols) != 137:
        raise RuntimeError(
            f"Frozen feature contract mismatch: {len(value_cols)=}, "
            f"{len(feature_cols)=}"
        )

    illustration = select_value_change_segment(val_full)
    y_train = train_full["anomaly"].copy()
    y_val = val_full["anomaly"].to_numpy(dtype="int8", copy=True)
    val_site_id = val_full["site_id"].to_numpy(dtype="int16", copy=True)
    val_building_id = val_full["building_id"].to_numpy(dtype="int16", copy=True)
    val_meter = val_full["meter"].to_numpy(dtype="int8", copy=True)
    ds_idx = downsample_indices(y_train)
    y_fit = y_train.loc[ds_idx].copy()

    log("Fitting the frozen 17-feature LightGBM comparison")
    baseline_scaler = StandardScaler()
    x_train_17 = baseline_scaler.fit_transform(
        train_full.loc[ds_idx, BASELINE_FEATURE_COLS]
    )
    x_val_17 = baseline_scaler.transform(val_full[BASELINE_FEATURE_COLS])
    baseline_model = lgb.LGBMClassifier(
        n_estimators=100,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    baseline_started = time.perf_counter()
    baseline_model.fit(x_train_17, y_fit)
    baseline_fit_seconds = time.perf_counter() - baseline_started
    baseline_pred = baseline_model.predict_proba(x_val_17)[:, 1]
    del x_train_17, x_val_17, baseline_model, baseline_scaler
    gc.collect()

    log("Fitting the frozen 137-feature four-model ensemble")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[ds_idx, feature_cols])
    x_val = scaler.transform(val_full[feature_cols])
    del df, train_full, val_full, y_train, scaler
    gc.collect()
    models, fit_seconds = fit_frozen_models(x_train, y_fit)
    predictions = {
        name: predict_probability(name, models[name], x_val) for name in MODEL_ORDER
    }
    predictions["ensemble"] = np.mean(
        [predictions[name] for name in MODEL_ORDER],
        axis=0,
    )

    metrics = {
        "m3_1_lightgbm": evaluation_summary(y_val, baseline_pred),
        **{
            name: evaluation_summary(y_val, prediction)
            for name, prediction in predictions.items()
        },
    }
    curves = {
        "m3_1_lightgbm": curve_summary(y_val, baseline_pred),
        **{
            name: curve_summary(y_val, prediction)
            for name, prediction in predictions.items()
        },
    }
    slices = {
        "by_site_id": grouped_metrics(val_site_id, y_val, predictions),
        "by_meter": grouped_metrics(val_meter, y_val, predictions),
    }

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions_out,
        validation_raw_index=validation_raw_indices,
        site_id=val_site_id,
        building_id=val_building_id,
        meter=val_meter,
        anomaly=y_val,
        m3_1_lightgbm=baseline_pred.astype("float32"),
        **{
            name: prediction.astype("float32")
            for name, prediction in predictions.items()
        },
    )

    checkpoint = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "complete": False,
        "stage": "full_predictions_and_curves_complete",
        "split": split,
        "metrics": metrics,
        "curves": curves,
        "slices": slices,
        "artifacts": {"predictions": artifact_path(args.predictions_out)},
    }
    write_json_with_provenance(
        args.checkpoint_out,
        checkpoint,
        root=ROOT,
        provenance={"note": "Recovery checkpoint; not a final report artifact."},
    )
    log(f"Saved recovery checkpoint: {args.checkpoint_out}")

    if args.skip_permutation:
        permutation: dict[str, Any] = {
            "status": "skipped",
            "reason": "--skip-permutation",
        }
    else:
        sample_idx = stratified_sample_indices(
            y_val,
            sample_size=args.permutation_sample_size,
            seed=RANDOM_STATE,
        )
        permutation = {
            "status": "completed",
            "sample_size": int(len(sample_idx)),
            "sample_validation_position_sha256": array_fingerprint(sample_idx),
            "sample_anomaly_rate": float(y_val[sample_idx].mean()),
            "repeats": int(args.permutation_repeats),
            "scoring": ["roc_auc", "average_precision"],
            **permutation_importance_observations(
                models,
                x_val[sample_idx].copy(),
                y_val[sample_idx],
                feature_cols,
                repeats=args.permutation_repeats,
                seed=RANDOM_STATE,
            ),
        }

    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "scope": (
            "Frozen M3 full-data pipeline with only the split unit changed "
            "from building_id to site_id."
        ),
        "frozen_contract": {
            "data_loader": "lead.load_m3_frame",
            "split": SITE_SPLIT_NAME,
            "split_rule": SITE_SPLIT_RULE,
            "only_pipeline_change": "building-held-out_to_site-held-out",
            "value_change_regime": VALUE_CHANGE_REGIME,
            "baseline_features": list(BASELINE_FEATURE_COLS),
            "value_change_shifts": list(SHIFTS),
            "feature_count": len(feature_cols),
            "downsampling": "lead.downsample_indices",
            "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
            "scaler": "sklearn.preprocessing.StandardScaler",
            "model_seed": RANDOM_STATE,
            "model_order": list(MODEL_ORDER),
            "model_contract": frozen_model_contract(),
            "ensemble": "equal_weight_probability_mean",
            "threshold": 0.5,
            "fit_budget": "complete_m3_compatible_downsampled_fit_set",
            "scoring_budget": "all_held_out_site_rows",
            "src_lead_modified": False,
            "existing_m3_runners_modified": False,
        },
        "split": {
            **split,
            "validation_raw_index_sha256": array_fingerprint(validation_raw_indices),
        },
        "fit": {
            "downsampled_rows": int(len(ds_idx)),
            "fit_anomaly_rate": float(y_fit.mean()),
            "m3_1_lightgbm_fit_seconds": float(baseline_fit_seconds),
            "model_fit_seconds": fit_seconds,
            "ensemble_fit_seconds_definition": "sum_of_four_component_fit_seconds",
            "ensemble_fit_seconds": float(sum(fit_seconds.values())),
        },
        "metrics": metrics,
        "curves": curves,
        "slices": slices,
        "permutation_importance": permutation,
        "value_change_illustration": illustration,
        "plot_data_contract": {
            "reference_asset_family": "docs/reports/assets/m3",
            "families_recorded": list(PLOT_DATA_FAMILIES),
            "workflow_figure_requires_numeric_data": False,
            "predictions_enable_additional_slices": True,
        },
        "artifacts": {
            "predictions": artifact_path(args.predictions_out),
            "checkpoint": artifact_path(args.checkpoint_out),
        },
        "environment": env,
        "timing_protocol": timing_protocol(),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    command = (
        ".\\.venv\\Scripts\\python.exe "
        "scripts/run_m3_full_site_transfer.py "
        f"--out {artifact_path(args.out)} "
        f"--predictions-out {artifact_path(args.predictions_out)} "
        f"--checkpoint-out {artifact_path(args.checkpoint_out)} "
        f"--permutation-sample-size {args.permutation_sample_size} "
        f"--permutation-repeats {args.permutation_repeats}"
    )
    write_json_with_provenance(
        args.out,
        payload,
        root=ROOT,
        provenance={
            "command": command,
            "note": (
                "Full M3 pipeline; the only experimental change is the "
                "building-to-site held-out split."
            ),
        },
    )
    log(f"Saved {args.out} in {payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
