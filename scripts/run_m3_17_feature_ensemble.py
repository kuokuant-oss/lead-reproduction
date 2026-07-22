"""Run a matched 17-feature Tree Ensemble on the M3 50/50 building split."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
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
from run_m3_figure_observations import (
    MODEL_ORDER,
    array_fingerprint,
    evaluation_summary,
    fit_frozen_models,
    frozen_model_contract,
    predict_probability,
)


DEFAULT_OUTPUT = PROC / "m3_17_feature_ensemble.json"
DEFAULT_PREDICTIONS = PROC / "m3_17_feature_ensemble_predictions.npz"
REFERENCE_PREDICTIONS = PROC / "m3_figure_predictions_50_50.npz"


def grouped_site_metrics(
    site_id: np.ndarray,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value in range(16):
        mask = site_id == value
        output[str(value)] = {
            "rows": int(mask.sum()),
            "anomalies": int(y_true[mask].sum()),
            "anomaly_rate": float(y_true[mask].mean()),
            "models": {
                name: evaluation_summary(y_true[mask], pred[mask])
                for name, pred in predictions.items()
            },
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--predictions-out", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--reference-predictions",
        type=Path,
        default=REFERENCE_PREDICTIONS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    print("Loading frozen M3 frame", flush=True)
    frame = load_m3_frame()
    validation_mask = (frame["building_id"] % 2 == 1).to_numpy()
    validation_raw_index = np.flatnonzero(validation_mask).astype("int64")
    train_buildings = set(frame.loc[~validation_mask, "building_id"].unique().tolist())
    validation_buildings = set(
        frame.loc[validation_mask, "building_id"].unique().tolist()
    )
    if train_buildings & validation_buildings:
        raise RuntimeError("building overlap in frozen 50/50 split")

    # Preserve the canonical timestamp-merge row order.  Although this runner
    # consumes only the 17 baseline columns, downsampling is position-based and
    # must see the same ordered frame as the reported M3.1/M3.4 runs.
    print("Building canonical timestamp-merge tables", flush=True)
    train = add_value_change_features(
        frame.loc[~validation_mask],
        list(SHIFTS),
        value_change_regime="timestamp_merge",
    )
    validation = add_value_change_features(
        frame.loc[validation_mask],
        list(SHIFTS),
        value_change_regime="timestamp_merge",
    )
    del frame
    gc.collect()

    y_train = train["anomaly"].copy()
    y_validation = validation["anomaly"].to_numpy(dtype="int8", copy=True)
    validation_site_id = validation["site_id"].to_numpy(dtype="int16", copy=True)
    fit_index = downsample_indices(y_train)
    y_fit = y_train.loc[fit_index].copy()

    print("Scaling 17 baseline features", flush=True)
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(train.loc[fit_index, BASELINE_FEATURE_COLS])
    x_validation = scaler.transform(validation[BASELINE_FEATURE_COLS])
    del train, validation, y_train, scaler
    gc.collect()

    print("Fitting matched four-model 17-feature ensemble", flush=True)
    models, fit_seconds = fit_frozen_models(x_fit, y_fit)
    predictions = {
        name: predict_probability(name, models[name], x_validation)
        for name in MODEL_ORDER
    }
    predictions["ensemble"] = np.mean(
        [predictions[name] for name in MODEL_ORDER],
        axis=0,
    )

    metrics = {
        name: evaluation_summary(y_validation, pred)
        for name, pred in predictions.items()
    }

    # Timestamp feature construction reorders the legacy artifact, so validate
    # the LightGBM arm by its order-invariant pooled metrics rather than position.
    with np.load(args.reference_predictions) as reference:
        reference_lightgbm_metrics = evaluation_summary(
            np.asarray(reference["anomaly"]),
            np.asarray(reference["m3_1_lightgbm"]),
        )
    lightgbm_metric_deltas = {
        metric: metrics["lightgbm"][metric] - reference_lightgbm_metrics[metric]
        for metric in ("roc_auc", "pr_auc")
    }
    if any(abs(delta) > 1e-6 for delta in lightgbm_metric_deltas.values()):
        raise RuntimeError(
            "17-feature LightGBM failed to reproduce M3.1 pooled metrics: "
            f"{lightgbm_metric_deltas}"
        )

    site_metrics = grouped_site_metrics(
        validation_site_id,
        y_validation,
        predictions,
    )

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions_out,
        validation_raw_index=validation_raw_index,
        site_id=validation_site_id,
        anomaly=y_validation,
        **{name: pred.astype("float32") for name, pred in predictions.items()},
    )
    payload = {
        "schema_version": 1,
        "experiment": "m3_17_feature_tree_ensemble",
        "purpose": "matched feature-count comparison for M3 report figures",
        "split": {
            "name": "50_50_mod2",
            "validation_rule": "building_id % 2 == 1",
            "train_buildings": len(train_buildings),
            "validation_buildings": len(validation_buildings),
            "building_overlap": 0,
            "validation_rows": len(y_validation),
            "validation_raw_index_sha256": array_fingerprint(validation_raw_index),
        },
        "frozen_contract": {
            "features": list(BASELINE_FEATURE_COLS),
            "feature_count": len(BASELINE_FEATURE_COLS),
            "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
            "model_seed": RANDOM_STATE,
            "model_order": list(MODEL_ORDER),
            "model_contract": frozen_model_contract(),
            "ensemble": "equal_weight_probability_mean",
        },
        "fit": {
            "downsampled_rows": len(fit_index),
            "model_fit_seconds": fit_seconds,
        },
        "reference_check": {
            "artifact": str(args.reference_predictions.relative_to(ROOT)),
            "lightgbm_metric_deltas": lightgbm_metric_deltas,
        },
        "metrics": metrics,
        "slices": {"by_site_id": site_metrics},
        "artifacts": {
            "predictions": str(args.predictions_out.relative_to(ROOT)),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_with_provenance(
        args.out,
        payload,
        root=ROOT,
        provenance={
            "note": "Additive matched 17-feature ensemble; canonical M3 artifacts unchanged."
        },
    )
    print(
        "17-feature ensemble: "
        f"ROC-AUC={metrics['ensemble']['roc_auc']:.6f}, "
        f"PR-AUC={metrics['ensemble']['pr_auc']:.6f}",
        flush=True,
    )
    print(f"Saved {args.predictions_out}", flush=True)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
