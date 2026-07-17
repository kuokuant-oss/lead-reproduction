"""Run additive M6 site-transfer experiment cells on the frozen M3 pipeline.

The runner supports A2 reverse transfer, A3 grouped folds, A4 LOSO, A5 in-site
oracles, B1 stratified meter learning-curve cells, and B2 matched-anomaly cells.
It imports the accepted M3 feature/model helpers without changing them. Use
``--prepare-only`` to inspect and freeze a manifest without fitting models.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

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
from m6_site_transfer_protocol import (
    A3_SITE_FOLDS,
    array_fingerprint,
    grouped_support,
    in_site_oracle_masks,
    location_split_masks,
    matched_anomaly_fit_indices,
    meter_budget_manifest,
    meter_manifest_mask,
    source_meter_units,
    source_building_calibration_masks,
    split_manifest,
)
from run_m3_figure_observations import (
    MODEL_ORDER,
    curve_summary,
    evaluation_summary,
    fit_frozen_models,
    frozen_model_contract,
    predict_probability,
)
from run_m3_full_site_transfer import grouped_metrics


EXPERIMENT = "m6_site_transfer"
VALUE_CHANGE_REGIME = "timestamp_merge"
SUPPORTED_EXPERIMENTS = ("a2", "a3", "a4", "a5", "b1", "b2")

# B1 sweeps a meter budget over whatever the base split calls "source".
# a1/a2 hold out sites; a0odd/a0even hold out buildings and keep every site in
# training, which is what makes a site-seen learning curve possible. Running
# both a0 folds gives every building an out-of-fold prediction, so the two can
# be unioned and scored over a whole site -- the rows B1's own curves cover.
B1_DIRECTION_SPLITS = {
    "a1": "a1_even_to_odd",
    "a2": "a2_odd_to_even",
    "a0odd": "a0_building_mod2",
    "a0even": "a0_building_mod2_inverse",
}
PLOT_FAMILIES = (
    "pooled_roc_pr",
    "split_robustness",
    "per_site_forest",
    "site_by_model_heatmap",
    "pr_auc_vs_prevalence",
    "threshold_0_5_confusion",
    "source_calibrated_recall_0_90",
    "paired_in_site_oracle",
    "training_meter_learning_curve",
    "matched_anomaly_support",
    "runtime_and_matrix_cost",
    "score_distributions",
)


def log(message: str) -> None:
    print(message, flush=True)


def resolve_output_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def artifact_path(path: Path) -> str:
    resolved = resolve_output_path(path)
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def metric_distribution(values: list[float]) -> dict[str, float | int | None]:
    """Return macro distribution statistics without hiding unscorable sites."""
    if not values:
        return {
            "n_scorable": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype="float64")
    return {
        "n_scorable": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def macro_site_summary(site_slices: dict[str, Any]) -> dict[str, Any]:
    """Summarize site-level metrics for forest/robustness plots."""
    models = sorted(
        {model for site in site_slices.values() for model in site.get("models", {})}
    )
    output: dict[str, Any] = {}
    for model in models:
        roc_values = []
        pr_values = []
        unscorable = 0
        for site in site_slices.values():
            metrics = site["models"][model]
            if metrics.get("roc_auc") is None or metrics.get("pr_auc") is None:
                unscorable += 1
                continue
            roc_values.append(float(metrics["roc_auc"]))
            pr_values.append(float(metrics["pr_auc"]))
        output[model] = {
            "sites_total": int(len(site_slices)),
            "sites_unscorable": int(unscorable),
            "roc_auc": metric_distribution(roc_values),
            "pr_auc": metric_distribution(pr_values),
        }
    return output


def score_histograms(
    y_true: np.ndarray,
    group_values: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    bins: int = 100,
) -> dict[str, Any]:
    """Store compact score distributions while exact scores remain in NPZ."""
    edges = np.linspace(0.0, 1.0, bins + 1)

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        return {
            model: {
                "normal": np.histogram(prediction[mask & (y_true == 0)], edges)[
                    0
                ].tolist(),
                "anomaly": np.histogram(prediction[mask & (y_true == 1)], edges)[
                    0
                ].tolist(),
            }
            for model, prediction in predictions.items()
        }

    overall_mask = np.ones(len(y_true), dtype=bool)
    return {
        "bin_edges": edges.tolist(),
        "overall": summarize(overall_mask),
        "by_site_id": {
            str(int(site)): summarize(group_values == site)
            for site in sorted(np.unique(group_values).tolist())
        },
    }


def sampled_nan_fraction(
    values: np.ndarray, *, max_rows: int = 100_000
) -> dict[str, Any]:
    """Estimate matrix NaN fraction without an extra full-matrix scan."""
    if len(values) <= max_rows:
        sample = values
        positions = np.arange(len(values), dtype="int64")
    else:
        positions = np.linspace(0, len(values) - 1, max_rows, dtype="int64")
        sample = values[positions]
    return {
        "value": float(np.isnan(sample).mean()),
        "sample_rows": int(len(sample)),
        "sample_position_sha256": array_fingerprint(positions),
    }


def fixed_recall_threshold(
    y_true: np.ndarray,
    prediction: np.ndarray,
    *,
    target_recall: float = 0.90,
) -> float:
    """Choose the highest source-calibration threshold reaching target recall."""
    positive_scores = np.asarray(prediction)[np.asarray(y_true) == 1]
    if len(positive_scores) == 0:
        raise ValueError("fixed recall calibration requires anomaly rows")
    required = int(np.ceil(target_recall * len(positive_scores)))
    return float(np.sort(positive_scores)[::-1][required - 1])


def evaluation_at_threshold(
    y_true: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predicted = np.asarray(prediction) >= threshold
    truth = np.asarray(y_true) == 1
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    tn = int(np.sum(~truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive_rate": float(predicted.mean()),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def parse_meter_budget(value: str, available: int) -> int:
    if value.lower() == "all":
        return int(available)
    try:
        budget = int(value)
    except ValueError as error:
        raise ValueError("meter budget must be a positive integer or 'all'") from error
    if budget <= 0:
        raise ValueError("meter budget must be positive")
    return budget


def experiment_stem(args: argparse.Namespace) -> str:
    if args.experiment == "a2":
        suffix = "a2_odd_to_even"
    elif args.experiment == "a3":
        suffix = f"a3_fold{args.fold}"
    elif args.experiment == "a4":
        suffix = f"a4_site{args.site_id}"
    elif args.experiment == "a5":
        suffix = f"a5_oracle_site{args.site_id}"
    elif args.experiment == "b1":
        budget = str(args.meter_budget).lower()
        suffix = f"b1_{args.direction}_meters{budget}_seed{args.selection_seed}"
    else:
        suffix = (
            f"b2_{args.base_split}_pos{args.positive_budget}_seed{args.selection_seed}"
        )
    if args.source_calibration:
        suffix += "_sourcecal"
    return f"m6_site_transfer_{suffix}"


def validate_args(args: argparse.Namespace) -> None:
    if args.experiment == "a3" and args.fold not in A3_SITE_FOLDS:
        raise ValueError(
            f"--fold is required and must be one of {sorted(A3_SITE_FOLDS)}"
        )
    if args.experiment in {"a4", "a5"} and args.site_id is None:
        raise ValueError(f"--site-id is required for {args.experiment}")
    if args.experiment == "b1" and args.meter_budget is None:
        raise ValueError("--meter-budget is required for b1")
    if args.experiment == "b2" and args.positive_budget is None:
        raise ValueError("--positive-budget is required for b2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=SUPPORTED_EXPERIMENTS, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--site-id", type=int)
    parser.add_argument("--direction", choices=tuple(B1_DIRECTION_SPLITS), default="a1")
    parser.add_argument("--meter-budget")
    parser.add_argument("--base-split", choices=("a0", "a1", "a2"), default="a1")
    parser.add_argument("--positive-budget", type=int)
    parser.add_argument("--selection-seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--source-calibration",
        action="store_true",
        help=(
            "Add a source-only building-disjoint calibration split and store "
            "deployable fixed-recall 0.90 operating points."
        ),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument(
        "--manifest-in",
        type=Path,
        help="Prepared manifest to verify before a formal model run.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the split/sampling manifest and stop before feature/model work.",
    )
    return parser.parse_args()


def build_experiment_masks(
    frame: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any] | None]:
    """Build the exact raw-row masks and optional B1 selection manifest."""
    selection: dict[str, Any] | None = None
    if args.experiment == "a2":
        train, test, manifest = location_split_masks(frame, "a2_odd_to_even")
    elif args.experiment == "a3":
        train, test, manifest = location_split_masks(
            frame,
            "a3_group4",
            fold=args.fold,
        )
    elif args.experiment == "a4":
        train, test, manifest = location_split_masks(
            frame,
            "a4_loso",
            site_id=args.site_id,
        )
    elif args.experiment == "a5":
        train, test, manifest = in_site_oracle_masks(frame, int(args.site_id))
    elif args.experiment == "b1":
        split_name = B1_DIRECTION_SPLITS[args.direction]
        source, test, base_manifest = location_split_masks(frame, split_name)
        available = len(source_meter_units(frame, source))
        budget = parse_meter_budget(args.meter_budget, available)
        selection = meter_budget_manifest(
            frame,
            source,
            budget=budget,
            seed=args.selection_seed,
        )
        train = source & meter_manifest_mask(frame, selection)
        # The a0 folds hold out buildings, not sites, so every site is on both
        # sides by construction and the site-disjoint assert does not apply.
        unit_type = base_manifest["unit_type"]
        manifest = split_manifest(
            frame,
            train,
            test,
            name="b1_training_meter_curve",
            # Wording is byte-identical to the already-completed B1 cells so a
            # re-run reproduces their manifests exactly. It stays accurate for
            # the a0 folds: meter_budget_manifest stratifies by source site
            # there too, across all 16.
            rule=(
                f"{split_name}; source-site-stratified nested meter budget "
                f"{budget}, seed {args.selection_seed}"
            ),
            unit_type=unit_type,
            require_site_disjoint=unit_type == "site_id",
        )
        manifest["base_location_split"] = base_manifest
        manifest["excluded_source_rows"] = int((source & ~train).sum())
    else:
        split_names = {
            "a0": "a0_building_mod2",
            "a1": "a1_even_to_odd",
            "a2": "a2_odd_to_even",
        }
        train, test, manifest = location_split_masks(
            frame,
            split_names[args.base_split],
        )
        available_anomalies = int(frame.loc[train, "anomaly"].sum())
        if args.positive_budget <= 0 or args.positive_budget > available_anomalies:
            raise ValueError(
                f"B2 positive budget must be in 1..{available_anomalies} "
                f"for {args.base_split}"
            )
        manifest["b2_positive_budget"] = int(args.positive_budget)
        manifest["b2_available_source_anomalies"] = available_anomalies
        manifest["b2_selection_seed"] = int(args.selection_seed)
    return train, test, manifest, selection


def prepare_payload(
    args: argparse.Namespace,
    split: dict[str, Any],
    selection: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "cell": args.experiment,
        "variant": "source_calibrated" if args.source_calibration else "canonical",
        "status": "manifest_prepared",
        "split": split,
        "selection": selection,
        "frozen_contract": {
            "source": "M3/M4 timestamp_merge pipeline",
            "src_lead_modified": False,
            "existing_m3_runners_modified": False,
        },
    }


def verify_prepared_manifest(
    path: Path,
    split: dict[str, Any],
    selection: dict[str, Any] | None,
) -> str:
    """Require the formal cell to match its pre-model manifest exactly."""
    raw = path.read_bytes()
    prepared = json.loads(raw.decode("utf-8"))
    if prepared.get("status") != "manifest_prepared":
        raise ValueError(f"{path} is not an M6 prepared manifest")
    if prepared.get("split") != split or prepared.get("selection") != selection:
        raise AssertionError("formal M6 cell drifted from its prepared manifest")
    return hashlib.sha256(raw).hexdigest()


def fit_index_for_cell(
    y_train: pd.Series,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if args.experiment == "b2":
        return matched_anomaly_fit_indices(
            y_train,
            positive_budget=int(args.positive_budget),
            selection_seed=args.selection_seed,
        )
    fit_index = downsample_indices(y_train)
    unique_positive = y_train.index[y_train == 1].to_numpy()
    return fit_index, {
        "selection": "lead.downsample_indices",
        "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
        "unique_anomaly_rows": int(len(unique_positive)),
        "positive_blocks": 2,
        "negative_blocks": 2,
        "fit_rows": int(len(fit_index)),
        "effective_fit_anomaly_rate": 0.5,
        "fit_index_sha256": array_fingerprint(fit_index),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    stem = experiment_stem(args)
    default_name = f"{stem}_manifest.json" if args.prepare_only else f"{stem}.json"
    args.out = resolve_output_path(args.out or (PROC / default_name))
    args.predictions_out = resolve_output_path(
        args.predictions_out or (PROC / f"{stem}_predictions.npz")
    )
    args.manifest_in = (
        resolve_output_path(args.manifest_in) if args.manifest_in is not None else None
    )
    started = time.perf_counter()
    timing_breakdown: dict[str, Any] = {}

    log("Loading frozen M3 frame for M6 manifest")
    load_started = time.perf_counter()
    frame = load_m3_frame().copy()
    frame["_raw_index"] = np.arange(len(frame), dtype="int64")
    timing_breakdown["load_frame_seconds"] = time.perf_counter() - load_started
    manifest_started = time.perf_counter()
    train_mask, test_mask, split, selection = build_experiment_masks(frame, args)
    calibration_mask = np.zeros(len(frame), dtype=bool)
    if args.source_calibration:
        train_mask, calibration_mask, calibration_manifest = (
            source_building_calibration_masks(frame, train_mask)
        )
        split["source_calibration"] = calibration_manifest
        if args.experiment == "b2":
            model_fit_anomalies = int(frame.loc[train_mask, "anomaly"].sum())
            if args.positive_budget > model_fit_anomalies:
                raise ValueError(
                    "B2 positive budget exceeds anomalies remaining after "
                    f"source calibration: {args.positive_budget} > "
                    f"{model_fit_anomalies}"
                )
            split["b2_model_fit_anomalies_after_calibration"] = model_fit_anomalies
    support = {
        "train_by_site_id": grouped_support(frame, train_mask),
        "test_by_site_id": grouped_support(frame, test_mask),
    }
    if args.source_calibration:
        support["calibration_by_site_id"] = grouped_support(
            frame,
            calibration_mask,
        )
    timing_breakdown["manifest_and_support_seconds"] = (
        time.perf_counter() - manifest_started
    )
    payload = prepare_payload(args, split, selection)
    payload["support"] = support
    payload["timing_breakdown"] = timing_breakdown
    if args.prepare_only:
        payload["elapsed_seconds"] = float(time.perf_counter() - started)
        write_json_with_provenance(
            args.out,
            payload,
            root=ROOT,
            provenance={
                "note": "M6 manifest only; no feature engineering or model fitting ran."
            },
        )
        log(f"Saved manifest without fitting: {args.out}")
        return
    if args.manifest_in is None:
        manifest_sha256 = None
        log("WARNING: formal run has no --manifest-in freeze check")
    else:
        manifest_sha256 = verify_prepared_manifest(
            args.manifest_in,
            split,
            selection,
        )
        log(f"Verified prepared manifest: {args.manifest_in}")

    log("Building separate source/test timestamp-merge feature tables")
    train_feature_started = time.perf_counter()
    train_full = add_value_change_features(
        frame.loc[train_mask],
        list(SHIFTS),
        value_change_regime=VALUE_CHANGE_REGIME,
    )
    timing_breakdown["train_feature_seconds"] = (
        time.perf_counter() - train_feature_started
    )
    test_feature_started = time.perf_counter()
    test_full = add_value_change_features(
        frame.loc[test_mask],
        list(SHIFTS),
        value_change_regime=VALUE_CHANGE_REGIME,
    )
    timing_breakdown["test_feature_seconds"] = (
        time.perf_counter() - test_feature_started
    )
    calibration_full = None
    if args.source_calibration:
        calibration_feature_started = time.perf_counter()
        calibration_full = add_value_change_features(
            frame.loc[calibration_mask],
            list(SHIFTS),
            value_change_regime=VALUE_CHANGE_REGIME,
        )
        timing_breakdown["calibration_feature_seconds"] = (
            time.perf_counter() - calibration_feature_started
        )
    value_cols = [column for column in train_full if column.startswith("lag_value_")]
    feature_cols = [*BASELINE_FEATURE_COLS, *value_cols]
    if len(value_cols) != 120 or len(feature_cols) != 137:
        raise RuntimeError(
            f"Frozen feature contract mismatch: {len(value_cols)=}, "
            f"{len(feature_cols)=}"
        )
    y_train = train_full["anomaly"].copy()
    y_test = test_full["anomaly"].to_numpy(dtype="int8", copy=True)
    y_calibration = (
        None
        if calibration_full is None
        else calibration_full["anomaly"].to_numpy(dtype="int8", copy=True)
    )
    if y_train.nunique() < 2:
        raise RuntimeError("M6 source fit requires both anomaly classes")
    if len(np.unique(y_test)) < 2:
        raise RuntimeError("M6 test cell is unscorable because it has one class")
    if y_calibration is not None and len(np.unique(y_calibration)) < 2:
        raise RuntimeError("M6 source calibration requires both anomaly classes")
    fit_index, fit_selection = fit_index_for_cell(y_train, args)
    y_fit = y_train.loc[fit_index].copy()

    log("Scaling frozen 137-feature matrices")
    scaling_started = time.perf_counter()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[fit_index, feature_cols])
    x_test = scaler.transform(test_full[feature_cols])
    x_calibration = (
        None
        if calibration_full is None
        else scaler.transform(calibration_full[feature_cols])
    )
    timing_breakdown["scaling_seconds"] = time.perf_counter() - scaling_started
    matrix_profile = {
        "fit_shape": [int(value) for value in x_train.shape],
        "test_shape": [int(value) for value in x_test.shape],
        "dtype": str(x_train.dtype),
        "fit_matrix_bytes": int(x_train.nbytes),
        "test_matrix_bytes": int(x_test.nbytes),
        "fit_nan_fraction_sample": sampled_nan_fraction(x_train),
        "test_nan_fraction_sample": sampled_nan_fraction(x_test),
    }
    if x_calibration is not None:
        matrix_profile.update(
            {
                "calibration_shape": [int(value) for value in x_calibration.shape],
                "calibration_matrix_bytes": int(x_calibration.nbytes),
                "calibration_nan_fraction_sample": sampled_nan_fraction(x_calibration),
            }
        )
    test_raw_index = test_full["_raw_index"].to_numpy(dtype="int64", copy=True)
    test_site_id = test_full["site_id"].to_numpy(dtype="int16", copy=True)
    test_building_id = test_full["building_id"].to_numpy(dtype="int16", copy=True)
    test_meter = test_full["meter"].to_numpy(dtype="int8", copy=True)
    test_timestamp_ns = (
        pd.to_datetime(test_full["timestamp"]).astype("int64").to_numpy(copy=True)
    )
    calibration_arrays = None
    if calibration_full is not None:
        calibration_arrays = {
            "raw_index": calibration_full["_raw_index"].to_numpy(
                dtype="int64", copy=True
            ),
            "timestamp_ns": pd.to_datetime(calibration_full["timestamp"])
            .astype("int64")
            .to_numpy(copy=True),
            "site_id": calibration_full["site_id"].to_numpy(dtype="int16", copy=True),
            "building_id": calibration_full["building_id"].to_numpy(
                dtype="int16", copy=True
            ),
            "meter": calibration_full["meter"].to_numpy(dtype="int8", copy=True),
        }
    del frame, train_full, test_full, calibration_full, y_train, scaler
    gc.collect()

    log("Fitting frozen M3 four-model family")
    models, fit_seconds = fit_frozen_models(x_train, y_fit)
    prediction_seconds: dict[str, float] = {}
    predictions = {}
    for name in MODEL_ORDER:
        prediction_started = time.perf_counter()
        predictions[name] = predict_probability(name, models[name], x_test)
        prediction_seconds[name] = time.perf_counter() - prediction_started
    ensemble_started = time.perf_counter()
    predictions["ensemble"] = np.mean(
        [predictions[name] for name in MODEL_ORDER],
        axis=0,
    )
    ensemble_combine_seconds = time.perf_counter() - ensemble_started
    calibration_predictions: dict[str, np.ndarray] | None = None
    calibration_prediction_seconds: dict[str, float] | None = None
    if x_calibration is not None:
        calibration_predictions = {}
        calibration_prediction_seconds = {}
        for name in MODEL_ORDER:
            prediction_started = time.perf_counter()
            calibration_predictions[name] = predict_probability(
                name,
                models[name],
                x_calibration,
            )
            calibration_prediction_seconds[name] = (
                time.perf_counter() - prediction_started
            )
        ensemble_started = time.perf_counter()
        calibration_predictions["ensemble"] = np.mean(
            [calibration_predictions[name] for name in MODEL_ORDER],
            axis=0,
        )
        calibration_prediction_seconds["ensemble_probability_combine"] = (
            time.perf_counter() - ensemble_started
        )
    metrics_started = time.perf_counter()
    metrics = {
        name: evaluation_summary(y_test, prediction)
        for name, prediction in predictions.items()
    }
    curves = {
        name: curve_summary(y_test, prediction)
        for name, prediction in predictions.items()
    }
    slices = {
        "by_site_id": grouped_metrics(test_site_id, y_test, predictions),
        "by_meter": grouped_metrics(test_meter, y_test, predictions),
    }
    macro_metrics = macro_site_summary(slices["by_site_id"])
    histograms = score_histograms(y_test, test_site_id, predictions)
    operating_points: dict[str, Any] = {"threshold_0_5": "stored_in_metrics"}
    if calibration_predictions is not None and y_calibration is not None:
        operating_points["source_calibrated_recall_0_90"] = {}
        for name, prediction in predictions.items():
            threshold = fixed_recall_threshold(
                y_calibration,
                calibration_predictions[name],
                target_recall=0.90,
            )
            operating_points["source_calibrated_recall_0_90"][name] = {
                "threshold_source": "source_building_mod5_calibration",
                "target_recall": 0.90,
                "threshold": threshold,
                "calibration": evaluation_at_threshold(
                    y_calibration,
                    calibration_predictions[name],
                    threshold,
                ),
                "test": evaluation_at_threshold(y_test, prediction, threshold),
                "test_by_site_id": {
                    str(int(site)): evaluation_at_threshold(
                        y_test[test_site_id == site],
                        prediction[test_site_id == site],
                        threshold,
                    )
                    for site in sorted(np.unique(test_site_id).tolist())
                },
            }
    timing_breakdown["metrics_curves_slices_seconds"] = (
        time.perf_counter() - metrics_started
    )

    serialization_started = time.perf_counter()
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    prediction_artifact = {
        "validation_raw_index": test_raw_index,
        "timestamp_ns": test_timestamp_ns,
        "site_id": test_site_id,
        "building_id": test_building_id,
        "meter": test_meter,
        "anomaly": y_test,
        **{
            name: prediction.astype("float32")
            for name, prediction in predictions.items()
        },
    }
    if (
        calibration_predictions is not None
        and calibration_arrays is not None
        and y_calibration is not None
    ):
        prediction_artifact.update(
            {
                "calibration_raw_index": calibration_arrays["raw_index"],
                "calibration_timestamp_ns": calibration_arrays["timestamp_ns"],
                "calibration_site_id": calibration_arrays["site_id"],
                "calibration_building_id": calibration_arrays["building_id"],
                "calibration_meter": calibration_arrays["meter"],
                "calibration_anomaly": y_calibration,
                **{
                    f"calibration_{name}": prediction.astype("float32")
                    for name, prediction in calibration_predictions.items()
                },
            }
        )
    np.savez_compressed(args.predictions_out, **prediction_artifact)
    timing_breakdown["prediction_npz_seconds"] = (
        time.perf_counter() - serialization_started
    )

    payload.update(
        {
            "status": "completed",
            "scope": (
                "Additive M6 experiment cell using the frozen M3/M4 pipeline; "
                "only the declared split or support selection changes."
            ),
            "frozen_contract": {
                "data_loader": "lead.load_m3_frame",
                "value_change_regime": VALUE_CHANGE_REGIME,
                "feature_count": len(feature_cols),
                "feature_names": feature_cols,
                "downsampling": "lead.downsample_indices-compatible",
                "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
                "scaler": "sklearn.preprocessing.StandardScaler",
                "model_seed": RANDOM_STATE,
                "model_order": list(MODEL_ORDER),
                "model_contract": frozen_model_contract(),
                "ensemble": "equal_weight_probability_mean",
                "canonical_operating_point": 0.5,
                "scoring_budget": "all_declared_test_rows",
                "src_lead_modified": False,
                "existing_m3_runners_modified": False,
            },
            "split": {
                **split,
                "test_raw_index_sha256": array_fingerprint(test_raw_index),
            },
            "fit": {
                **fit_selection,
                "model_fit_seconds": fit_seconds,
                "ensemble_fit_seconds_definition": "sum_of_four_component_fit_seconds",
                "ensemble_fit_seconds": float(sum(fit_seconds.values())),
            },
            "prediction_seconds": {
                **prediction_seconds,
                "ensemble_probability_combine": float(ensemble_combine_seconds),
                "source_calibration": calibration_prediction_seconds,
            },
            "matrix_profile": matrix_profile,
            "metrics": metrics,
            "curves": curves,
            "slices": slices,
            "macro_site_metrics": macro_metrics,
            "score_histograms": histograms,
            "operating_points": operating_points,
            "support": support,
            "plot_data_contract": {
                "schema_version": 1,
                "families_recorded": list(PLOT_FAMILIES),
                "exact_predictions_available": True,
                "exact_row_identity_columns": [
                    "validation_raw_index",
                    "timestamp_ns",
                    "site_id",
                    "building_id",
                    "meter",
                    "anomaly",
                ],
                "calibration_identity_columns": (
                    [
                        "calibration_raw_index",
                        "calibration_timestamp_ns",
                        "calibration_site_id",
                        "calibration_building_id",
                        "calibration_meter",
                        "calibration_anomaly",
                    ]
                    if args.source_calibration
                    else []
                ),
                "threshold_free_curves_reconstructable": True,
                "threshold_0_5_reconstructable": True,
                "source_calibrated_recall_0_90_available": bool(
                    args.source_calibration
                ),
                "source_calibrated_recall_0_90_reason": (
                    None
                    if args.source_calibration
                    else "requires --source-calibration additive variant"
                ),
                "test_label_calibrated_threshold_forbidden": True,
            },
            "artifacts": {"predictions": artifact_path(args.predictions_out)},
            "environment": host_environment(),
            "timing_protocol": timing_protocol(),
            "timing_breakdown": timing_breakdown,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    )
    if args.manifest_in is not None:
        payload["artifacts"]["prepared_manifest"] = artifact_path(args.manifest_in)
        payload["split"]["prepared_manifest_sha256"] = manifest_sha256
    command_parts = [
        ".\\.venv\\Scripts\\python.exe",
        "scripts/run_m6_site_transfer.py",
        "--experiment",
        args.experiment,
    ]
    if args.fold is not None:
        command_parts.extend(["--fold", str(args.fold)])
    if args.site_id is not None:
        command_parts.extend(["--site-id", str(args.site_id)])
    if args.experiment == "b1":
        command_parts.extend(
            [
                "--direction",
                args.direction,
                "--meter-budget",
                str(args.meter_budget),
                "--selection-seed",
                str(args.selection_seed),
            ]
        )
    if args.experiment == "b2":
        command_parts.extend(
            [
                "--base-split",
                args.base_split,
                "--positive-budget",
                str(args.positive_budget),
                "--selection-seed",
                str(args.selection_seed),
            ]
        )
    if args.manifest_in is not None:
        command_parts.extend(["--manifest-in", artifact_path(args.manifest_in)])
    if args.source_calibration:
        command_parts.append("--source-calibration")
    command_parts.extend(
        [
            "--out",
            artifact_path(args.out),
            "--predictions-out",
            artifact_path(args.predictions_out),
        ]
    )
    write_json_with_provenance(
        args.out,
        payload,
        root=ROOT,
        provenance={
            "command": " ".join(command_parts),
            "note": "M6 additive runner; frozen M3 sources were not modified.",
        },
    )
    log(f"Saved {args.out} in {payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
