"""Observe the frozen M3 50/50 baseline for report figures.

This is an additive observer. It intentionally does not import or modify the
existing M3 runners' execution path. The duplicated model declarations below
are guarded by tests and numeric gates against the accepted 50/50 artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import re
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from lead import (
    BASELINE_FEATURE_COLS,
    DOWNSAMPLE_SEEDS,
    PROC,
    RANDOM_STATE,
    ROOT,
    SHIFTS,
    add_value_change_features,
    assert_no_building_overlap,
    downsample_indices,
    load_m3_frame,
    write_json_with_provenance,
)


MODEL_ORDER = (
    "lightgbm",
    "xgboost",
    "catboost",
    "hist_gradient_boosting",
)
VALUE_CHANGE_REGIME = "timestamp_merge"
EXPECTED_SPLIT = {
    "train_buildings": 725,
    "validation_buildings": 724,
    "train_rows": 10_078_945,
    "validation_rows": 10_137_155,
}
EXPECTED_AUCS = {
    "lightgbm": 0.9910452735866242,
    "xgboost": 0.9891696407517456,
    "catboost": 0.9882823167925664,
    "hist_gradient_boosting": 0.9914599806350353,
    "ensemble": 0.9918063907984431,
}
NOISE_FLOOR_ROC_AUC = 0.0005
PR_TOLERANCE = 0.001
RECALL_TOLERANCE = 0.001


def log(message: str) -> None:
    print(message, flush=True)


def frozen_model_contract(seed: int = RANDOM_STATE) -> dict[str, dict[str, Any]]:
    """Return the duplicated, test-guarded M3 model declarations."""
    return {
        "lightgbm": {
            "class": "LGBMClassifier",
            "params": {"n_estimators": 100, "verbose": -1, "random_state": seed},
        },
        "xgboost": {
            "class": "XGBClassifier",
            "params": {
                "n_estimators": 100,
                "eval_metric": "logloss",
                "verbosity": 0,
                "random_state": seed,
            },
        },
        "catboost": {
            "class": "CatBoostClassifier",
            "params": {
                "iterations": 1000,
                "verbose": False,
                "random_seed": seed,
                "allow_writing_files": False,
            },
        },
        "hist_gradient_boosting": {
            "class": "HistGradientBoostingClassifier",
            "params": {"max_iter": 100, "random_state": seed},
        },
    }


def build_frozen_models(seed: int = RANDOM_STATE) -> dict[str, Any]:
    contract = frozen_model_contract(seed)
    return {
        "lightgbm": lgb.LGBMClassifier(**contract["lightgbm"]["params"]),
        "xgboost": xgb.XGBClassifier(**contract["xgboost"]["params"]),
        "catboost": CatBoostClassifier(**contract["catboost"]["params"]),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            **contract["hist_gradient_boosting"]["params"]
        ),
    }


def _model_matrix(model_name: str, values: np.ndarray) -> np.ndarray:
    if model_name == "hist_gradient_boosting":
        return np.nan_to_num(values, nan=0)
    return values


def predict_probability(model_name: str, model: Any, values: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names.*",
            category=UserWarning,
        )
        return model.predict_proba(_model_matrix(model_name, values))[:, 1]


def predict_probability_with_constant_columns(
    model_name: str,
    model: Any,
    values: np.ndarray,
    columns: list[int],
    *,
    batch_size: int = 200_000,
) -> np.ndarray:
    """Predict an ablation without mutating or copying the full validation set."""
    output = np.empty(len(values), dtype="float64")
    for start in range(0, len(values), batch_size):
        end = min(start + batch_size, len(values))
        batch = np.array(values[start:end], copy=True)
        batch[:, columns] = 0
        output[start:end] = predict_probability(model_name, model, batch)
    return output


def fit_frozen_models(
    x_train: np.ndarray,
    y_train: pd.Series,
    *,
    seed: int = RANDOM_STATE,
) -> tuple[dict[str, Any], dict[str, float]]:
    models = build_frozen_models(seed)
    timings: dict[str, float] = {}
    for model_name in MODEL_ORDER:
        started = time.perf_counter()
        models[model_name].fit(_model_matrix(model_name, x_train), y_train)
        timings[model_name] = time.perf_counter() - started
        log(f"Fitted {model_name} in {timings[model_name]:.3f}s")
    return models, timings


def evaluation_summary(y_true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    pred_label = (pred >= 0.5).astype("int8")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred_label,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, pred_label, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, pred)),
        "pr_auc": float(average_precision_score(y_true, pred)),
        "threshold_0_5": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        },
    }


def _compress_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_points: int,
) -> dict[str, list[float]]:
    if len(x) <= max_points:
        indices = np.arange(len(x))
    else:
        indices = np.unique(np.linspace(0, len(x) - 1, max_points).astype(int))
    return {
        "x": [float(value) for value in x[indices]],
        "y": [float(value) for value in y[indices]],
    }


def curve_summary(
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    max_points: int = 1200,
) -> dict[str, Any]:
    fpr, tpr, _ = roc_curve(y_true, pred)
    precision, recall, _ = precision_recall_curve(y_true, pred)
    return {
        "roc": _compress_curve(fpr, tpr, max_points=max_points),
        "precision_recall": _compress_curve(
            recall,
            precision,
            max_points=max_points,
        ),
    }


def array_fingerprint(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def stratified_sample_indices(
    y_true: np.ndarray,
    *,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    if sample_size >= len(y_true):
        return np.arange(len(y_true), dtype="int64")
    rng = np.random.RandomState(seed)
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    positive_count = max(1, round(sample_size * len(positive) / len(y_true)))
    negative_count = sample_size - positive_count
    sampled = np.concatenate(
        [
            rng.choice(negative, negative_count, replace=False),
            rng.choice(positive, positive_count, replace=False),
        ]
    )
    return np.sort(sampled.astype("int64"))


def _empty_importance_records(feature_cols: list[str]) -> dict[str, dict[str, Any]]:
    return {
        model_name: {
            feature: {"roc_auc_decrease": [], "pr_auc_decrease": []}
            for feature in feature_cols
        }
        for model_name in (*MODEL_ORDER, "ensemble")
    }


def _summarize_importance(
    records: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for model_name, features in records.items():
        rows = []
        for feature, metric_values in features.items():
            roc_values = np.asarray(metric_values["roc_auc_decrease"], dtype=float)
            pr_values = np.asarray(metric_values["pr_auc_decrease"], dtype=float)
            rows.append(
                {
                    "feature": feature,
                    "roc_auc_decrease_mean": float(roc_values.mean()),
                    "roc_auc_decrease_std": float(roc_values.std()),
                    "pr_auc_decrease_mean": float(pr_values.mean()),
                    "pr_auc_decrease_std": float(pr_values.std()),
                    "repeats": int(len(roc_values)),
                }
            )
        output[model_name] = sorted(
            rows,
            key=lambda row: row["roc_auc_decrease_mean"],
            reverse=True,
        )
    return output


def permutation_importance_observations(
    models: dict[str, Any],
    x_sample: np.ndarray,
    y_sample: np.ndarray,
    feature_cols: list[str],
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    baseline_predictions = {
        model_name: predict_probability(model_name, models[model_name], x_sample)
        for model_name in MODEL_ORDER
    }
    baseline_predictions["ensemble"] = np.mean(
        [baseline_predictions[name] for name in MODEL_ORDER],
        axis=0,
    )
    baseline_metrics = {
        name: evaluation_summary(y_sample, pred)
        for name, pred in baseline_predictions.items()
    }
    records = _empty_importance_records(feature_cols)
    work = x_sample.copy()
    for feature_idx, feature in enumerate(feature_cols):
        original = work[:, feature_idx].copy()
        for repeat in range(repeats):
            rng = np.random.RandomState(seed + feature_idx * 10_000 + repeat)
            work[:, feature_idx] = original[rng.permutation(len(original))]
            permuted_predictions = {
                model_name: predict_probability(
                    model_name,
                    models[model_name],
                    work,
                )
                for model_name in MODEL_ORDER
            }
            permuted_predictions["ensemble"] = np.mean(
                [permuted_predictions[name] for name in MODEL_ORDER],
                axis=0,
            )
            for model_name, pred in permuted_predictions.items():
                permuted = evaluation_summary(y_sample, pred)
                records[model_name][feature]["roc_auc_decrease"].append(
                    baseline_metrics[model_name]["roc_auc"] - permuted["roc_auc"]
                )
                records[model_name][feature]["pr_auc_decrease"].append(
                    baseline_metrics[model_name]["pr_auc"] - permuted["pr_auc"]
                )
        work[:, feature_idx] = original
        if (feature_idx + 1) % 10 == 0 or feature_idx + 1 == len(feature_cols):
            log(f"Permutation importance: {feature_idx + 1}/{len(feature_cols)}")
    summarized = _summarize_importance(records)
    return {
        "baseline_metrics": baseline_metrics,
        "models": {name: summarized[name] for name in MODEL_ORDER},
        "ensemble": summarized["ensemble"],
        "consensus": consensus_importance(summarized),
        "screening": screen_importance_candidates(summarized["ensemble"]),
    }


def consensus_importance(
    summarized: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_model = {
        model_name: {
            row["feature"]: {**row, "rank": rank}
            for rank, row in enumerate(summarized[model_name], start=1)
        }
        for model_name in MODEL_ORDER
    }
    features = list(by_model[MODEL_ORDER[0]])
    rows = []
    for feature in features:
        ranks = [by_model[name][feature]["rank"] for name in MODEL_ORDER]
        decreases = [
            by_model[name][feature]["roc_auc_decrease_mean"] for name in MODEL_ORDER
        ]
        rows.append(
            {
                "feature": feature,
                "models_in_top10": int(sum(rank <= 10 for rank in ranks)),
                "mean_rank": float(np.mean(ranks)),
                "rank_std": float(np.std(ranks)),
                "mean_roc_auc_decrease": float(np.mean(decreases)),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -row["models_in_top10"],
            row["mean_rank"],
            -row["mean_roc_auc_decrease"],
        ),
    )


def screen_importance_candidates(
    ensemble_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []
    for row in ensemble_rows:
        roc_indistinguishable = (
            row["roc_auc_decrease_mean"] <= row["roc_auc_decrease_std"]
        )
        pr_indistinguishable = row["pr_auc_decrease_mean"] <= row["pr_auc_decrease_std"]
        if (
            row["roc_auc_decrease_mean"] <= 0
            or row["pr_auc_decrease_mean"] <= 0
            or (roc_indistinguishable and pr_indistinguishable)
        ):
            candidates.append(
                {
                    **row,
                    "screen_label": "redundant_or_replaceable",
                    "reason": "zero_negative_or_indistinguishable_permutation_effect",
                }
            )
    return sorted(
        candidates,
        key=lambda row: (
            row["roc_auc_decrease_mean"],
            row["pr_auc_decrease_mean"],
        ),
    )


_VALUE_CHANGE_RE = re.compile(r"^lag_value_(diff|ratio)_(-?\d+)$")


def correlated_group_for_feature(
    feature: str,
    feature_cols: list[str],
) -> list[str]:
    match = _VALUE_CHANGE_RE.match(feature)
    if not match:
        return [feature]
    kind, raw_shift = match.groups()
    shift = int(raw_shift)
    other_kind = "ratio" if kind == "diff" else "diff"
    candidates = {
        feature,
        f"lag_value_{other_kind}_{shift}",
        f"lag_value_{kind}_{shift - 1}",
        f"lag_value_{kind}_{shift + 1}",
    }
    return [name for name in feature_cols if name in candidates]


def grouped_permutation_observations(
    models: dict[str, Any],
    x_sample: np.ndarray,
    y_sample: np.ndarray,
    feature_cols: list[str],
    screened: list[dict[str, Any]],
    *,
    max_groups: int,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    baseline_predictions = {
        name: predict_probability(name, models[name], x_sample) for name in MODEL_ORDER
    }
    baseline_ensemble = np.mean(
        [baseline_predictions[name] for name in MODEL_ORDER], axis=0
    )
    baseline = evaluation_summary(y_sample, baseline_ensemble)
    groups: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in screened:
        group = correlated_group_for_feature(candidate["feature"], feature_cols)
        key = tuple(group)
        if key not in seen:
            seen.add(key)
            groups.append(group)
        if len(groups) >= max_groups:
            break
    output = []
    work = x_sample.copy()
    feature_index = {name: idx for idx, name in enumerate(feature_cols)}
    for group_idx, group in enumerate(groups):
        originals = {name: work[:, feature_index[name]].copy() for name in group}
        roc_decreases = []
        pr_decreases = []
        for repeat in range(repeats):
            rng = np.random.RandomState(seed + group_idx * 10_000 + repeat)
            order = rng.permutation(len(y_sample))
            for name in group:
                work[:, feature_index[name]] = originals[name][order]
            predictions = [
                predict_probability(name, models[name], work) for name in MODEL_ORDER
            ]
            metrics = evaluation_summary(y_sample, np.mean(predictions, axis=0))
            roc_decreases.append(baseline["roc_auc"] - metrics["roc_auc"])
            pr_decreases.append(baseline["pr_auc"] - metrics["pr_auc"])
        for name, values in originals.items():
            work[:, feature_index[name]] = values
        output.append(
            {
                "features": group,
                "roc_auc_decrease_mean": float(np.mean(roc_decreases)),
                "roc_auc_decrease_std": float(np.std(roc_decreases)),
                "pr_auc_decrease_mean": float(np.mean(pr_decreases)),
                "pr_auc_decrease_std": float(np.std(pr_decreases)),
            }
        )
    return output


def classify_ablation(
    baseline: dict[str, Any],
    ablated: dict[str, Any],
) -> tuple[str, dict[str, float]]:
    delta = {
        "roc_auc": ablated["roc_auc"] - baseline["roc_auc"],
        "pr_auc": ablated["pr_auc"] - baseline["pr_auc"],
        "recall_0_5": (
            ablated["threshold_0_5"]["recall"] - baseline["threshold_0_5"]["recall"]
        ),
    }
    if all(value >= 0 for value in delta.values()):
        label = "possible_removal_candidate"
    elif (
        delta["roc_auc"] >= -NOISE_FLOOR_ROC_AUC
        and delta["pr_auc"] >= -PR_TOLERANCE
        and delta["recall_0_5"] >= -RECALL_TOLERANCE
    ):
        label = "redundant_or_replaceable"
    else:
        label = "harmful_to_remove"
    return label, delta


def targeted_constant_ablation(
    x_train: np.ndarray,
    y_train: pd.Series,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: list[str],
    baseline_metrics: dict[str, Any],
    groups: list[list[str]],
    *,
    max_candidates: int,
    on_candidate_completed: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    output = []
    index = {name: idx for idx, name in enumerate(feature_cols)}
    selected_groups = groups[:max_candidates]
    for candidate_index, group in enumerate(selected_groups, start=1):
        log(
            "Targeted ablation "
            f"{candidate_index}/{len(selected_groups)}: {', '.join(group)}"
        )
        columns = [index[name] for name in group]
        ablated_train = np.array(x_train, copy=True)
        ablated_train[:, columns] = 0
        models, timings = fit_frozen_models(ablated_train, y_train)
        predictions = {
            name: predict_probability_with_constant_columns(
                name,
                models[name],
                x_val,
                columns,
            )
            for name in MODEL_ORDER
        }
        predictions["ensemble"] = np.mean(
            [predictions[name] for name in MODEL_ORDER], axis=0
        )
        metrics = {
            name: evaluation_summary(y_val, pred) for name, pred in predictions.items()
        }
        label, delta = classify_ablation(
            baseline_metrics["ensemble"], metrics["ensemble"]
        )
        output.append(
            {
                "features": group,
                "method": "constantize_standardized_columns_to_zero",
                "candidate_status": label,
                "ensemble_delta": delta,
                "metrics": metrics,
                "fit_seconds": timings,
            }
        )
        del ablated_train, models, predictions
        gc.collect()
        log(
            f"Completed targeted ablation {candidate_index}/{len(selected_groups)}: "
            f"{label}"
        )
        if on_candidate_completed is not None:
            on_candidate_completed(output)
    return output


def select_value_change_segment(
    validation_frame: pd.DataFrame,
    *,
    radius_hours: int = 24,
) -> dict[str, Any]:
    ratio_col = "lag_value_ratio_1"
    diff_col = "lag_value_diff_1"
    candidates = validation_frame.loc[
        (validation_frame["anomaly"] == 1)
        & validation_frame[ratio_col].between(0.05, 20.0)
        & validation_frame[diff_col].notna()
    ].copy()
    if candidates.empty:
        raise RuntimeError("No bounded anomalous value-change illustration candidate")
    candidates["_score"] = np.abs(np.log(candidates[ratio_col]))
    anchor = candidates.sort_values(
        ["_score", "building_id", "meter", "timestamp"],
        ascending=[False, True, True, True],
    ).iloc[0]
    start = anchor["timestamp"] - pd.Timedelta(hours=radius_hours)
    end = anchor["timestamp"] + pd.Timedelta(hours=radius_hours)
    segment = validation_frame.loc[
        (validation_frame["building_id"] == anchor["building_id"])
        & (validation_frame["meter"] == anchor["meter"])
        & validation_frame["timestamp"].between(start, end),
        [
            "timestamp",
            "meter_reading",
            "anomaly",
            diff_col,
            ratio_col,
        ],
    ].sort_values("timestamp")
    return {
        "selection_rule": (
            "highest abs(log(lag_value_ratio_1)) among anomaly rows with "
            "0.05 <= ratio <= 20 and finite one-hour difference"
        ),
        "building_id": int(anchor["building_id"]),
        "meter": int(anchor["meter"]),
        "anchor_timestamp": anchor["timestamp"].isoformat(),
        "shift_hours": 1,
        "points": [
            {
                "timestamp": row.timestamp.isoformat(),
                "meter_reading": float(row.meter_reading),
                "anomaly": int(row.anomaly),
                "difference": None if pd.isna(row[diff_col]) else float(row[diff_col]),
                "ratio": None if pd.isna(row[ratio_col]) else float(row[ratio_col]),
            }
            for _, row in segment.iterrows()
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "m3_figure_observations_50_50.json",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=PROC / "m3_figure_predictions_50_50.npz",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=PROC / "m3_figure_observations_50_50.checkpoint.json",
    )
    parser.add_argument("--permutation-sample-size", type=int, default=50_000)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--max-group-checks", type=int, default=10)
    parser.add_argument("--max-ablation-candidates", type=int, default=3)
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    df = load_m3_frame()
    mask_val = (df["building_id"] % 2 == 1).to_numpy()
    train_buildings = set(df.loc[~mask_val, "building_id"].unique())
    val_buildings = set(df.loc[mask_val, "building_id"].unique())
    overlap = assert_no_building_overlap(
        train_buildings,
        val_buildings,
        split_name="50_50_mod2_figure_observer",
    )
    observed_split = {
        "train_buildings": len(train_buildings),
        "validation_buildings": len(val_buildings),
        "train_rows": int((~mask_val).sum()),
        "validation_rows": int(mask_val.sum()),
    }
    if observed_split != EXPECTED_SPLIT or overlap:
        raise RuntimeError(
            f"Frozen 50/50 split mismatch: {observed_split}, overlap={len(overlap)}"
        )
    validation_raw_indices = np.flatnonzero(mask_val).astype("int64")
    log("Building frozen timestamp-merge feature tables")
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
    ds_idx = downsample_indices(y_train)
    y_fit = y_train.loc[ds_idx].copy()

    log("Observing M3.1 17-feature LightGBM on the final 50/50 split")
    baseline_scaler = StandardScaler()
    x_train_17 = baseline_scaler.fit_transform(
        train_full.loc[ds_idx, BASELINE_FEATURE_COLS]
    )
    x_val_17 = baseline_scaler.transform(val_full[BASELINE_FEATURE_COLS])
    m3_1_model = lgb.LGBMClassifier(
        n_estimators=100,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    m3_1_started = time.perf_counter()
    m3_1_model.fit(x_train_17, y_fit)
    m3_1_fit_seconds = time.perf_counter() - m3_1_started
    m3_1_pred = m3_1_model.predict_proba(x_val_17)[:, 1]
    del x_train_17, x_val_17, m3_1_model, baseline_scaler

    log("Observing four frozen 137-feature models and equal-weight ensemble")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[ds_idx, feature_cols])
    x_val = scaler.transform(val_full[feature_cols])
    del df, train_full, val_full, y_train, scaler
    gc.collect()
    log("Released raw and feature DataFrames after model-matrix construction")
    models, fit_seconds = fit_frozen_models(x_train, y_fit)
    predictions = {
        name: predict_probability(name, models[name], x_val) for name in MODEL_ORDER
    }
    predictions["ensemble"] = np.mean(
        [predictions[name] for name in MODEL_ORDER], axis=0
    )
    metrics = {
        "m3_1_lightgbm": evaluation_summary(y_val, m3_1_pred),
        **{
            name: evaluation_summary(y_val, prediction)
            for name, prediction in predictions.items()
        },
    }
    for name, expected in EXPECTED_AUCS.items():
        delta = metrics[name]["roc_auc"] - expected
        if abs(delta) > NOISE_FLOOR_ROC_AUC:
            raise RuntimeError(
                f"Frozen {name} AUC gate failed: {metrics[name]['roc_auc']:.9f} "
                f"vs {expected:.9f} ({delta:+.9f})"
            )
    curves = {
        "m3_1_lightgbm": curve_summary(y_val, m3_1_pred),
        **{
            name: curve_summary(y_val, prediction)
            for name, prediction in predictions.items()
        },
    }
    if not args.skip_predictions:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.predictions_out,
            validation_raw_index=validation_raw_indices,
            anomaly=y_val,
            m3_1_lightgbm=m3_1_pred.astype("float32"),
            **{
                name: prediction.astype("float32")
                for name, prediction in predictions.items()
            },
        )
    del predictions, m3_1_pred
    gc.collect()
    log("Released full prediction arrays after metrics, curves, and NPZ capture")

    sample_idx = stratified_sample_indices(
        y_val,
        sample_size=args.permutation_sample_size,
        seed=RANDOM_STATE,
    )
    permutation = permutation_importance_observations(
        models,
        x_val[sample_idx].copy(),
        y_val[sample_idx],
        feature_cols,
        repeats=args.permutation_repeats,
        seed=RANDOM_STATE,
    )
    grouped = grouped_permutation_observations(
        models,
        x_val[sample_idx].copy(),
        y_val[sample_idx],
        feature_cols,
        permutation["screening"],
        max_groups=args.max_group_checks,
        repeats=args.permutation_repeats,
        seed=RANDOM_STATE + 1_000_000,
    )
    permutation["grouped_checks"] = grouped
    del models
    gc.collect()
    log("Released baseline models after permutation and grouped checks")
    ablation_groups = [row["features"] for row in grouped]

    def write_checkpoint(
        stage: str,
        targeted: list[dict[str, Any]],
    ) -> None:
        write_json_with_provenance(
            args.checkpoint_out,
            {
                "schema_version": 1,
                "experiment": "m3_figure_observations_50_50",
                "complete": False,
                "stage": stage,
                "observation_only": True,
                "split": {
                    **observed_split,
                    "building_overlap": 0,
                },
                "feature_count": len(feature_cols),
                "metrics": metrics,
                "curves": curves,
                "permutation_importance": {
                    "sample_size": len(sample_idx),
                    "sample_validation_position_sha256": array_fingerprint(sample_idx),
                    "sample_anomaly_rate": float(y_val[sample_idx].mean()),
                    "repeats": args.permutation_repeats,
                    "scoring": ["roc_auc", "average_precision"],
                    **permutation,
                    "targeted_ablation": targeted,
                },
                "value_change_illustration": illustration,
                "canonical_feature_set_changed": False,
                "elapsed_seconds": time.perf_counter() - started,
            },
            root=ROOT,
            provenance={
                "note": "Incomplete observer checkpoint; not a report artifact."
            },
        )
        log(f"Saved checkpoint: {stage}")

    write_checkpoint("permutation_and_grouped_checks_complete", [])
    if args.skip_ablation:
        targeted_ablation: list[dict[str, Any]] = []
    else:
        targeted_ablation = targeted_constant_ablation(
            x_train,
            y_fit,
            x_val,
            y_val,
            feature_cols,
            metrics,
            ablation_groups,
            max_candidates=args.max_ablation_candidates,
            on_candidate_completed=lambda rows: write_checkpoint(
                f"targeted_ablation_{len(rows)}_complete",
                rows,
            ),
        )
    permutation["targeted_ablation"] = targeted_ablation

    payload = {
        "schema_version": 1,
        "experiment": "m3_figure_observations_50_50",
        "observation_only": True,
        "frozen_contract": {
            "split": "50_50_mod2",
            "validation_rule": "building_id % 2 == 1",
            "value_change_regime": VALUE_CHANGE_REGIME,
            "baseline_features": list(BASELINE_FEATURE_COLS),
            "value_change_shifts": list(SHIFTS),
            "feature_count": len(feature_cols),
            "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
            "model_seed": RANDOM_STATE,
            "model_order": list(MODEL_ORDER),
            "model_contract": frozen_model_contract(),
            "ensemble": "equal_weight_probability_mean",
            "threshold": 0.5,
            "existing_code_modified": False,
        },
        "split": {
            **observed_split,
            "building_overlap": 0,
            "validation_raw_index_sha256": array_fingerprint(validation_raw_indices),
        },
        "fit": {
            "downsampled_rows": len(ds_idx),
            "m3_1_lightgbm_fit_seconds": m3_1_fit_seconds,
            "model_fit_seconds": fit_seconds,
            "ensemble_fit_seconds_definition": "sum_of_four_component_fit_seconds",
            "ensemble_fit_seconds": float(sum(fit_seconds.values())),
        },
        "metrics": metrics,
        "curves": curves,
        "permutation_importance": {
            "sample_size": len(sample_idx),
            "sample_validation_position_sha256": array_fingerprint(sample_idx),
            "sample_anomaly_rate": float(y_val[sample_idx].mean()),
            "repeats": args.permutation_repeats,
            "scoring": ["roc_auc", "average_precision"],
            **permutation,
        },
        "feature_pruning_policy": {
            "screening_is_not_removal_proof": True,
            "roc_auc_noise_floor": NOISE_FLOOR_ROC_AUC,
            "pr_auc_tolerance": PR_TOLERANCE,
            "recall_tolerance": RECALL_TOLERANCE,
            "canonical_feature_set_changed": False,
        },
        "value_change_illustration": illustration,
        "artifacts": {
            "predictions": None
            if args.skip_predictions
            else str(args.predictions_out.relative_to(ROOT)),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_with_provenance(
        args.out,
        payload,
        root=ROOT,
        provenance={
            "note": (
                "Observation only: existing M3 code, inputs, split, sampling, "
                "features, model settings, execution order, predictions, and "
                "ensemble behavior are unchanged."
            )
        },
    )
    log(f"Saved {args.out}")


if __name__ == "__main__":
    main()
