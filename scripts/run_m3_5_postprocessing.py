"""Run M3.5 post-processing on the M3.4 seed-42 ensemble."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from run_m3_4_ensemble import (
    BASELINE_FEATURE_COLS,
    DOWNSAMPLE_SEEDS,
    PROC,
    RANDOM_STATE,
    SHIFTS,
    add_value_change_features,
    classification_metrics,
    downsample_indices,
    fit_predict_models,
    load_m3_frame,
    log,
)


M3_2_REFERENCE_AUC = 0.9920
M3_4_REFERENCE_AUC = 0.9928
RULE_2B_DAYOFYEAR_THRESHOLD = 366.9583


def apply_rules(
    base_pred: np.ndarray,
    *,
    meter_reading: pd.Series,
    dayofyear: pd.Series,
    rule_2a_mask: np.ndarray | None,
    include_rule_1: bool = True,
    include_rule_2a: bool = True,
    include_rule_2b: bool = True,
) -> np.ndarray:
    pred = base_pred.copy()
    if include_rule_1:
        pred[meter_reading.to_numpy() == 1.0] = 1.0
    if include_rule_2a and rule_2a_mask is not None:
        pred[rule_2a_mask] = 0.0
    if include_rule_2b:
        pred[dayofyear.to_numpy() > RULE_2B_DAYOFYEAR_THRESHOLD] = 0.0
    return pred


def auc_delta(y_true: pd.Series, pred: np.ndarray, base_auc: float) -> dict[str, float]:
    auc = float(roc_auc_score(y_true, pred))
    return {"val_auc": auc, "delta_auc_vs_pre": float(auc - base_auc)}


def trigger_summary(mask: np.ndarray, y_true: pd.Series) -> dict[str, float | int]:
    n = int(mask.sum())
    return {
        "n_rows": n,
        "trigger_rate": float(mask.mean()),
        "n_anomalies": int(y_true.to_numpy()[mask].sum()),
        "anomaly_rate": float(y_true.to_numpy()[mask].mean()) if n else 0.0,
    }


def jan1_eda(val_full: pd.DataFrame, y_val: pd.Series) -> dict[str, object]:
    day1_mask = val_full["dayofyear"].to_numpy() == 1.0
    day1 = val_full.loc[day1_mask, ["building_id", "meter_reading", "dayofyear"]].copy()
    day1["anomaly"] = y_val.loc[day1.index].to_numpy()
    building_stats = (
        day1.groupby("building_id", sort=True)["anomaly"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(
            columns={
                "count": "n_rows",
                "sum": "n_anomalies",
                "mean": "anomaly_rate",
            }
        )
    )
    anomalous_buildings = building_stats.loc[building_stats["n_anomalies"] > 0]
    top_anomalous = anomalous_buildings.sort_values(
        ["n_anomalies", "anomaly_rate", "building_id"],
        ascending=[False, False, True],
    ).head(10)

    n_rows = int(day1_mask.sum())
    n_anomalies = int(day1["anomaly"].sum()) if n_rows else 0
    anomaly_rate = float(day1["anomaly"].mean()) if n_rows else 0.0
    n_buildings = int(day1["building_id"].nunique()) if n_rows else 0
    n_anomalous_buildings = int(len(anomalous_buildings))

    # M2's Rule 2a was a LEAD-subset range exception. In M3, Jan-1 is not
    # treated as a blanket normal override unless it is almost purely normal.
    if n_rows == 0:
        conclusion = "not_applicable_no_rows"
        apply_mask = None
        rationale = "No validation rows have dayofyear == 1.0."
    elif anomaly_rate <= 0.005 and n_anomalous_buildings <= 1:
        conclusion = "blanket_dayofyear_1_to_normal"
        apply_mask = day1_mask
        rationale = (
            "Jan-1 validation rows are almost entirely normal, so a dataset-level "
            "start-point override is defensible."
        )
    else:
        conclusion = "not_applicable"
        apply_mask = None
        rationale = (
            "Jan-1 validation rows include enough anomalies and anomalous buildings "
            "that the M2 building_id range should not be translated to M3."
        )

    return {
        "n_rows": n_rows,
        "anomaly_rate": anomaly_rate,
        "n_anomalies": n_anomalies,
        "n_buildings": n_buildings,
        "n_anomalous_buildings": n_anomalous_buildings,
        "top_anomalous_buildings": top_anomalous.to_dict(orient="records"),
        "conclusion": conclusion,
        "rationale": rationale,
        "rule_2a_applied": apply_mask is not None,
        "rule_2a_mask": apply_mask,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "m3_5_results.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=PROC / "m3_5_val_predictions.csv.gz",
        help="Output per-row validation prediction CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    if len(SHIFTS) != 60:
        raise AssertionError("Unexpected value-change shift set")

    df = load_m3_frame()
    mask_val = (df["building_id"] % 5 == 4).to_numpy()
    train_buildings = set(df.loc[~mask_val, "building_id"].unique())
    val_buildings = set(df.loc[mask_val, "building_id"].unique())
    overlap = train_buildings & val_buildings
    if overlap:
        raise AssertionError(f"building overlap: {sorted(overlap)[:5]}")

    log("Adding M3.2 offline value-change features")
    train_full = add_value_change_features(df.loc[~mask_val])
    val_full = add_value_change_features(df.loc[mask_val])
    value_cols = [c for c in train_full.columns if c.startswith("lag_value_")]
    feature_cols = BASELINE_FEATURE_COLS + value_cols
    if len(feature_cols) != 137:
        raise AssertionError(f"Expected 137 M3.2 features, got {len(feature_cols)}")
    if len(val_full) != int(mask_val.sum()):
        raise AssertionError("Validation feature rows are misaligned after FE")

    y_train = train_full["anomaly"]
    y_val = val_full["anomaly"]
    ds_idx = downsample_indices(y_train)
    log(f"Downsampled train rows: {len(ds_idx):,}")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[ds_idx, feature_cols])
    x_val = scaler.transform(val_full[feature_cols])
    y_fit = y_train.loc[ds_idx]
    log(f"Scaled train/val matrices: {x_train.shape} / {x_val.shape}")

    run = fit_predict_models(
        x_train,
        y_fit,
        x_val,
        y_val,
        RANDOM_STATE,
        return_predictions=True,
    )
    model_preds = run.pop("raw_predictions")
    ensemble_pred = run.pop("raw_ensemble_prediction")
    pre_metrics = classification_metrics(y_val, ensemble_pred)
    pre_auc = pre_metrics["val_auc"]
    if abs(pre_auc - M3_4_REFERENCE_AUC) > 0.0001:
        raise AssertionError(
            f"Expected M3.4 seed-42 ensemble AUC near {M3_4_REFERENCE_AUC}, got {pre_auc}"
        )

    val_prediction_frame = pd.DataFrame(
        {
            "row_id": np.arange(len(val_full), dtype=np.int32),
            "building_id": val_full["building_id"].to_numpy(),
            "meter_reading": val_full["meter_reading"].to_numpy(),
            "dayofyear": val_full["dayofyear"].to_numpy(),
            "anomaly": y_val.to_numpy(),
            "pred_lightgbm": model_preds["lightgbm"],
            "pred_xgboost": model_preds["xgboost"],
            "pred_catboost": model_preds["catboost"],
            "pred_hist_gradient_boosting": model_preds["hist_gradient_boosting"],
            "pred_ensemble": ensemble_pred,
        }
    )
    if len(val_prediction_frame) != len(ensemble_pred):
        raise AssertionError("Prediction frame length does not match predictions")
    if not np.array_equal(val_prediction_frame["anomaly"].to_numpy(), y_val.to_numpy()):
        raise AssertionError("Validation labels are not aligned with predictions")

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    val_prediction_frame.to_csv(args.predictions_out, index=False)
    log(f"Saved {args.predictions_out}")

    rule_1_mask = val_full["meter_reading"].to_numpy() == 1.0
    if int(rule_1_mask.sum()) == 0:
        raise RuntimeError("Rule 1 triggered 0 rows; stopping as requested")

    eda = jan1_eda(val_full, y_val)
    rule_2a_mask = eda.pop("rule_2a_mask")
    rule_2b_mask = val_full["dayofyear"].to_numpy() > RULE_2B_DAYOFYEAR_THRESHOLD

    post_by_rule = {
        "rule_1_only": auc_delta(
            y_val,
            apply_rules(
                ensemble_pred,
                meter_reading=val_full["meter_reading"],
                dayofyear=val_full["dayofyear"],
                rule_2a_mask=rule_2a_mask,
                include_rule_2a=False,
                include_rule_2b=False,
            ),
            pre_auc,
        ),
        "rule_2a_only": auc_delta(
            y_val,
            apply_rules(
                ensemble_pred,
                meter_reading=val_full["meter_reading"],
                dayofyear=val_full["dayofyear"],
                rule_2a_mask=rule_2a_mask,
                include_rule_1=False,
                include_rule_2b=False,
            ),
            pre_auc,
        )
        if rule_2a_mask is not None
        else {"val_auc": pre_auc, "delta_auc_vs_pre": 0.0},
        "rule_2b_only": auc_delta(
            y_val,
            apply_rules(
                ensemble_pred,
                meter_reading=val_full["meter_reading"],
                dayofyear=val_full["dayofyear"],
                rule_2a_mask=rule_2a_mask,
                include_rule_1=False,
                include_rule_2a=False,
            ),
            pre_auc,
        ),
    }

    combined_pred = apply_rules(
        ensemble_pred,
        meter_reading=val_full["meter_reading"],
        dayofyear=val_full["dayofyear"],
        rule_2a_mask=rule_2a_mask,
    )
    combined_metrics = classification_metrics(y_val, combined_pred)
    combined_delta = float(combined_metrics["val_auc"] - pre_auc)
    if combined_delta < 0 or combined_delta > 0.01:
        raise RuntimeError(
            f"Post-processing delta {combined_delta:+.6f} outside expected range"
        )

    rule_2a_trigger = (
        trigger_summary(rule_2a_mask, y_val)
        if rule_2a_mask is not None
        else {
            "n_rows": 0,
            "trigger_rate": 0.0,
            "n_anomalies": 0,
            "anomaly_rate": 0.0,
        }
    )
    log(f"Rule 1 trigger rows: {int(rule_1_mask.sum()):,}")
    log(f"Rule 2a trigger rows: {int(rule_2a_trigger['n_rows']):,}")
    log(f"Rule 2b trigger rows: {int(rule_2b_mask.sum()):,}")
    log(
        "AUC pre/post/delta: "
        f"{pre_auc:.6f} / {combined_metrics['val_auc']:.6f} / {combined_delta:+.6f}"
    )

    results: dict[str, object] = {
        "experiment": "m3_5_postprocessing",
        "canonical_line": "80_20_mod5_offline",
        "feature_set": "m3_2_137_features",
        "random_state": RANDOM_STATE,
        "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
        "postprocessing_base": "m3_4_seed_42_equal_weight_4_model_ensemble",
        "split": {
            "name": "80_20_mod5",
            "n_train_buildings": int(len(train_buildings)),
            "n_val_buildings": int(len(val_buildings)),
            "n_train_rows": int((~mask_val).sum()),
            "n_val_rows": int(mask_val.sum()),
            "train_anomaly_rate": float(df.loc[~mask_val, "anomaly"].mean()),
            "val_anomaly_rate": float(df.loc[mask_val, "anomaly"].mean()),
            "building_overlap": int(len(overlap)),
        },
        "feature_counts": {
            "baseline": int(len(BASELINE_FEATURE_COLS)),
            "value_change": int(len(value_cols)),
            "total": int(len(feature_cols)),
        },
        "n_train_downsampled": int(len(ds_idx)),
        "pre": {
            **pre_metrics,
            "delta_auc_vs_m3_2": float(pre_auc - M3_2_REFERENCE_AUC),
        },
        "post_by_rule": post_by_rule,
        "post_combined": {
            **combined_metrics,
            "delta_auc_vs_pre": combined_delta,
            "delta_auc_vs_m3_2": float(
                combined_metrics["val_auc"] - M3_2_REFERENCE_AUC
            ),
        },
        "rules": {
            "rule_1_meter_reading_eq_1_to_1": trigger_summary(rule_1_mask, y_val),
            "rule_2a_dayofyear_eq_1_to_0": {
                **rule_2a_trigger,
                "eda": eda,
            },
            "rule_2b_dayofyear_gt_366_9583_to_0": trigger_summary(rule_2b_mask, y_val),
            "application_order": [
                "rule_1_meter_reading_eq_1_to_1",
                "rule_2a_dayofyear_eq_1_to_0",
                "rule_2b_dayofyear_gt_366_9583_to_0",
            ],
            "rule_2_overrides_rule_1": True,
        },
        "artifacts": {
            "val_predictions": str(args.predictions_out),
        },
        "m3_4_run": run,
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"Saved {args.out}")


if __name__ == "__main__":
    main()
