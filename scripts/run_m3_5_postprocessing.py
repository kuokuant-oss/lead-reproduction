"""Run M3.5 post-processing on the M3.4 seed-42 ensemble."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from lead import (
    BASELINE_FEATURE_COLS,
    DOWNSAMPLE_SEEDS,
    PROC,
    RANDOM_STATE,
    SHIFTS,
    add_value_change_features,
    classification_metrics,
    downsample_indices,
    load_m3_frame,
)
from run_m3_4_ensemble import (
    fit_predict_models,
    log,
)


M3_2_REFERENCE_AUC = 0.9920
M3_4_REFERENCE_AUC = 0.9928
RULE_2B_DAYOFYEAR_THRESHOLD = 366.9583
SHUFFLE_SEEDS = (42, 123, 999)
METER_NAMES = {
    0: "electricity",
    1: "chilled_water",
    2: "steam",
    3: "hot_water",
}


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


def validate_alignment(
    *,
    val_full: pd.DataFrame,
    x_val: np.ndarray,
    y_val: pd.Series,
    ensemble_pred: np.ndarray,
    model_preds: dict[str, np.ndarray],
    rule_masks: dict[str, np.ndarray],
) -> dict[str, int | bool]:
    n_val = len(val_full)
    if not isinstance(val_full.index, pd.RangeIndex):
        raise AssertionError("Validation frame index must be a RangeIndex after FE")
    if not val_full.index.equals(y_val.index):
        raise AssertionError("Validation labels index does not match val_full")
    if x_val.shape[0] != n_val:
        raise AssertionError(f"x_val rows {x_val.shape[0]} != val rows {n_val}")
    if len(ensemble_pred) != n_val:
        raise AssertionError("Ensemble prediction length does not match val_full")
    for name, pred in model_preds.items():
        if len(pred) != n_val:
            raise AssertionError(f"{name} prediction length does not match val_full")
    for name, mask in rule_masks.items():
        if len(mask) != n_val:
            raise AssertionError(f"{name} mask length does not match val_full")
        if mask.dtype != np.bool_:
            raise AssertionError(f"{name} mask must be boolean")
    return {
        "n_val_rows": int(n_val),
        "x_val_rows": int(x_val.shape[0]),
        "ensemble_pred_rows": int(len(ensemble_pred)),
        "all_model_pred_rows_match": True,
        "all_rule_mask_rows_match": True,
    }


def fit_lightgbm_metrics(
    train_full: pd.DataFrame,
    val_full: pd.DataFrame,
    feature_cols: list[str],
    *,
    label_shuffle_seed: int | None = None,
) -> dict[str, float | int]:
    y_train = train_full["anomaly"]
    if label_shuffle_seed is None:
        y_fit = y_train
    else:
        y_fit = y_train.sample(frac=1, random_state=label_shuffle_seed)
        y_fit.index = y_train.index
    ds_idx = downsample_indices(y_fit)
    # Preserved for M3 numeric parity with the original script path.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[ds_idx, feature_cols])
    x_val = scaler.transform(val_full[feature_cols])
    model = lgb.LGBMClassifier(
        n_estimators=100,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_fit.loc[ds_idx])
    pred = model.predict_proba(x_val)[:, 1]
    return {
        **classification_metrics(val_full["anomaly"], pred),
        "n_train_downsampled": int(len(ds_idx)),
        "label_shuffle_seed": label_shuffle_seed,
    }


def run_label_shuffle_diagnostics(
    train_full: pd.DataFrame,
    val_full: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, object]:
    log("Running LightGBM label-shuffle diagnostics")
    runs = {}
    for seed in SHUFFLE_SEEDS:
        metrics = fit_lightgbm_metrics(
            train_full,
            val_full,
            feature_cols,
            label_shuffle_seed=seed,
        )
        runs[str(seed)] = metrics
        log(f"  shuffle seed {seed}: AUC={metrics['val_auc']:.4f}")
    aucs = [run["val_auc"] for run in runs.values()]
    return {
        "model": "lightgbm",
        "purpose": "base_rate_memorization_check",
        "runs": runs,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "min_auc": float(np.min(aucs)),
        "max_auc": float(np.max(aucs)),
    }


def run_site_heldout_ensemble(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, object]:
    log("Running site-held-out 4-model ensemble")
    mask_val = (df["site_id"] % 5 == 4).to_numpy()
    train_sites = set(df.loc[~mask_val, "site_id"].unique())
    val_sites = set(df.loc[mask_val, "site_id"].unique())
    site_overlap = train_sites & val_sites
    if site_overlap:
        raise AssertionError(f"site overlap: {sorted(site_overlap)}")
    train_buildings = set(df.loc[~mask_val, "building_id"].unique())
    val_buildings = set(df.loc[mask_val, "building_id"].unique())
    building_overlap = train_buildings & val_buildings
    if building_overlap:
        raise AssertionError(f"building overlap: {sorted(building_overlap)[:5]}")

    train_full = add_value_change_features(df.loc[~mask_val], list(SHIFTS))
    val_full = add_value_change_features(df.loc[mask_val], list(SHIFTS))
    if any(col not in train_full.columns for col in feature_cols):
        raise AssertionError("Site-held-out features do not match canonical features")
    y_train = train_full["anomaly"]
    y_val = val_full["anomaly"]
    ds_idx = downsample_indices(y_train)
    # Preserved for M3 numeric parity with the original script path.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[ds_idx, feature_cols])
    x_val = scaler.transform(val_full[feature_cols])
    run = fit_predict_models(
        x_train,
        y_train.loc[ds_idx],
        x_val,
        y_val,
        RANDOM_STATE,
    )
    return {
        "split": "site_id_mod5_eq_4",
        "val_sites": sorted(int(site) for site in val_sites),
        "n_train_sites": int(len(train_sites)),
        "n_val_sites": int(len(val_sites)),
        "n_train_buildings": int(len(train_buildings)),
        "n_val_buildings": int(len(val_buildings)),
        "n_train_rows": int((~mask_val).sum()),
        "n_val_rows": int(mask_val.sum()),
        "train_anomaly_rate": float(df.loc[~mask_val, "anomaly"].mean()),
        "val_anomaly_rate": float(df.loc[mask_val, "anomaly"].mean()),
        "site_overlap": int(len(site_overlap)),
        "building_overlap": int(len(building_overlap)),
        "n_train_downsampled": int(len(ds_idx)),
        "run": run,
    }


def meter_auc_breakdown(y_true: pd.Series, pred: np.ndarray, meter: pd.Series) -> dict:
    out = {}
    for meter_id, name in METER_NAMES.items():
        mask = meter.to_numpy() == meter_id
        y_meter = y_true.to_numpy()[mask]
        if len(np.unique(y_meter)) < 2:
            auc = None
        else:
            auc = float(roc_auc_score(y_meter, pred[mask]))
        out[name] = {
            "meter": int(meter_id),
            "n_rows": int(mask.sum()),
            "n_anomalies": int(y_meter.sum()),
            "anomaly_rate": float(y_meter.mean()) if len(y_meter) else 0.0,
            "val_auc": auc,
        }
    return out


def value_change_gap_diagnostics(df: pd.DataFrame) -> dict[str, object]:
    stats = (
        df.groupby("building_id", sort=True)["timestamp"]
        .agg(["min", "max", "nunique"])
        .rename(columns={"nunique": "n_observed_hours"})
    )
    expected_within_range = (
        (stats["max"] - stats["min"]).dt.total_seconds() // 3600 + 1
    ).astype("int64")
    stats["expected_hours_within_range"] = expected_within_range
    stats["missing_hours_within_range"] = (
        stats["expected_hours_within_range"] - stats["n_observed_hours"]
    )
    stats["missing_hours_vs_full_2016"] = 8784 - stats["n_observed_hours"]
    buildings_with_gaps = stats["missing_hours_within_range"] > 0
    non_full_year = stats["n_observed_hours"] < 8784
    top_missing = (
        stats.sort_values(
            ["missing_hours_within_range", "missing_hours_vs_full_2016"],
            ascending=False,
        )
        .head(10)
        .reset_index()
    )
    top_missing["min"] = top_missing["min"].astype(str)
    top_missing["max"] = top_missing["max"].astype(str)
    return {
        "n_buildings": int(len(stats)),
        "n_buildings_with_missing_within_range": int(buildings_with_gaps.sum()),
        "pct_buildings_with_missing_within_range": float(buildings_with_gaps.mean()),
        "n_buildings_non_full_8784_hours": int(non_full_year.sum()),
        "pct_buildings_non_full_8784_hours": float(non_full_year.mean()),
        "min_observed_hours": int(stats["n_observed_hours"].min()),
        "median_observed_hours": float(stats["n_observed_hours"].median()),
        "max_observed_hours": int(stats["n_observed_hours"].max()),
        "total_missing_hours_within_range": int(
            stats["missing_hours_within_range"].sum()
        ),
        "top_missing_buildings": top_missing.to_dict(orient="records"),
        "limitation": (
            "M3 value-change features still use groupby().shift(), so shifts are "
            "row-offset approximations across timestamp gaps rather than exact "
            "timestamp+timedelta merges."
        ),
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
    parser.add_argument(
        "--allow-null",
        action="store_true",
        help="Write results when hard rules produce negative delta AUC.",
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
    train_full = add_value_change_features(df.loc[~mask_val], list(SHIFTS))
    val_full = add_value_change_features(df.loc[mask_val], list(SHIFTS))
    value_cols = [c for c in train_full.columns if c.startswith("lag_value_")]
    feature_cols = BASELINE_FEATURE_COLS + value_cols
    if len(feature_cols) != 137:
        raise AssertionError(f"Expected 137 M3.2 features, got {len(feature_cols)}")
    if len(val_full) != int(mask_val.sum()):
        raise AssertionError("Validation feature rows are misaligned after FE")
    if "site_id" not in val_full.columns:
        raise AssertionError("site_id is required for M3.5 diagnostics")

    y_train = train_full["anomaly"]
    y_val = val_full["anomaly"]
    ds_idx = downsample_indices(y_train)
    log(f"Downsampled train rows: {len(ds_idx):,}")

    # Preserved for M3 numeric parity with the original script path.
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
            "site_id": val_full["site_id"].to_numpy(),
            "meter": val_full["meter"].to_numpy(),
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

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    val_prediction_frame.to_csv(args.predictions_out, index=False)
    log(f"Saved {args.predictions_out}")

    rule_1_mask = val_full["meter_reading"].to_numpy() == 1.0
    if int(rule_1_mask.sum()) == 0:
        raise RuntimeError("Rule 1 triggered 0 rows; stopping as requested")

    eda = jan1_eda(val_full, y_val)
    rule_2a_mask = eda.pop("rule_2a_mask")
    rule_2b_mask = val_full["dayofyear"].to_numpy() > RULE_2B_DAYOFYEAR_THRESHOLD
    alignment = validate_alignment(
        val_full=val_full,
        x_val=x_val,
        y_val=y_val,
        ensemble_pred=ensemble_pred,
        model_preds=model_preds,
        rule_masks={
            "rule_1": rule_1_mask,
            "rule_2b": rule_2b_mask,
            **({"rule_2a": rule_2a_mask} if rule_2a_mask is not None else {}),
        },
    )

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
    if combined_delta > 0.01:
        raise RuntimeError(
            f"Post-processing delta {combined_delta:+.6f} outside expected range"
        )
    if combined_delta < 0:
        warning = (
            f"Post-processing delta {combined_delta:+.6f} is negative; "
            "rules do not transfer to M3."
        )
        if not args.allow_null:
            raise RuntimeError(
                warning + " Re-run with --allow-null to persist results."
            )
        log(f"WARNING: {warning}")

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
    label_shuffle = run_label_shuffle_diagnostics(train_full, val_full, feature_cols)
    site_heldout = run_site_heldout_ensemble(df, feature_cols)
    per_meter_auc = meter_auc_breakdown(y_val, ensemble_pred, val_full["meter"])
    value_gap = value_change_gap_diagnostics(df)

    results: dict[str, object] = {
        "experiment": "m3_5_postprocessing",
        "canonical_line": "80_20_mod5_offline",
        "feature_set": "m3_2_137_features",
        "random_state": RANDOM_STATE,
        "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
        "postprocessing_base": "m3_4_seed_42_equal_weight_4_model_ensemble",
        "conclusion": "rules_do_not_transfer"
        if combined_delta < 0
        else "pending_review",
        "allow_null": bool(args.allow_null),
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
        "alignment_checks": alignment,
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
        "diagnostics": {
            "label_shuffle_lightgbm": label_shuffle,
            "site_heldout_ensemble": site_heldout,
            "per_meter_type_auc": per_meter_auc,
            "value_change_gap": value_gap,
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
