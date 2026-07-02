"""INV-5: compare causal building split with same-building time holdout."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from lead import (
    BASELINE_FEATURE_COLS,
    DOWNSAMPLE_SEEDS,
    PAST_SHIFTS,
    PROC,
    RANDOM_STATE,
    ROOT,
    add_value_change_features,
    assert_no_building_overlap,
    classification_metrics,
    downsample_indices,
    load_m3_frame,
    write_json_with_provenance,
)
from run_m3_4_ensemble import fit_predict_models


VALUE_CHANGE_REGIME = "row_offset_meter_aware"


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "inv5_time_holdout.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def split_frames(df, split_name: str):
    if split_name == "80_20_mod5_causal":
        mask_val = (df["building_id"] % 5 == 4).to_numpy()
        return df.loc[~mask_val], df.loc[mask_val]
    if split_name == "time_holdout_2016_causal":
        timestamps = df["timestamp"]
        train = df.loc[(timestamps >= "2016-01-01") & (timestamps < "2016-09-01")]
        val = df.loc[(timestamps >= "2016-09-01") & (timestamps < "2017-01-01")]
        return train, val
    raise ValueError(f"Unknown split: {split_name}")


def split_metadata(train_df, val_df, split_name: str) -> dict[str, Any]:
    train_buildings = set(train_df["building_id"].unique())
    val_buildings = set(val_df["building_id"].unique())
    overlap = train_buildings & val_buildings
    if split_name == "80_20_mod5_causal":
        overlap = assert_no_building_overlap(
            train_buildings,
            val_buildings,
            split_name=split_name,
        )
    return {
        "name": split_name,
        "n_train_buildings": int(len(train_buildings)),
        "n_val_buildings": int(len(val_buildings)),
        "building_overlap": int(len(overlap)),
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "train_anomaly_rate": float(train_df["anomaly"].mean()),
        "val_anomaly_rate": float(val_df["anomaly"].mean()),
        "train_timestamp_min": str(train_df["timestamp"].min()),
        "train_timestamp_max": str(train_df["timestamp"].max()),
        "val_timestamp_min": str(val_df["timestamp"].min()),
        "val_timestamp_max": str(val_df["timestamp"].max()),
    }


def fit_single(train_full, val_full, feature_cols: list[str]) -> dict[str, Any]:
    y_train = train_full["anomaly"]
    y_val = val_full["anomaly"]
    ds_idx = downsample_indices(y_train)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[ds_idx, feature_cols])
    x_val = scaler.transform(val_full[feature_cols])
    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=100,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train.loc[ds_idx])
    pred = model.predict_proba(x_val)[:, 1]
    return {
        **classification_metrics(y_val, pred),
        "n_train_downsampled": int(len(ds_idx)),
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }


def fit_ensemble(train_full, val_full, feature_cols: list[str]) -> dict[str, Any]:
    y_train = train_full["anomaly"]
    y_val = val_full["anomaly"]
    ds_idx = downsample_indices(y_train)
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
    run["n_train_downsampled"] = int(len(ds_idx))
    return run


def run_split(df, split_name: str) -> dict[str, Any]:
    log(f"Split: {split_name}")
    t0 = time.time()
    train_df, val_df = split_frames(df, split_name)
    metadata = split_metadata(train_df, val_df, split_name)
    train_full = add_value_change_features(
        train_df,
        list(PAST_SHIFTS),
        value_change_regime=VALUE_CHANGE_REGIME,
    )
    val_full = add_value_change_features(
        val_df,
        list(PAST_SHIFTS),
        value_change_regime=VALUE_CHANGE_REGIME,
    )
    value_cols = [col for col in train_full.columns if col.startswith("lag_value_")]
    feature_cols = BASELINE_FEATURE_COLS + value_cols
    if len(feature_cols) != 77:
        raise AssertionError(f"Expected 77 causal features, got {len(feature_cols)}")

    log("  fitting single LightGBM")
    single = fit_single(train_full, val_full, feature_cols)
    log(f"    single AUC={single['val_auc']:.6f}")

    log("  fitting 4-model ensemble")
    ensemble = fit_ensemble(train_full, val_full, feature_cols)
    log(f"    ensemble AUC={ensemble['ensemble']['val_auc']:.6f}")
    return {
        "split": metadata,
        "feature_counts": {
            "baseline": int(len(BASELINE_FEATURE_COLS)),
            "past_value_change": int(len(value_cols)),
            "total": int(len(feature_cols)),
        },
        "single_lightgbm": single,
        "ensemble": ensemble,
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    df = load_m3_frame(verbose=True)
    splits = {
        name: run_split(df, name)
        for name in ("80_20_mod5_causal", "time_holdout_2016_causal")
    }
    building = splits["80_20_mod5_causal"]
    time_holdout = splits["time_holdout_2016_causal"]
    comparison = {
        "single_lightgbm_delta_time_minus_building": float(
            time_holdout["single_lightgbm"]["val_auc"]
            - building["single_lightgbm"]["val_auc"]
        ),
        "ensemble_delta_time_minus_building": float(
            time_holdout["ensemble"]["ensemble"]["val_auc"]
            - building["ensemble"]["ensemble"]["val_auc"]
        ),
        "interpretation": "pending",
    }
    if comparison["ensemble_delta_time_minus_building"] < -0.0005:
        comparison["interpretation"] = "time_holdout_lower_needs_report_caveat"
    else:
        comparison["interpretation"] = "time_holdout_not_lower_beyond_noise_floor"

    results = {
        "experiment": "inv5_time_holdout",
        "issue": 53,
        "scope": (
            "Compare causal same-building time holdout against causal building "
            "holdout under opt-in meter-aware value-change features."
        ),
        "value_change_regime": VALUE_CHANGE_REGIME,
        "shift_set": "PAST_SHIFTS",
        "random_state": RANDOM_STATE,
        "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
        "splits": splits,
        "comparison": comparison,
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }
    write_json_with_provenance(
        args.out,
        results,
        root=ROOT,
        provenance={
            "command": f"uv run python scripts/run_inv5_time_holdout.py --out {args.out}",
        },
    )
    log(f"Saved {args.out}")
    log(f"Comparison: {comparison}")


if __name__ == "__main__":
    main()
