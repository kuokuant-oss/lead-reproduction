"""Measure M3.2 row-offset vs timestamp-merge value-change regimes."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
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
    classification_metrics,
    downsample_indices,
    load_m3_frame,
    write_json_with_provenance,
)

M3_2_GOLDEN_AUC = 0.9920
NOISE_FLOOR_AUC = 0.0005
REGIMES = ("row_offset", "timestamp_merge")


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "m4_3_timestamp_value_change.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def fit_m3_2_regime(df, mask_val, value_change_regime: str) -> dict[str, object]:
    log(f"Adding M3.2 value-change features: {value_change_regime}")
    t0 = time.time()
    train_full = add_value_change_features(
        df.loc[~mask_val],
        list(SHIFTS),
        value_change_regime=value_change_regime,
    )
    val_full = add_value_change_features(
        df.loc[mask_val],
        list(SHIFTS),
        value_change_regime=value_change_regime,
    )
    value_cols = [c for c in train_full.columns if c.startswith("lag_value_")]
    feature_cols = BASELINE_FEATURE_COLS + value_cols
    if len(feature_cols) != 137:
        raise AssertionError(f"Expected 137 M3.2 features, got {len(feature_cols)}")

    y_train = train_full["anomaly"]
    y_val = val_full["anomaly"]
    ds_idx = downsample_indices(y_train)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[ds_idx, feature_cols])
    x_val = scaler.transform(val_full[feature_cols])

    model = lgb.LGBMClassifier(
        n_estimators=100,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train.loc[ds_idx])
    pred = model.predict_proba(x_val)[:, 1]
    metrics = classification_metrics(y_val, pred)
    log(
        f"  {value_change_regime}: AUC={metrics['val_auc']:.6f} "
        f"P/R/F1={metrics['precision_05']:.4f}/"
        f"{metrics['recall_05']:.4f}/{metrics['f1_05']:.4f}"
    )
    return {
        "value_change_regime": value_change_regime,
        **metrics,
        "n_features": int(len(feature_cols)),
        "n_value_change_features": int(len(value_cols)),
        "n_train_downsampled": int(len(ds_idx)),
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    if len(SHIFTS) != 60:
        raise AssertionError("Unexpected value-change shift set")

    df = load_m3_frame()
    mask_val = (df["building_id"] % 5 == 4).to_numpy()
    train_buildings = set(df.loc[~mask_val, "building_id"].unique())
    val_buildings = set(df.loc[mask_val, "building_id"].unique())
    overlap = assert_no_building_overlap(
        train_buildings, val_buildings, split_name="80_20_mod5"
    )

    runs = {regime: fit_m3_2_regime(df, mask_val, regime) for regime in REGIMES}
    row_offset_auc = runs["row_offset"]["val_auc"]
    timestamp_merge_auc = runs["timestamp_merge"]["val_auc"]
    delta_regime = timestamp_merge_auc - row_offset_auc
    row_offset_delta_vs_golden = row_offset_auc - M3_2_GOLDEN_AUC

    if abs(row_offset_delta_vs_golden) > NOISE_FLOOR_AUC:
        gate_status = "failed_environment_sanity"
    elif abs(delta_regime) > NOISE_FLOOR_AUC:
        gate_status = "outside_noise_floor"
    else:
        gate_status = "within_noise_floor"

    results = {
        "experiment": "m4_3_timestamp_value_change",
        "canonical_line": "80_20_mod5_offline",
        "random_state": RANDOM_STATE,
        "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
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
            "value_change": int(len(SHIFTS) * 2),
            "total": int(len(BASELINE_FEATURE_COLS) + len(SHIFTS) * 2),
        },
        "regimes": runs,
        "comparison": {
            "m3_2_golden_auc": M3_2_GOLDEN_AUC,
            "noise_floor_auc": NOISE_FLOOR_AUC,
            "row_offset_auc": row_offset_auc,
            "timestamp_merge_auc": timestamp_merge_auc,
            "delta_regime_timestamp_minus_row": delta_regime,
            "row_offset_delta_vs_golden": row_offset_delta_vs_golden,
            "timestamp_merge_delta_vs_golden": timestamp_merge_auc - M3_2_GOLDEN_AUC,
            "gate_status": gate_status,
        },
        "interpretation": {
            "only_intended_difference": "row_offset_vs_timestamp_hour_offset",
            "nan_treatment": (
                "No regime-specific fill is applied; LightGBM sees NaN natively."
            ),
            "chosen_default_regime": "row_offset",
        },
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }
    write_json_with_provenance(
        args.out,
        results,
        root=ROOT,
        provenance={
            "command": "python scripts/run_m4_3_timestamp_value_change.py",
        },
    )
    log(f"Saved {args.out}")
    log(
        "Delta timestamp-row: "
        f"{delta_regime:+.6f}; row-offset vs golden: "
        f"{row_offset_delta_vs_golden:+.6f}; gate={gate_status}"
    )

    if gate_status == "failed_environment_sanity":
        raise RuntimeError(
            "Row-offset M3.2 rerun does not match golden within the noise floor"
        )


if __name__ == "__main__":
    main()
