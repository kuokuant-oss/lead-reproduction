"""Run M3 PI-response split/causality experiments.

This reuses the M3.2 17 baseline features and 60 value-change shifts, then
compares building split protocol and value-change shift regime.
"""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lead import (
    BASELINE_FEATURE_COLS,
    DOWNSAMPLE_SEEDS,
    FUTURE_SHIFTS,
    PAST_SHIFTS,
    PROC,
    RANDOM_STATE,
    SHIFTS,
    add_value_change_features,
    assert_no_building_overlap,
    classification_metrics,
    downsample_indices,
    load_m3_frame,
    split_mask,
)


def log(message: str) -> None:
    print(message, flush=True)


def fit_eval(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    shuffle_labels: bool = False,
) -> dict[str, float | int]:
    y_train = train_df["anomaly"]
    y_fit = y_train.sample(frac=1, random_state=RANDOM_STATE)
    y_fit.index = y_train.index
    if not shuffle_labels:
        y_fit = y_train

    ds_idx = downsample_indices(y_fit)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df.loc[ds_idx, feature_cols])
    x_val = scaler.transform(val_df[feature_cols])

    model = lgb.LGBMClassifier(
        n_estimators=100,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_fit.loc[ds_idx])
    pred = model.predict_proba(x_val)[:, 1]
    metrics = classification_metrics(val_df["anomaly"], pred)
    return {
        **metrics,
        "n_train_downsampled": int(len(ds_idx)),
    }


def run_split(df: pd.DataFrame, split_name: str, regimes: list[str]) -> dict[str, dict]:
    t0 = time.time()
    mask_val = split_mask(df, split_name)
    train_buildings = set(df.loc[~mask_val, "building_id"].unique())
    val_buildings = set(df.loc[mask_val, "building_id"].unique())
    overlap = assert_no_building_overlap(
        train_buildings, val_buildings, split_name=split_name
    )

    max_shifts = SHIFTS if "offline" in regimes else PAST_SHIFTS
    train_split = add_value_change_features(df.loc[~mask_val], max_shifts)
    val_split = add_value_change_features(df.loc[mask_val], max_shifts)
    value_cols = [c for c in train_split.columns if c.startswith("lag_value_")]
    past_cols = [c for c in value_cols if "_-" not in c]
    future_cols = [c for c in value_cols if "_-" in c]
    metadata = {
        "n_train_buildings": int(len(train_buildings)),
        "n_val_buildings": int(len(val_buildings)),
        "n_train_rows": int((~mask_val).sum()),
        "n_val_rows": int(mask_val.sum()),
        "train_anomaly_rate": float(df.loc[~mask_val, "anomaly"].mean()),
        "val_anomaly_rate": float(df.loc[mask_val, "anomaly"].mean()),
        "building_overlap": int(len(overlap)),
        "elapsed_feature_minutes": round((time.time() - t0) / 60, 3),
    }
    log(
        f"{split_name}: {metadata['n_train_buildings']} train buildings, "
        f"{metadata['n_val_buildings']} val buildings, "
        f"{metadata['train_anomaly_rate']:.4f}/{metadata['val_anomaly_rate']:.4f} anomaly"
    )

    out = {}
    for regime in regimes:
        cols = BASELINE_FEATURE_COLS + (
            value_cols if regime == "offline" else past_cols
        )
        log(f"  fitting {split_name} / {regime} with {len(cols)} features")
        metrics = fit_eval(train_split, val_split, cols)
        out[regime] = {
            **metadata,
            **metrics,
            "n_features": int(len(cols)),
            "n_value_change_features": int(len(cols) - len(BASELINE_FEATURE_COLS)),
            "n_past_shift_features": int(len(past_cols)),
            "n_future_shift_features_available": int(len(future_cols)),
        }
        log(
            f"    AUC={metrics['val_auc']:.4f} "
            f"P/R/F1={metrics['precision_05']:.4f}/"
            f"{metrics['recall_05']:.4f}/{metrics['f1_05']:.4f}"
        )
    return out


def main() -> None:
    if len(SHIFTS) != 60 or len(PAST_SHIFTS) != 30 or len(FUTURE_SHIFTS) != 30:
        raise AssertionError("Unexpected value-change shift set")

    df = load_m3_frame()
    results = {
        "experiment": "m3_split_causality",
        "random_state": RANDOM_STATE,
        "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
        "baseline_feature_count": len(BASELINE_FEATURE_COLS),
        "shift_set": SHIFTS,
        "past_shifts": PAST_SHIFTS,
        "future_shifts": FUTURE_SHIFTS,
        "splits": {},
        "sanity_checks": {},
        "robustness_checks": {},
    }

    results["splits"]["80_20_mod5"] = run_split(df, "80_20_mod5", ["offline", "causal"])
    results["splits"]["50_50_mod2"] = run_split(df, "50_50_mod2", ["offline", "causal"])

    log("Running seeded-random 50/50 robustness check (offline)")
    results["robustness_checks"]["50_50_random42"] = run_split(
        df, "50_50_random42", ["offline"]
    )["offline"]

    log("Running 50/50 causal label-shuffle sanity check")
    mask_val = split_mask(df, "50_50_mod2")
    train_split = add_value_change_features(df.loc[~mask_val], PAST_SHIFTS)
    val_split = add_value_change_features(df.loc[mask_val], PAST_SHIFTS)
    past_cols = [c for c in train_split.columns if c.startswith("lag_value_")]
    shuffle_metrics = fit_eval(
        train_split,
        val_split,
        BASELINE_FEATURE_COLS + past_cols,
        shuffle_labels=True,
    )
    results["sanity_checks"]["50_50_mod2_causal_label_shuffle"] = shuffle_metrics
    log(f"  label-shuffle AUC={shuffle_metrics['val_auc']:.4f}")

    out = PROC / "m3_split_causality_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"Saved {out}")


if __name__ == "__main__":
    main()
