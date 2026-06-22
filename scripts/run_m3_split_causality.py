"""Run M3 PI-response split/causality experiments.

This reuses the M3.2 17 baseline features and 60 value-change shifts, then
compares building split protocol and value-change shift regime.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
M3 = ROOT / "data" / "raw" / "m3"
PROC = ROOT / "data" / "processed"

RANDOM_STATE = 42
DOWNSAMPLE_SEEDS = (10, 20)

BASELINE_FEATURE_COLS = [
    "meter",
    "meter_reading",
    "hour",
    "weekday",
    "month",
    "dayofyear",
    "primary_use_enc",
    "log_square_feet",
    "year_built",
    "floor_count",
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
]

SHIFTS = (
    list(range(-24, 0))
    + list(range(1, 25))
    + list(range(-168, -24, 24))
    + list(range(48, 169, 24))
)
PAST_SHIFTS = [n for n in SHIFTS if n > 0]
FUTURE_SHIFTS = [n for n in SHIFTS if n < 0]


def log(message: str) -> None:
    print(message, flush=True)


def load_m3_frame() -> pd.DataFrame:
    t0 = time.time()
    train = pd.read_csv(
        M3 / "train.csv",
        dtype={"building_id": "int16", "meter": "int8", "meter_reading": "float32"},
    )
    bad = pd.read_csv(M3 / "bad_meter_readings.csv")
    if len(bad) != len(train):
        raise ValueError("bad_meter_readings.csv must align 1:1 with train.csv")
    train["anomaly"] = bad["is_bad_meter_reading"].values.astype("int8")

    train["timestamp"] = pd.to_datetime(train["timestamp"])
    train["hour"] = train["timestamp"].dt.hour.astype("int8")
    train["weekday"] = train["timestamp"].dt.weekday.astype("int8")
    train["month"] = train["timestamp"].dt.month.astype("int8")
    train["dayofyear"] = (
        train["timestamp"].dt.dayofyear + train["timestamp"].dt.hour / 24
    ).astype("float32")

    meta = pd.read_csv(M3 / "building_metadata.csv")
    le = LabelEncoder()
    meta["primary_use_enc"] = le.fit_transform(
        meta["primary_use"].fillna("Unknown")
    ).astype("int8")
    meta["log_square_feet"] = np.log1p(meta["square_feet"]).astype("float32")
    meta_cols = [
        "building_id",
        "site_id",
        "primary_use_enc",
        "log_square_feet",
        "year_built",
        "floor_count",
    ]
    train = train.merge(meta[meta_cols], on="building_id", how="left")

    weather = pd.read_csv(M3 / "weather_train.csv")
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather["cloud_coverage"] = (
        weather["cloud_coverage"].replace({255: 10}).astype("float32")
    )
    weather_cols = [
        "site_id",
        "timestamp",
        "air_temperature",
        "cloud_coverage",
        "dew_temperature",
        "precip_depth_1_hr",
        "sea_level_pressure",
        "wind_direction",
        "wind_speed",
    ]
    train = train.merge(weather[weather_cols], on=["site_id", "timestamp"], how="left")
    keep_cols = ["building_id", "timestamp", "anomaly"] + BASELINE_FEATURE_COLS
    keep_cols = list(dict.fromkeys(keep_cols))
    log(f"Loaded M3 frame {train.shape} in {(time.time() - t0) / 60:.1f} min")
    return train[keep_cols]


def split_mask(df: pd.DataFrame, split_name: str) -> np.ndarray:
    building_ids = df["building_id"].drop_duplicates().to_numpy()
    if split_name == "80_20_mod5":
        return (df["building_id"] % 5 == 4).to_numpy()
    if split_name == "50_50_mod2":
        return (df["building_id"] % 2 == 1).to_numpy()
    if split_name == "50_50_random42":
        rng = np.random.RandomState(RANDOM_STATE)
        shuffled = building_ids.copy()
        rng.shuffle(shuffled)
        n_train = len(shuffled) // 2
        train_buildings = set(int(x) for x in shuffled[:n_train])
        return ~df["building_id"].isin(train_buildings).to_numpy()
    raise ValueError(f"Unknown split: {split_name}")


def add_value_change_features(df: pd.DataFrame, shifts: list[int]) -> pd.DataFrame:
    out = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True).copy()
    mr = out["meter_reading"]
    grouped = out.groupby("building_id")["meter_reading"]
    new_cols = {}
    for n in shifts:
        shifted = grouped.shift(n)
        new_cols[f"lag_value_diff_{n}"] = (mr - shifted).astype("float32")
        new_cols[f"lag_value_ratio_{n}"] = ((mr + 1) / (shifted + 1)).astype("float32")
    return pd.concat([out, pd.DataFrame(new_cols)], axis=1)


def downsample_indices(y: pd.Series) -> np.ndarray:
    neg_idx = y.index[y == 0].to_numpy()
    pos_idx = y.index[y == 1].to_numpy()
    n_pos = len(pos_idx)
    negs1 = np.random.RandomState(DOWNSAMPLE_SEEDS[0]).choice(
        neg_idx, n_pos, replace=False
    )
    negs2 = np.random.RandomState(DOWNSAMPLE_SEEDS[1]).choice(
        neg_idx, n_pos, replace=False
    )
    return np.concatenate([negs1, pos_idx, negs2, pos_idx])


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
    pred_label = (pred >= 0.5).astype("int8")
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_df["anomaly"], pred_label, average="binary", pos_label=1, zero_division=0
    )
    return {
        "val_auc": float(roc_auc_score(val_df["anomaly"], pred)),
        "precision_05": float(precision),
        "recall_05": float(recall),
        "f1_05": float(f1),
        "n_train_downsampled": int(len(ds_idx)),
    }


def run_split(df: pd.DataFrame, split_name: str, regimes: list[str]) -> dict[str, dict]:
    t0 = time.time()
    mask_val = split_mask(df, split_name)
    train_buildings = set(df.loc[~mask_val, "building_id"].unique())
    val_buildings = set(df.loc[mask_val, "building_id"].unique())
    overlap = train_buildings & val_buildings
    if overlap:
        raise AssertionError(
            f"{split_name} has building overlap: {sorted(overlap)[:5]}"
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
