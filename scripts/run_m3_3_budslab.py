"""Run M3.3 buds-lab feature alignment on the canonical 80/20 offline line."""

from __future__ import annotations

import json
import time
from pathlib import Path

import holidays
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
MODEL_SEEDS = (42, 123, 999)
SHUFFLE_SEEDS = (42, 123, 999, 2025, 7)
BUILDING_META_FEATURE_COLS = ["log_square_feet", "year_built", "floor_count"]

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

CYCLIC_FEATURE_COLS = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
]

WEATHER_LAG_BASE_COLS = [
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_speed",
]
WEATHER_WINDOWS = (7, 73)

M3_3_EXTRA_FEATURE_COLS = [
    *CYCLIC_FEATURE_COLS,
    *[
        feature
        for col in WEATHER_LAG_BASE_COLS
        for window in WEATHER_WINDOWS
        for feature in (f"{col}_lag_{window}", f"{col}_rollmean_{window}")
    ],
    "is_holiday",
    "gte_site_meter_anomaly",
    "primary_use_meter_enc",
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


def add_weather_rolls(weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.sort_values(["site_id", "timestamp"]).copy()
    for col in WEATHER_LAG_BASE_COLS:
        for window in WEATHER_WINDOWS:
            lag_col = f"{col}_lag_{window}"
            out_col = f"{col}_rollmean_{window}"
            weather[lag_col] = (
                weather.groupby("site_id", sort=False)[col]
                .shift(window)
                .astype("float32")
            )
            weather[out_col] = (
                weather.groupby("site_id", sort=False)[col]
                .transform(lambda s: s.rolling(window, min_periods=1).mean())
                .astype("float32")
            )
    return weather


def fit_target_encoder(
    train_df: pd.DataFrame, val_df: pd.DataFrame, prior_weight: float = 20.0
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    global_mean = float(train_df["anomaly"].mean())
    stats = (
        train_df.groupby(["site_id", "meter"], sort=False)["anomaly"]
        .agg(["sum", "count"])
        .reset_index()
    )
    stats["gte_site_meter_anomaly"] = (
        (stats["sum"] + prior_weight * global_mean) / (stats["count"] + prior_weight)
    ).astype("float32")
    mapping = stats.set_index(["site_id", "meter"])["gte_site_meter_anomaly"]

    train_key = pd.MultiIndex.from_frame(train_df[["site_id", "meter"]])
    val_key = pd.MultiIndex.from_frame(val_df[["site_id", "meter"]])
    train_encoded = pd.Series(
        mapping.reindex(train_key).to_numpy(), index=train_df.index
    )
    val_encoded = pd.Series(mapping.reindex(val_key).to_numpy(), index=val_df.index)
    train_encoded = train_encoded.fillna(global_mean).astype("float32")
    val_encoded = val_encoded.fillna(global_mean).astype("float32")
    metadata = {
        "global_mean": global_mean,
        "prior_weight": float(prior_weight),
        "n_groups": int(len(stats)),
    }
    return train_encoded, val_encoded, metadata


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

    train["hour_sin"] = np.sin(2 * np.pi * train["hour"] / 24).astype("float32")
    train["hour_cos"] = np.cos(2 * np.pi * train["hour"] / 24).astype("float32")
    train["weekday_sin"] = np.sin(2 * np.pi * train["weekday"] / 7).astype("float32")
    train["weekday_cos"] = np.cos(2 * np.pi * train["weekday"] / 7).astype("float32")
    train["month_sin"] = np.sin(2 * np.pi * (train["month"] - 1) / 12).astype("float32")
    train["month_cos"] = np.cos(2 * np.pi * (train["month"] - 1) / 12).astype("float32")

    us_holidays = holidays.country_holidays("US", years=[2016])
    train["is_holiday"] = train["timestamp"].dt.date.isin(us_holidays).astype("int8")

    meta = pd.read_csv(M3 / "building_metadata.csv")
    le = LabelEncoder()
    meta["primary_use_enc"] = le.fit_transform(
        meta["primary_use"].fillna("Unknown")
    ).astype("int8")
    meta["log_square_feet"] = np.log1p(meta["square_feet"]).astype("float32")
    meta_cols = [
        "building_id",
        "site_id",
        "primary_use",
        "primary_use_enc",
        "log_square_feet",
        "year_built",
        "floor_count",
    ]
    train = train.merge(meta[meta_cols], on="building_id", how="left")

    train.loc[(train["site_id"] == 0) & (train["meter"] == 0), "meter_reading"] *= (
        0.2931
    )

    train["primary_use_meter"] = (
        train["primary_use"].fillna("Unknown") + "_" + train["meter"].astype(str)
    )
    interaction_le = LabelEncoder()
    train["primary_use_meter_enc"] = interaction_le.fit_transform(
        train["primary_use_meter"]
    ).astype("int16")
    train["gte_site_meter_anomaly"] = np.nan

    weather = pd.read_csv(M3 / "weather_train.csv")
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather["cloud_coverage"] = (
        weather["cloud_coverage"].replace({255: 10}).astype("float32")
    )
    weather = add_weather_rolls(weather)
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
        *[
            feature
            for col in WEATHER_LAG_BASE_COLS
            for window in WEATHER_WINDOWS
            for feature in (f"{col}_lag_{window}", f"{col}_rollmean_{window}")
        ],
    ]
    train = train.merge(weather[weather_cols], on=["site_id", "timestamp"], how="left")

    keep_cols = [
        "building_id",
        "site_id",
        "timestamp",
        "anomaly",
        *BASELINE_FEATURE_COLS,
        *M3_3_EXTRA_FEATURE_COLS,
    ]
    keep_cols = list(dict.fromkeys(keep_cols))
    log(f"Loaded M3.3 frame {train.shape} in {(time.time() - t0) / 60:.1f} min")
    return train[keep_cols]


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
    *,
    shuffle_labels: bool = False,
    shuffle_random_state: int = RANDOM_STATE,
    model_random_state: int = RANDOM_STATE,
) -> dict[str, float | int]:
    y_train = train_df["anomaly"]
    y_fit = y_train.sample(frac=1, random_state=shuffle_random_state)
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
        random_state=model_random_state,
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


def main() -> None:
    t0 = time.time()
    df = load_m3_frame()
    mask_val = (df["building_id"] % 5 == 4).to_numpy()
    train_buildings = set(df.loc[~mask_val, "building_id"].unique())
    val_buildings = set(df.loc[mask_val, "building_id"].unique())
    overlap = train_buildings & val_buildings
    if overlap:
        raise AssertionError(f"building overlap: {sorted(overlap)[:5]}")

    train_base = df.loc[~mask_val].copy()
    val_base = df.loc[mask_val].copy()
    train_gte, val_gte, gte_metadata = fit_target_encoder(train_base, val_base)
    train_base["gte_site_meter_anomaly"] = train_gte
    val_base["gte_site_meter_anomaly"] = val_gte

    log("Adding offline value-change features")
    train_full = add_value_change_features(train_base, SHIFTS)
    val_full = add_value_change_features(val_base, SHIFTS)
    value_cols = [c for c in train_full.columns if c.startswith("lag_value_")]
    past_cols = [c for c in value_cols if "_-" not in c]
    future_cols = [c for c in value_cols if "_-" in c]
    base_m3_3_cols = BASELINE_FEATURE_COLS + M3_3_EXTRA_FEATURE_COLS
    full_cols = base_m3_3_cols + value_cols
    past_only_cols = base_m3_3_cols + past_cols
    future_only_cols = base_m3_3_cols + future_cols

    results: dict[str, object] = {
        "experiment": "m3_3_budslab_alignment",
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
            "m3_2_baseline": int(len(BASELINE_FEATURE_COLS) + len(value_cols)),
            "m3_3_extra": int(len(M3_3_EXTRA_FEATURE_COLS)),
            "m3_3_full": int(len(full_cols)),
            "past_value_change": int(len(past_cols)),
            "future_value_change": int(len(future_cols)),
        },
        "features_added": {
            "cyclic": CYCLIC_FEATURE_COLS,
            "weather_lags_and_rollmeans": [
                feature
                for col in WEATHER_LAG_BASE_COLS
                for window in WEATHER_WINDOWS
                for feature in (f"{col}_lag_{window}", f"{col}_rollmean_{window}")
            ],
            "holiday": ["is_holiday"],
            "target_encoder": ["gte_site_meter_anomaly"],
            "building_interaction": ["primary_use_meter_enc"],
            "site0_meter0_correction": "meter_reading *= 0.2931 before value-change",
        },
        "target_encoder": gte_metadata,
        "m3_2_reference": {
            "val_auc": 0.9920,
            "precision_05": 0.6409,
            "recall_05": 0.9665,
            "f1_05": 0.7707,
        },
        "main": {},
        "sanity_checks": {},
        "label_shuffle_diagnostics": {},
    }

    log(f"Fitting M3.3 full with {len(full_cols)} features")
    main_metrics = fit_eval(train_full, val_full, full_cols)
    results["main"] = {
        **main_metrics,
        "n_features": int(len(full_cols)),
        "delta_auc_vs_m3_2": float(main_metrics["val_auc"] - 0.9920),
        "delta_precision_vs_m3_2": float(main_metrics["precision_05"] - 0.6409),
        "delta_recall_vs_m3_2": float(main_metrics["recall_05"] - 0.9665),
        "delta_f1_vs_m3_2": float(main_metrics["f1_05"] - 0.7707),
    }
    log(
        "M3.3 full: "
        f"AUC={main_metrics['val_auc']:.4f} "
        f"P/R/F1={main_metrics['precision_05']:.4f}/"
        f"{main_metrics['recall_05']:.4f}/{main_metrics['f1_05']:.4f}"
    )

    log("Running temporal leakage sanity check")
    temporal = {
        "past_only": fit_eval(train_full, val_full, past_only_cols),
        "future_only": fit_eval(train_full, val_full, future_only_cols),
        "full_offline": main_metrics,
    }
    results["sanity_checks"]["temporal_leakage"] = temporal

    log("Running label-shuffle sanity check across seeds")
    label_shuffle_runs = {}
    for seed in SHUFFLE_SEEDS:
        label_shuffle_runs[str(seed)] = fit_eval(
            train_full,
            val_full,
            full_cols,
            shuffle_labels=True,
            shuffle_random_state=seed,
        )
        log(
            f"  shuffle seed {seed}: AUC={label_shuffle_runs[str(seed)]['val_auc']:.4f}"
        )
    label_shuffle_aucs = [m["val_auc"] for m in label_shuffle_runs.values()]
    label_shuffle_summary = {
        "runs": label_shuffle_runs,
        "mean_auc": float(np.mean(label_shuffle_aucs)),
        "std_auc": float(np.std(label_shuffle_aucs)),
        "min_auc": float(np.min(label_shuffle_aucs)),
        "max_auc": float(np.max(label_shuffle_aucs)),
        "m3_2_label_shuffle_auc": 0.5669,
    }
    results["sanity_checks"]["label_shuffle"] = label_shuffle_summary

    log("Running label-shuffle ablations")
    no_gte_cols = [c for c in full_cols if c != "gte_site_meter_anomaly"]
    no_gte_no_building_meta_cols = [
        c for c in no_gte_cols if c not in BUILDING_META_FEATURE_COLS
    ]
    results["label_shuffle_diagnostics"] = {
        "full": label_shuffle_summary,
        "without_gte_site_meter_anomaly": {
            **fit_eval(
                train_full,
                val_full,
                no_gte_cols,
                shuffle_labels=True,
                shuffle_random_state=RANDOM_STATE,
            ),
            "n_features": int(len(no_gte_cols)),
            "removed_features": ["gte_site_meter_anomaly"],
        },
        "without_gte_and_building_meta": {
            **fit_eval(
                train_full,
                val_full,
                no_gte_no_building_meta_cols,
                shuffle_labels=True,
                shuffle_random_state=RANDOM_STATE,
            ),
            "n_features": int(len(no_gte_no_building_meta_cols)),
            "removed_features": ["gte_site_meter_anomaly", *BUILDING_META_FEATURE_COLS],
        },
    }

    log("Running multi-seed sanity check")
    multi_seed = {}
    for seed in MODEL_SEEDS:
        multi_seed[str(seed)] = fit_eval(
            train_full, val_full, full_cols, model_random_state=seed
        )
        log(f"  seed {seed}: AUC={multi_seed[str(seed)]['val_auc']:.4f}")
    seed_aucs = [m["val_auc"] for m in multi_seed.values()]
    results["sanity_checks"]["multi_seed"] = {
        "runs": multi_seed,
        "mean_auc": float(np.mean(seed_aucs)),
        "std_auc": float(np.std(seed_aucs)),
    }

    results["elapsed_minutes"] = round((time.time() - t0) / 60, 3)
    out = PROC / "m3_3_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"Saved {out}")


if __name__ == "__main__":
    main()
