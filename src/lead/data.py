"""Data loading helpers for the LEAD/M3 reproduction pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[2]
M3 = ROOT / "data" / "raw" / "m3"
PROC = ROOT / "data" / "processed"

RANDOM_STATE = 42
DOWNSAMPLE_SEEDS = (10, 20)
MODEL_SEEDS = (42, 123, 999)
SHUFFLE_SEEDS = (42, 123, 999, 2025, 7)

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

BUILDING_META_FEATURE_COLS = ["log_square_feet", "year_built", "floor_count"]

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


def _log(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


def _assign_positional_labels(train: pd.DataFrame) -> pd.DataFrame:
    """Preserve the current M3 positional label assignment until M4.2."""
    bad = pd.read_csv(M3 / "bad_meter_readings.csv")
    if len(bad) != len(train):
        raise ValueError("bad_meter_readings.csv must align 1:1 with train.csv")
    train["anomaly"] = bad["is_bad_meter_reading"].values.astype("int8")
    return train


def _add_time_features(train: pd.DataFrame) -> pd.DataFrame:
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    train["hour"] = train["timestamp"].dt.hour.astype("int8")
    train["weekday"] = train["timestamp"].dt.weekday.astype("int8")
    train["month"] = train["timestamp"].dt.month.astype("int8")
    train["dayofyear"] = (
        train["timestamp"].dt.dayofyear + train["timestamp"].dt.hour / 24
    ).astype("float32")
    return train


def _add_cyclic_and_holiday_features(train: pd.DataFrame) -> pd.DataFrame:
    train["hour_sin"] = np.sin(2 * np.pi * train["hour"] / 24).astype("float32")
    train["hour_cos"] = np.cos(2 * np.pi * train["hour"] / 24).astype("float32")
    train["weekday_sin"] = np.sin(2 * np.pi * train["weekday"] / 7).astype("float32")
    train["weekday_cos"] = np.cos(2 * np.pi * train["weekday"] / 7).astype("float32")
    train["month_sin"] = np.sin(2 * np.pi * (train["month"] - 1) / 12).astype("float32")
    train["month_cos"] = np.cos(2 * np.pi * (train["month"] - 1) / 12).astype("float32")
    us_holidays = holidays.country_holidays("US", years=[2016])
    train["is_holiday"] = train["timestamp"].dt.date.isin(us_holidays).astype("int8")
    return train


def _building_metadata() -> pd.DataFrame:
    meta = pd.read_csv(M3 / "building_metadata.csv")
    le = LabelEncoder()
    meta["primary_use_enc"] = le.fit_transform(
        meta["primary_use"].fillna("Unknown")
    ).astype("int8")
    meta["log_square_feet"] = np.log1p(meta["square_feet"]).astype("float32")
    return meta


def _add_weather_rolls(weather: pd.DataFrame) -> pd.DataFrame:
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


def _weather_frame(*, include_budslab_features: bool) -> pd.DataFrame:
    weather = pd.read_csv(M3 / "weather_train.csv")
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather["cloud_coverage"] = (
        weather["cloud_coverage"].replace({255: 10}).astype("float32")
    )
    if include_budslab_features:
        weather = _add_weather_rolls(weather)
    return weather


def load_m3_frame(
    *,
    include_budslab_features: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the current M3 frame.

    This intentionally preserves M3's positional label assignment. M4.2 owns
    the key-aligned label-join change.
    """
    t0 = time.time()
    train = pd.read_csv(
        M3 / "train.csv",
        dtype={"building_id": "int16", "meter": "int8", "meter_reading": "float32"},
    )
    train = _assign_positional_labels(train)
    train = _add_time_features(train)
    if include_budslab_features:
        train = _add_cyclic_and_holiday_features(train)

    meta = _building_metadata()
    meta_cols = [
        "building_id",
        "site_id",
        "primary_use_enc",
        "log_square_feet",
        "year_built",
        "floor_count",
    ]
    if include_budslab_features:
        meta_cols.insert(2, "primary_use")
    train = train.merge(meta[meta_cols], on="building_id", how="left")

    if include_budslab_features:
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

    weather = _weather_frame(include_budslab_features=include_budslab_features)
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
    if include_budslab_features:
        weather_cols.extend(
            feature
            for col in WEATHER_LAG_BASE_COLS
            for window in WEATHER_WINDOWS
            for feature in (f"{col}_lag_{window}", f"{col}_rollmean_{window}")
        )
    train = train.merge(weather[weather_cols], on=["site_id", "timestamp"], how="left")

    keep_cols = [
        "building_id",
        "site_id",
        "timestamp",
        "anomaly",
        *BASELINE_FEATURE_COLS,
    ]
    if include_budslab_features:
        keep_cols.extend(M3_3_EXTRA_FEATURE_COLS)
    keep_cols = list(dict.fromkeys(keep_cols))
    label = "M3.3 frame" if include_budslab_features else "M3 frame"
    _log(
        f"Loaded {label} {train.shape} in {(time.time() - t0) / 60:.1f} min",
        verbose=verbose,
    )
    return train[keep_cols]
