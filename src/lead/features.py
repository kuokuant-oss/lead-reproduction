"""Feature engineering helpers."""

from __future__ import annotations

import warnings
from typing import Literal

import pandas as pd


ValueChangeRegime = Literal["row_offset", "row_offset_meter_aware", "timestamp_merge"]


def _row_offset_shifted(
    out: pd.DataFrame, shift_hours: int, *, meter_aware: bool
) -> pd.Series:
    group_keys = ["building_id", "meter"] if meter_aware else ["building_id"]
    missing = [key for key in group_keys if key not in out.columns]
    if missing:
        raise ValueError(
            "meter-aware row-offset value-change requires columns: "
            + ", ".join(missing)
        )
    grouped = out.groupby(group_keys, sort=False)["meter_reading"]
    return grouped.shift(shift_hours)


def _timestamp_merge_shifted(out: pd.DataFrame, shift_hours: int) -> pd.Series:
    join_keys = ["building_id", "timestamp"]
    if "meter" in out.columns:
        join_keys.insert(1, "meter")
    shifted = out[[*join_keys, "meter_reading"]].copy()
    shifted["timestamp"] = shifted["timestamp"] + pd.Timedelta(hours=shift_hours)
    shifted = shifted.rename(columns={"meter_reading": "_shifted_meter_reading"})
    merged = out[join_keys].merge(
        shifted,
        on=join_keys,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    return merged["_shifted_meter_reading"]


def _assign_feature_column(frame: pd.DataFrame, name: str, values: pd.Series) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        frame[name] = values


def add_value_change_features(
    df: pd.DataFrame,
    shifts: list[int],
    *,
    value_change_regime: ValueChangeRegime = "timestamp_merge",
) -> pd.DataFrame:
    """Add value-change features under an explicit offset regime.

    The default now uses the buds-lab-faithful timestamp + n-hour join.
    `row_offset` and `row_offset_meter_aware` remain available for historical
    ablation runs. Merge misses stay NaN for LightGBM's native missing-value
    handling.
    """
    if value_change_regime not in (
        "row_offset",
        "row_offset_meter_aware",
        "timestamp_merge",
    ):
        raise ValueError(
            "value_change_regime must be one of: row_offset, "
            "row_offset_meter_aware, timestamp_merge"
        )
    sort_keys = ["building_id", "timestamp"]
    if value_change_regime == "row_offset_meter_aware":
        if "meter" not in df.columns:
            raise ValueError(
                "meter-aware row-offset value-change requires columns: meter"
            )
        sort_keys = ["building_id", "meter", "timestamp"]
    out = df.sort_values(sort_keys).reset_index(drop=True).copy()
    mr = out["meter_reading"]
    for n in shifts:
        if value_change_regime == "timestamp_merge":
            shifted = _timestamp_merge_shifted(out, n)
        else:
            shifted = _row_offset_shifted(
                out,
                n,
                meter_aware=value_change_regime == "row_offset_meter_aware",
            )
        # Assign each completed column directly. Keeping all 120 arrays in a
        # dictionary and concatenating at the end makes pandas allocate a
        # second ~4.5 GiB block for a full M5 half while the originals are
        # still live. Direct assignment preserves values and insertion order
        # without that full-matrix transient copy.
        _assign_feature_column(
            out,
            f"lag_value_diff_{n}",
            (mr - shifted).astype("float32"),
        )
        _assign_feature_column(
            out,
            f"lag_value_ratio_{n}",
            ((mr + 1) / (shifted + 1)).astype("float32"),
        )
    return out
