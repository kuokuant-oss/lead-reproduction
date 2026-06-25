"""Feature engineering helpers."""

from __future__ import annotations

from typing import Literal

import pandas as pd


ValueChangeRegime = Literal["row_offset", "timestamp_merge"]


def _row_offset_shifted(out: pd.DataFrame, shift_hours: int) -> pd.Series:
    grouped = out.groupby("building_id", sort=False)["meter_reading"]
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


def add_value_change_features(
    df: pd.DataFrame,
    shifts: list[int],
    *,
    value_change_regime: ValueChangeRegime = "row_offset",
) -> pd.DataFrame:
    """Add value-change features under an explicit offset regime.

    The default preserves M3's `groupby().shift()` row-offset semantics.
    `timestamp_merge` uses exact timestamp + n-hour joins and leaves merge
    misses as NaN for LightGBM's native missing-value handling.
    """
    if value_change_regime not in ("row_offset", "timestamp_merge"):
        raise ValueError(
            "value_change_regime must be one of: row_offset, timestamp_merge"
        )
    out = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True).copy()
    mr = out["meter_reading"]
    new_cols = {}
    for n in shifts:
        if value_change_regime == "row_offset":
            shifted = _row_offset_shifted(out, n)
        else:
            shifted = _timestamp_merge_shifted(out, n)
        new_cols[f"lag_value_diff_{n}"] = (mr - shifted).astype("float32")
        new_cols[f"lag_value_ratio_{n}"] = ((mr + 1) / (shifted + 1)).astype("float32")
    return pd.concat([out, pd.DataFrame(new_cols)], axis=1)
