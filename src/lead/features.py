"""Feature engineering helpers."""

from __future__ import annotations

import pandas as pd


def add_value_change_features(df: pd.DataFrame, shifts: list[int]) -> pd.DataFrame:
    """Add the current row-offset value-change features.

    This intentionally preserves M3's `groupby().shift()` semantics. Timestamp
    merge semantics are deferred to M4.3.
    """
    out = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True).copy()
    mr = out["meter_reading"]
    grouped = out.groupby("building_id", sort=False)["meter_reading"]
    new_cols = {}
    for n in shifts:
        shifted = grouped.shift(n)
        new_cols[f"lag_value_diff_{n}"] = (mr - shifted).astype("float32")
        new_cols[f"lag_value_ratio_{n}"] = ((mr + 1) / (shifted + 1)).astype("float32")
    return pd.concat([out, pd.DataFrame(new_cols)], axis=1)
