"""Train/validation split helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RANDOM_STATE


def split_mask(df: pd.DataFrame, split_name: str) -> np.ndarray:
    """Return a validation mask for the named building-level split."""
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


def assert_no_building_overlap(
    train_buildings: set[int],
    val_buildings: set[int],
    *,
    split_name: str,
) -> set[int]:
    """Assert that train/validation building sets are disjoint."""
    overlap = train_buildings & val_buildings
    if overlap:
        raise AssertionError(
            f"{split_name} has building overlap: {sorted(overlap)[:5]}"
        )
    return overlap


def leave_site_out_mask(df: pd.DataFrame, site_ids: list[int]) -> np.ndarray:
    """Reserved for M5/FDD site-held-out evaluation."""
    return df["site_id"].isin(site_ids).to_numpy()
