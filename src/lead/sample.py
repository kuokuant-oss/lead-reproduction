"""Sampling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DOWNSAMPLE_SEEDS


def downsample_indices(y: pd.Series) -> np.ndarray:
    """Return the M3-compatible ``[negs1, pos, negs2, pos]`` fit index.

    Positive rows are intentionally duplicated in both positive blocks. This is
    a reproduction-compatibility choice that preserves the original solution's
    effective 50:50 fit set without changing the accepted M3 numeric line.
    """
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
