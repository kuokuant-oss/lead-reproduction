from __future__ import annotations

import numpy as np
import pandas as pd
import unittest
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from m5_ek_steam_budget_runner import build_selected_timestamp_features  # noqa: E402
from run_m5_tree_ensemble_matched_context import (  # noqa: E402
    build_features_keeping_index,
)


class TestM5EkRunner(unittest.TestCase):
    def test_selected_timestamp_features_match_frozen_timestamp_merge(self) -> None:
        hours = 240
        source = pd.DataFrame(
            {
                "building_id": np.repeat([2, 4], hours),
                "meter": np.tile(np.repeat([2, 3], hours // 2), 2),
                "timestamp": pd.Timestamp("2016-01-01")
                + pd.to_timedelta(np.tile(np.arange(hours // 2), 4), unit="h"),
                "meter_reading": np.arange(2 * hours, dtype="float32"),
            },
            index=np.arange(1_000, 1_000 + 2 * hours, dtype="int64"),
        )
        raw_index = source.index.to_numpy(dtype="int64")[::17]
        expected = build_features_keeping_index(source.copy()).loc[raw_index]
        actual = build_selected_timestamp_features(source, raw_index)

        self.assertTrue(expected.index.equals(actual.index))
        for column in (name for name in expected if name.startswith("lag_value_")):
            self.assertTrue(
                np.array_equal(
                    expected[column].to_numpy(),
                    actual[column].to_numpy(),
                    equal_nan=True,
                ),
                column,
            )
