from __future__ import annotations

import unittest

import pandas as pd

from lead import add_value_change_features


def gapped_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "building_id": [1, 1, 1, 2, 2],
            "timestamp": pd.to_datetime(
                [
                    "2016-01-01 00:00:00",
                    "2016-01-01 01:00:00",
                    "2016-01-01 03:00:00",
                    "2016-01-01 00:00:00",
                    "2016-01-01 01:00:00",
                ]
            ),
            "meter_reading": [10.0, 12.0, 15.0, 20.0, 18.0],
        }
    )


def multi_meter_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "building_id": [1, 1, 1, 1],
            "meter": [0, 1, 0, 1],
            "timestamp": pd.to_datetime(
                [
                    "2016-01-01 00:00:00",
                    "2016-01-01 00:00:00",
                    "2016-01-01 01:00:00",
                    "2016-01-01 01:00:00",
                ]
            ),
            "meter_reading": [10.0, 100.0, 12.0, 105.0],
        }
    )


class TestValueChangeRegimes(unittest.TestCase):
    def test_row_offset_is_default_regime(self) -> None:
        df = gapped_frame()

        default = add_value_change_features(df, [1])
        explicit = add_value_change_features(df, [1], value_change_regime="row_offset")

        pd.testing.assert_frame_equal(default, explicit)

    def test_timestamp_merge_leaves_nan_across_timestamp_gaps(self) -> None:
        df = gapped_frame()

        row_offset = add_value_change_features(
            df, [1], value_change_regime="row_offset"
        )
        timestamp_merge = add_value_change_features(
            df, [1], value_change_regime="timestamp_merge"
        )

        gap_row = timestamp_merge["timestamp"] == pd.Timestamp("2016-01-01 03:00:00")
        self.assertFalse(row_offset.loc[gap_row, "lag_value_diff_1"].isna().item())
        self.assertTrue(timestamp_merge.loc[gap_row, "lag_value_diff_1"].isna().item())

    def test_timestamp_merge_preserves_exact_hour_matches(self) -> None:
        df = gapped_frame()

        out = add_value_change_features(df, [1], value_change_regime="timestamp_merge")

        matched_row = out["timestamp"] == pd.Timestamp("2016-01-01 01:00:00")
        self.assertEqual(out.loc[matched_row, "lag_value_diff_1"].iloc[0], 2.0)
        self.assertAlmostEqual(
            out.loc[matched_row, "lag_value_ratio_1"].iloc[0],
            13.0 / 11.0,
        )

    def test_timestamp_merge_uses_meter_key_when_available(self) -> None:
        df = multi_meter_frame()

        out = add_value_change_features(df, [1], value_change_regime="timestamp_merge")

        self.assertEqual(len(out), len(df))
        meter_0_hour_1 = (out["meter"] == 0) & (
            out["timestamp"] == pd.Timestamp("2016-01-01 01:00:00")
        )
        meter_1_hour_1 = (out["meter"] == 1) & (
            out["timestamp"] == pd.Timestamp("2016-01-01 01:00:00")
        )
        self.assertEqual(out.loc[meter_0_hour_1, "lag_value_diff_1"].iloc[0], 2.0)
        self.assertEqual(out.loc[meter_1_hour_1, "lag_value_diff_1"].iloc[0], 5.0)

    def test_unknown_regime_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "value_change_regime"):
            add_value_change_features(
                gapped_frame(),
                [1],
                value_change_regime="calendar_magic",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
