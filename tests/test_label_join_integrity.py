from __future__ import annotations

import unittest

import pandas as pd

from lead.data import _assign_positional_labels


def train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "building_id": [10, 10, 11],
            "meter": [0, 1, 0],
            "timestamp": pd.to_datetime(
                [
                    "2016-01-01 00:00:00",
                    "2016-01-01 00:00:00",
                    "2016-01-01 01:00:00",
                ]
            ),
            "meter_reading": [1.0, 2.0, 3.0],
        }
    )


def labels(values: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame({"is_bad_meter_reading": values or [1, 0, 1]})


class TestLabelJoinIntegrity(unittest.TestCase):
    def test_assigns_labels_positionally_when_integrity_checks_pass(self) -> None:
        train = train_frame()

        out = _assign_positional_labels(train, labels())

        self.assertEqual(out["anomaly"].tolist(), [1, 0, 1])
        self.assertEqual(str(out["anomaly"].dtype), "int8")

    def test_bad_meter_readings_must_have_exact_expected_column(self) -> None:
        bad = labels()
        bad["building_id"] = [10, 10, 11]

        with self.assertRaisesRegex(ValueError, "exactly one column"):
            _assign_positional_labels(train_frame(), bad)

    def test_bad_meter_readings_must_match_train_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "align 1:1"):
            _assign_positional_labels(train_frame(), labels([1, 0]))

    def test_train_row_identity_must_be_unique(self) -> None:
        train = train_frame()
        train.loc[1, ["building_id", "meter", "timestamp"]] = train.loc[
            0, ["building_id", "meter", "timestamp"]
        ]

        with self.assertRaisesRegex(ValueError, "row identity is undefined"):
            _assign_positional_labels(train, labels())

    def test_row_order_mismatch_fails_loudly(self) -> None:
        train = train_frame().iloc[[1, 0, 2]]

        with self.assertRaisesRegex(ValueError, "raw train.csv row order"):
            _assign_positional_labels(train, labels())


if __name__ == "__main__":
    unittest.main()
