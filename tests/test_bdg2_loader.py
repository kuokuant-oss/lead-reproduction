from __future__ import annotations

import unittest

from lead.bdg2 import load_bdg2_frame
from lead.split import assert_no_building_overlap, leave_site_out_mask, split_mask


class TestBDG2Loader(unittest.TestCase):
    def test_unlabeled_smoke_frame_schema(self) -> None:
        frame = load_bdg2_frame(
            "tests/fixtures/bdg2_fox_smoke.csv",
            verbose=False,
        )
        self.assertEqual(
            list(frame.columns),
            [
                "building_id",
                "site_id",
                "timestamp",
                "meter",
                "meter_reading",
                "hour",
                "weekday",
                "month",
                "dayofyear",
            ],
        )
        self.assertNotIn("anomaly", frame.columns)
        self.assertEqual(set(frame["site_id"]), {"Fox"})

    def test_optional_labels_align_by_row_identity(self) -> None:
        frame = load_bdg2_frame(
            "tests/fixtures/bdg2_fox_smoke.csv",
            label_path="tests/fixtures/bdg2_fox_smoke_labels.csv",
            verbose=False,
        )
        self.assertIn("anomaly", frame.columns)
        self.assertEqual(int(frame["anomaly"].sum()), 2)

    def test_building_and_site_split_helpers_have_no_overlap(self) -> None:
        frame = load_bdg2_frame(
            "tests/fixtures/bdg2_fox_smoke.csv",
            verbose=False,
        )
        val_mask = split_mask(frame, "80_20_mod5")
        train_buildings = set(frame.loc[~val_mask, "building_id"].unique())
        val_buildings = set(frame.loc[val_mask, "building_id"].unique())
        self.assertEqual(
            assert_no_building_overlap(
                train_buildings, val_buildings, split_name="80_20_mod5"
            ),
            set(),
        )

        site_mask = leave_site_out_mask(frame, ["Fox"])
        self.assertTrue(site_mask.all())

    def test_full_download_requires_future_explicit_implementation(self) -> None:
        with self.assertRaises(NotImplementedError):
            load_bdg2_frame(
                "tests/fixtures/bdg2_fox_smoke.csv",
                allow_download=True,
                verbose=False,
            )


if __name__ == "__main__":
    unittest.main()
