from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lead.bdg2 import load_bdg2_frame


def write_fixture(root: Path) -> None:
    pd.DataFrame(
        {
            "building_id": ["Panther_lodging_Dean", "Panther_office_Hannah"],
            "site_id": ["Panther", "Panther"],
            "building_id_kaggle": [0.0, ""],
            "site_id_kaggle": [0.0, ""],
            "primaryspaceusage": ["Lodging/residential", "Office"],
            "sub_primaryspaceusage": ["Residence Hall", "Office"],
            "sqm": [508.8, 100.0],
            "sqft": [5477.0, 1076.0],
            "lat": [28.5, 28.5],
            "lng": [-81.3, -81.3],
            "timezone": ["US/Eastern", "US/Eastern"],
            "electricity": ["Yes", "Yes"],
            "hotwater": ["", ""],
            "chilledwater": ["", ""],
            "steam": ["", ""],
            "water": ["Yes", ""],
            "irrigation": ["", ""],
            "solar": ["", ""],
            "gas": ["", ""],
            "industry": ["", ""],
            "subindustry": ["", ""],
            "heatingtype": ["", ""],
            "yearbuilt": [1989.0, 2001.0],
            "date_opened": ["", ""],
            "numberoffloors": [3.0, 4.0],
            "occupants": ["", ""],
            "energystarscore": ["", ""],
            "eui": [271.0, 62.0],
            "site_eui": ["", ""],
            "source_eui": ["", ""],
            "leed_level": ["None", "None"],
            "rating": ["", ""],
        }
    ).to_csv(root / "metadata.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": ["2016-01-01 00:00:00", "2016-01-01 01:00:00"],
            "Panther_lodging_Dean": [1.0, 2.0],
            "Panther_office_Hannah": [10.0, 20.0],
        }
    ).to_csv(root / "electricity_cleaned.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": ["2016-01-01 00:00:00", "2016-01-01 01:00:00"],
            "site_id": ["Panther", "Panther"],
            "airTemperature": [19.4, 21.1],
            "cloudCoverage": [None, 6.0],
            "dewTemperature": [19.4, 21.1],
            "precipDepth1HR": [0.0, -1.0],
            "precipDepth6HR": [None, None],
            "seaLvlPressure": [None, 1019.4],
            "windDirection": [0.0, 0.0],
            "windSpeed": [0.0, 0.0],
        }
    ).to_csv(root / "weather.csv", index=False)


class TestBdg2Loader(unittest.TestCase):
    def test_loads_real_schema_wide_meter_file_to_long_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root)

            frame = load_bdg2_frame(
                bdg2_dir=root,
                meter_types=["electricity"],
                building_ids=["Panther_lodging_Dean", "Panther_office_Hannah"],
            )

        self.assertEqual(len(frame), 4)
        self.assertNotIn("anomaly", frame.columns)
        self.assertEqual(
            set(
                [
                    "building_id",
                    "site_id",
                    "timestamp",
                    "meter",
                    "meter_reading",
                    "building_id_kaggle",
                    "site_id_kaggle",
                    "is_gepiii_overlap",
                    "primary_use",
                    "square_feet",
                    "year_built",
                    "floor_count",
                    "timezone",
                    "air_temperature",
                ]
            )
            - set(frame.columns),
            set(),
        )
        self.assertEqual(set(frame["meter"]), {"electricity"})
        self.assertEqual(
            set(frame["building_id"]),
            {"Panther_lodging_Dean", "Panther_office_Hannah"},
        )
        self.assertEqual(frame["is_gepiii_overlap"].dtype, bool)
        overlap_by_building = frame.groupby("building_id")["is_gepiii_overlap"].first()
        self.assertTrue(bool(overlap_by_building["Panther_lodging_Dean"]))
        self.assertFalse(bool(overlap_by_building["Panther_office_Hannah"]))
        self.assertEqual(
            int(frame.duplicated(["building_id", "meter", "timestamp"]).sum()),
            0,
        )

    def test_real_metadata_has_partial_gepiii_overlap(self) -> None:
        bdg2_dir = Path("data/raw/bdg2")
        if not (bdg2_dir / "metadata.csv").exists():
            self.skipTest("real BDG2 metadata.csv is not available")

        frame = load_bdg2_frame(
            bdg2_dir=bdg2_dir,
            meter_types=["electricity"],
            building_ids=["Panther_parking_Lorriane", "Mouse_health_Buddy"],
            nrows=1,
            include_weather=False,
        )

        overlap_count = int(
            pd.read_csv(bdg2_dir / "metadata.csv")["building_id_kaggle"].notna().sum()
        )
        self.assertIn("is_gepiii_overlap", frame.columns)
        self.assertEqual(frame["is_gepiii_overlap"].dtype, bool)
        self.assertGreater(overlap_count, 0)
        self.assertLess(overlap_count, 1636)

    def test_rejects_meter_building_not_present_in_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root)
            pd.DataFrame(
                {
                    "timestamp": ["2016-01-01 00:00:00"],
                    "Imaginary_office_Bad": [1.0],
                }
            ).to_csv(root / "electricity_cleaned.csv", index=False)

            with self.assertRaisesRegex(ValueError, "absent from metadata"):
                load_bdg2_frame(
                    bdg2_dir=root,
                    meter_types=["electricity"],
                )

    def test_rejects_duplicate_weather_site_timestamp_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root)
            pd.DataFrame(
                {
                    "timestamp": ["2016-01-01 00:00:00", "2016-01-01 00:00:00"],
                    "site_id": ["Panther", "Panther"],
                }
            ).to_csv(root / "weather.csv", index=False)

            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_bdg2_frame(
                    bdg2_dir=root,
                    meter_types=["electricity"],
                )


if __name__ == "__main__":
    unittest.main()
