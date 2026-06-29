from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lead.data import _holiday_country_for_timezone, _holiday_mask_for_frame

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_m3_5_postprocessing import (  # noqa: E402
    BDG2_METER_NAMES,
    meter_auc_breakdown,
    rule_2b_end_of_year_mask,
)


class TestTimeAndPostprocessingSemantics(unittest.TestCase):
    def test_holiday_mask_uses_frame_years_and_timezone_country(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2016-07-04 12:00:00", "2017-12-25 12:00:00"]
                ),
                "timezone": ["US/Eastern", "Europe/London"],
            }
        )

        mask = _holiday_mask_for_frame(frame)

        self.assertEqual(mask.tolist(), [True, True])
        self.assertEqual(_holiday_country_for_timezone("Europe/Dublin"), "IE")
        self.assertEqual(_holiday_country_for_timezone("Europe/London"), "GB")
        self.assertEqual(_holiday_country_for_timezone("US/Pacific"), "US")

    def test_rule_2b_end_of_year_mask_uses_each_row_year_length(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2016-12-31 23:00:00",
                        "2017-12-31 23:00:00",
                        "2017-12-31 22:00:00",
                    ]
                ),
                "dayofyear": np.array([366.9584, 365.9584, 365.9167]),
            }
        )

        mask = rule_2b_end_of_year_mask(frame["dayofyear"], frame["timestamp"])

        self.assertEqual(mask.tolist(), [True, True, False])

    def test_meter_auc_breakdown_accepts_bdg2_meter_strings(self) -> None:
        y_true = pd.Series([0, 1, 0, 1])
        pred = np.array([0.1, 0.9, 0.2, 0.8])
        meter = pd.Series(["electricity", "electricity", "solar", "solar"])

        out = meter_auc_breakdown(
            y_true,
            pred,
            meter,
            meter_names=BDG2_METER_NAMES,
        )

        self.assertEqual(out["electricity"]["meter"], "electricity")
        self.assertEqual(out["solar"]["meter"], "solar")
        self.assertEqual(out["electricity"]["val_auc"], 1.0)
        self.assertEqual(out["solar"]["val_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
