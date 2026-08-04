from __future__ import annotations

import unittest

import numpy as np

from scripts.report_m5_building_curve import aggregate_cell


class TestM5BuildingCurveReporting(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "sampling_profile": "representative",
            "building_budget": 10,
            "features": 17,
            "score_names": ["tabpfn", "ensemble"],
        }
        self.payload = {
            "validation_raw_index": np.arange(12, dtype="int64"),
            "anomaly": np.tile([0, 1], 6).astype("int8"),
            "building_id": np.repeat([1, 3, 5], 4).astype("int16"),
            "site_id": np.repeat([0, 1, 2], 4).astype("int8"),
            "meter": np.tile(np.arange(4), 3).astype("int8"),
            "tabpfn": np.tile([0.1, 0.9], 6).astype("float32"),
            "ensemble": np.tile([0.2, 0.8], 6).astype("float32"),
        }

    def test_outputs_overall_meter_site_metrics_and_both_curve_types(self) -> None:
        metrics, curves = aggregate_cell(self.metadata, self.payload)
        self.assertEqual(
            {row["grouping"] for row in metrics}, {"overall", "meter", "site"}
        )
        overall = [row for row in metrics if row["grouping"] == "overall"]
        self.assertEqual(len(overall), 2)
        self.assertTrue(all(row["roc_auc"] == 1.0 for row in overall))
        self.assertEqual({row["curve"] for row in curves}, {"roc", "precision_recall"})

    def test_duplicate_identity_and_nonfinite_scores_fail_closed(self) -> None:
        broken = dict(self.payload)
        broken["validation_raw_index"] = np.zeros(12, dtype="int64")
        with self.assertRaisesRegex(AssertionError, "not unique"):
            aggregate_cell(self.metadata, broken)
        broken = dict(self.payload)
        broken["tabpfn"] = broken["tabpfn"].copy()
        broken["tabpfn"][0] = np.nan
        with self.assertRaisesRegex(AssertionError, "invalid prediction"):
            aggregate_cell(self.metadata, broken)

    def test_missing_meter_key_is_rejected(self) -> None:
        broken = dict(self.payload)
        del broken["meter"]
        with self.assertRaisesRegex(ValueError, "missing"):
            aggregate_cell(self.metadata, broken)


if __name__ == "__main__":
    unittest.main()
