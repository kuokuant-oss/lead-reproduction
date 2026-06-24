from __future__ import annotations

import json
import unittest
from pathlib import Path

from lead import (
    add_value_change_features,
    downsample_indices,
    load_m3_frame,
    split_mask,
)


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden_metrics.json"
REFACTOR_CHECK = ROOT / "data" / "processed" / "m4_1_refactor_check.json"


class TestRefactorRegression(unittest.TestCase):
    def test_public_api_imports_from_src_lead(self) -> None:
        self.assertEqual(load_m3_frame.__module__, "lead.data")
        self.assertEqual(add_value_change_features.__module__, "lead.features")
        self.assertEqual(downsample_indices.__module__, "lead.sample")
        self.assertEqual(split_mask.__module__, "lead.split")

    def test_m4_1_refactor_check_matches_golden_auc(self) -> None:
        if not REFACTOR_CHECK.exists():
            self.skipTest(
                "Run M4.1 reruns to create data/processed/m4_1_refactor_check.json"
            )

        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        check = json.loads(REFACTOR_CHECK.read_text(encoding="utf-8"))
        noise_floor = golden["noise_floor_auc"]

        comparisons = {
            "m3_2_lightgbm_80_20_offline_auc": check["checks"]["m3_2"]["actual_auc"],
            "m3_4_ensemble_80_20_offline_auc": check["checks"]["m3_4"]["actual_auc"],
        }
        for metric_name, actual_auc in comparisons.items():
            expected_auc = golden["metrics"][metric_name]["value"]
            delta = actual_auc - expected_auc
            self.assertLessEqual(
                abs(delta),
                noise_floor,
                f"{metric_name} delta {delta} exceeds {noise_floor}",
            )


if __name__ == "__main__":
    unittest.main()
