from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_m3_full_site_transfer.py"


def load_runner():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("m3_full_site_transfer", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestM3FullSiteTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_runner()

    def test_imports_only_frozen_lead_api(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        imported_from_lead: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "lead":
                imported_from_lead.update(alias.name for alias in node.names)

        import lead

        self.assertTrue(imported_from_lead)
        self.assertTrue(imported_from_lead <= set(lead.__all__))

    def test_site_split_matches_m5_mod2_contract(self) -> None:
        frame = pd.DataFrame(
            {
                "site_id": [0, 0, 1, 1, 2, 3],
                "building_id": [10, 10, 11, 12, 13, 14],
                "anomaly": [0, 1, 0, 1, 0, 1],
            }
        )
        mask = self.m.site_transfer_mask(frame)
        np.testing.assert_array_equal(mask, [False, False, True, True, False, True])
        self.assertEqual(self.m.SITE_SPLIT_NAME, "site_id_mod2_50_50")
        self.assertEqual(
            self.m.SITE_SPLIT_RULE,
            "validation sites are site_id % 2 == 1",
        )

    def test_contract_keeps_frozen_m3_semantics(self) -> None:
        self.assertEqual(self.m.VALUE_CHANGE_REGIME, "timestamp_merge")
        self.assertEqual(self.m.RANDOM_STATE, 42)
        self.assertIn("confusion_matrix", self.m.PLOT_DATA_FAMILIES)
        self.assertIn("roc_curves", self.m.PLOT_DATA_FAMILIES)
        self.assertIn("precision_recall_curves", self.m.PLOT_DATA_FAMILIES)
        self.assertIn("model_permutation_importance", self.m.PLOT_DATA_FAMILIES)
        self.assertIn("site_and_meter_slices", self.m.PLOT_DATA_FAMILIES)

    def test_relative_cli_output_paths_are_resolved_and_serializable(self) -> None:
        relative = Path("data/processed/m3_full_site_transfer_predictions.npz")
        resolved = self.m.resolve_output_path(relative)

        self.assertTrue(resolved.is_absolute())
        self.assertEqual(
            self.m.artifact_path(resolved),
            "data/processed/m3_full_site_transfer_predictions.npz",
        )

    def test_grouped_metrics_records_all_models(self) -> None:
        groups = np.array([1, 1, 3, 3])
        y_true = np.array([0, 1, 0, 1])
        predictions = {
            "lightgbm": np.array([0.1, 0.9, 0.2, 0.8]),
            "ensemble": np.array([0.2, 0.8, 0.3, 0.7]),
        }
        result = self.m.grouped_metrics(groups, y_true, predictions)
        self.assertEqual(set(result), {"1", "3"})
        self.assertEqual(set(result["1"]["models"]), set(predictions))
        self.assertEqual(result["3"]["models"]["ensemble"]["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
