from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_m5_phaseC_tabpfn_spike.py"


class TestM5TabPFNSpike(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("m5_tabpfn_spike", SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {SCRIPT}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.runner = module

    def test_runner_imports_frozen_lead_api_helpers(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "lead":
                imports.update(alias.name for alias in node.names)

        self.assertGreaterEqual(
            imports,
            {
                "BASELINE_FEATURE_COLS",
                "SHIFTS",
                "add_value_change_features",
                "assert_no_building_overlap",
                "classification_metrics",
                "downsample_indices",
                "load_m3_frame",
                "write_json_with_provenance",
            },
        )

    def test_runner_uses_m3_split_sample_and_row_offset_path(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('building_id"] % 5 == 4', source)
        self.assertIn("downsample_indices(y_train_full)", source)
        self.assertIn('VALUE_CHANGE_REGIME = "row_offset"', source)
        self.assertIn("list(SHIFTS)", source)

    def test_runner_supports_local_checkpoint_without_token(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--model-path", source)
        self.assertIn("TABPFN_MODEL_CACHE_DIR", source)
        self.assertIn("local_checkpoint_available", source)
        self.assertIn("not_run_missing_weights_and_token", source)
        self.assertIn(
            "model_path=model_path if local_checkpoint_available else None", source
        )

    def test_tabpfn_metrics_are_derived_from_tabpfn_probabilities(self) -> None:
        class FakeTabPFN:
            def fit(self, x_train, y_train) -> None:
                self.fit_shape = x_train.shape

            def predict_proba(self, x_val):
                self.predict_shape = x_val.shape
                return np.asarray(
                    [
                        [0.9, 0.1],
                        [0.2, 0.8],
                        [0.4, 0.6],
                        [0.7, 0.3],
                    ]
                )

        x_train = np.zeros((4, 2))
        y_train = pd.Series([0, 1, 1, 0])
        x_val = np.zeros((4, 2))
        y_val = pd.Series([0, 1, 0, 1])

        with mock.patch.object(
            self.runner, "tabpfn_classifier", return_value=FakeTabPFN()
        ):
            metrics = self.runner.fit_tabpfn(
                x_train,
                y_train,
                x_val,
                y_val,
                device="cpu",
                model_path=None,
            )

        expected = self.runner.classification_metrics(
            y_val, np.asarray([0.1, 0.8, 0.6, 0.3])
        )
        for key, value in expected.items():
            self.assertEqual(metrics[key], value)
        self.assertTrue(metrics["cold_start"])
        self.assertIn("model_init_seconds", metrics)
        self.assertIn("fit_seconds", metrics)
        self.assertIn("predict_proba_seconds", metrics)


if __name__ == "__main__":
    unittest.main()
