from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_m5_phaseD_foundation_vs_gbdt.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("m5_phaseD_compare", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPhaseDComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_runner()

    def test_imports_only_frozen_lead_api(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        imported_from_lead: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "lead":
                imported_from_lead.update(alias.name for alias in node.names)
        # Every name pulled from `lead` must be in the frozen public API.
        import lead

        self.assertTrue(imported_from_lead)
        self.assertTrue(imported_from_lead <= set(lead.__all__))

    def test_balanced_subsample_is_balanced_and_bounded(self) -> None:
        y = pd.Series([0] * 80 + [1] * 20)
        ds_idx = y.index.to_numpy()
        out = self.m.balanced_subsample_indices(ds_idx, y, max_rows=20, seed=42)
        self.assertLessEqual(len(out), 20)
        labels = y.loc[out]
        # 20 positives available, per_class=10, so 10 pos + 10 neg.
        self.assertEqual(int((labels == 1).sum()), 10)
        self.assertEqual(int((labels == 0).sum()), 10)

    def test_random_val_indices_deterministic(self) -> None:
        index = np.arange(1000)
        a = self.m.random_val_indices(index, 100, seed=7)
        b = self.m.random_val_indices(index, 100, seed=7)
        self.assertEqual(len(a), 100)
        np.testing.assert_array_equal(a, b)

    def test_cell_metrics_include_pr_auc(self) -> None:
        y = pd.Series([0, 1, 0, 1])
        pred = np.array([0.1, 0.9, 0.2, 0.7])
        metrics = self.m.cell_metrics(y, pred)
        for key in ("val_auc", "pr_auc", "precision_05", "recall_05", "f1_05"):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["val_auc"], 1.0)

    def test_aggregate_reports_mean_std_and_skips_failures(self) -> None:
        cells = [
            {
                "status": "completed",
                "val_auc": 0.90,
                "pr_auc": 0.5,
                "precision_05": 0.5,
                "recall_05": 0.5,
                "f1_05": 0.5,
                "fit_predict_seconds": 1.0,
            },
            {
                "status": "completed",
                "val_auc": 0.94,
                "pr_auc": 0.6,
                "precision_05": 0.5,
                "recall_05": 0.5,
                "f1_05": 0.5,
                "fit_predict_seconds": 3.0,
            },
            {"status": "failed", "error": "boom"},
        ]
        agg = self.m.aggregate(cells)
        self.assertEqual(agg["n_runs"], 3)
        self.assertEqual(agg["n_completed"], 2)
        self.assertAlmostEqual(agg["mean"]["val_auc"], 0.92)
        self.assertGreater(agg["std"]["val_auc"], 0.0)

    def test_tabpfn_limit_fit_flags_documented_budget(self) -> None:
        within = self.m.tabpfn_limit_fit(10_000, 137)
        self.assertTrue(within["fits_documented_tabpfn3_limit"])
        beyond = self.m.tabpfn_limit_fit(4_285_104, 137)
        self.assertFalse(beyond["fits_documented_tabpfn3_limit"])

    def test_tabpfn_metrics_derive_from_tabpfn_probabilities(self) -> None:
        class FakeTabPFN:
            def fit(self, x_train, y_train) -> None:
                self.fit_shape = np.asarray(x_train).shape

            def predict_proba(self, x_val):
                n = np.asarray(x_val).shape[0]
                col1 = np.linspace(0.1, 0.9, n)
                return np.column_stack([1 - col1, col1])

        x_train = np.zeros((6, 3))
        y_train = pd.Series([0, 1, 0, 1, 0, 1])
        x_val = np.zeros((4, 3))
        y_val = pd.Series([0, 1, 0, 1])
        with mock.patch.object(self.m, "tabpfn_classifier", return_value=FakeTabPFN()):
            metrics = self.m.fit_tabpfn(
                x_train, y_train, x_val, y_val, device="cpu", model_path=None
            )
        for key in (
            "model_init_seconds",
            "fit_seconds",
            "predict_proba_seconds",
            "fit_predict_seconds",
            "val_auc",
            "pr_auc",
        ):
            self.assertIn(key, metrics)

    def test_tabpfn_cell_skips_when_unavailable(self) -> None:
        args = mock.Mock(skip_tabpfn=True, model_path=None, tabpfn_batch_size=256)
        env = {"tabpfn_installed": False, "torch_installed": False, "device": "cpu"}
        runner = self.m.Runner(args, env)
        cell = runner.tabpfn_cell(
            np.zeros((2, 2)),
            pd.Series([0, 1]),
            np.zeros((2, 2)),
            pd.Series([0, 1]),
            seed=42,
            fit_rows=2,
        )
        self.assertEqual(cell["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
