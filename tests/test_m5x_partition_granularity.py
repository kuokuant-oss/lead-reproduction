from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_m5x_partition_granularity.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("m5x_partition_granularity", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestM5XPartitionGranularity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_runner()

    def test_causal_feature_count_is_77(self) -> None:
        self.assertEqual(len(self.m.past_feature_cols()), 77)

    def test_eval_idx_same_across_configs(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2016-11-01", periods=30, freq="h"),
                "window": ["test"] * 30,
            }
        )
        a = self.m.eval_indices(df, eval_rows=10, seed=7)
        b = self.m.eval_indices(df, eval_rows=10, seed=7)
        self.assertEqual(
            self.m.index_record(a)["sha256"], self.m.index_record(b)["sha256"]
        )

    def test_eval_idx_sha_assert_compares_config_records(self) -> None:
        idx = np.array([1, 3, 5, 7])
        record = self.m.index_record(idx)
        passed = self.m.eval_idx_sha_assert(
            record,
            {"C1": record, "C2": dict(record)},
        )
        self.assertTrue(passed["passed"])
        failed = self.m.eval_idx_sha_assert(
            record,
            {"C1": record, "C2": self.m.index_record(np.array([1, 3, 5, 8]))},
        )
        self.assertFalse(failed["passed"])

    def test_calib_idx_sha_assert_compares_completed_model_cells(self) -> None:
        cells = {
            "lightgbm": {"status": "completed", "calib_idx_sha256": "abc"},
            "xgboost": {"status": "completed", "calib_idx_sha256": "abc"},
            "tabpfn": {"status": "skipped"},
        }
        passed = self.m.calib_idx_sha_all_models_equal_assert(cells)
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["checked_models"], 2)
        cells["xgboost"]["calib_idx_sha256"] = "def"
        failed = self.m.calib_idx_sha_all_models_equal_assert(cells)
        self.assertFalse(failed["passed"])

    def test_no_future_leak_train_precedes_test_for_scorable_unit(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2016-01-01", "2016-02-01", "2016-11-01", "2016-12-01"]
                ),
                "anomaly": [0, 1, 0, 1],
                "unit": ["a", "a", "a", "a"],
            }
        )
        df["window"] = self.m.assign_windows(df)
        train = df.loc[df["window"].eq("train")]
        test = df.loc[df["window"].eq("test")]
        self.assertLess(train["timestamp"].max(), test["timestamp"].min())
        self.assertTrue(self.m.scorable_train_index(df, train.index.to_numpy()))
        self.assertTrue(
            self.m.no_future_leak_for_unit(
                df,
                train.index.to_numpy(),
                test.index.to_numpy(),
            )
        )

    def test_no_future_leak_assert_fails_when_unit_train_reaches_test(self) -> None:
        df = pd.DataFrame(
            {"timestamp": pd.to_datetime(["2016-01-01", "2016-11-02", "2016-11-01"])}
        )
        self.assertFalse(
            self.m.no_future_leak_for_unit(
                df,
                np.array([0, 1]),
                np.array([2]),
            )
        )

    def test_timestamp_merge_gap_keeps_causal_lag_nan(self) -> None:
        df = pd.DataFrame(
            {
                "building_id": [1, 1],
                "meter": [0, 0],
                "timestamp": pd.to_datetime(["2016-01-01 00:00", "2016-01-01 03:00"]),
                "meter_reading": [10.0, 12.0],
            }
        )
        out = self.m.add_value_change_features(
            df,
            [1],
            value_change_regime="timestamp_merge",
        )
        self.assertTrue(pd.isna(out.loc[1, "lag_value_diff_1"]))
        self.assertTrue(pd.isna(out.loc[1, "lag_value_ratio_1"]))

    def test_balanced_fit_is_half_positive_half_negative_and_bounded(self) -> None:
        y = pd.Series([0] * 80 + [1] * 20)
        ds_idx = y.index.to_numpy()
        out = self.m.balanced_subsample_indices(ds_idx, y, max_rows=20, seed=42)
        labels = y.loc[out]
        self.assertLessEqual(len(out), 20)
        self.assertEqual(int((labels == 1).sum()), 10)
        self.assertEqual(int((labels == 0).sum()), 10)

    def test_tabpfn_skip_guard(self) -> None:
        args = mock.Mock(skip_tabpfn=True, model_path=None)
        env = {"tabpfn_installed": False, "torch_installed": False, "device": "cpu"}
        runner = self.m.Runner(args, env)
        matrices = {
            "x_fit": np.zeros((4, 2)),
            "y_fit": pd.Series([0, 1, 0, 1]),
            "x_calib": np.zeros((4, 2)),
            "y_calib": pd.Series([0, 1, 0, 1]),
            "x_eval": np.zeros((4, 2)),
            "y_eval": pd.Series([0, 1, 0, 1]),
        }
        with mock.patch.object(self.m, "tree_models", return_value={}):
            out = self.m.fit_score_all_models(runner, matrices, seed=42)
        self.assertEqual(out["tabpfn"]["status"], "skipped")

    def test_imports_only_frozen_lead_api(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        imported_from_lead: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "lead":
                imported_from_lead.update(alias.name for alias in node.names)
        import lead

        self.assertTrue(imported_from_lead)
        self.assertTrue(imported_from_lead <= set(lead.__all__))


if __name__ == "__main__":
    unittest.main()
