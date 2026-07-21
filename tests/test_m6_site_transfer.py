from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PROTOCOL = SCRIPTS / "m6_site_transfer_protocol.py"
RUNNER = SCRIPTS / "run_m6_site_transfer.py"
COMPARATOR = SCRIPTS / "compare_m6_site_oracle.py"
AGGREGATOR = SCRIPTS / "aggregate_m6_site_transfer.py"


def load_script(name: str, path: Path):
    scripts_dir = str(SCRIPTS)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestM6SiteTransferProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_script("m6_site_transfer_protocol", PROTOCOL)

    @staticmethod
    def split_frame() -> pd.DataFrame:
        rows = []
        for site in range(16):
            for building in (site * 10, site * 10 + 1):
                for meter in (0, 1):
                    rows.append(
                        {
                            "site_id": site,
                            "building_id": building,
                            "meter": meter,
                            "anomaly": int((building + meter) % 3 == 0),
                        }
                    )
        return pd.DataFrame(rows)

    def test_a3_folds_cover_each_site_once(self) -> None:
        flattened = [
            site
            for fold in sorted(self.m.A3_SITE_FOLDS)
            for site in self.m.A3_SITE_FOLDS[fold]
        ]
        self.assertEqual(sorted(flattened), list(range(16)))
        self.assertEqual(len(flattened), len(set(flattened)))
        frame = self.split_frame()
        for fold in range(4):
            train, test, manifest = self.m.location_split_masks(
                frame,
                "a3_group4",
                fold=fold,
            )
            self.assertFalse(np.any(train & test))
            self.assertEqual(manifest["site_overlap"], [])
            self.assertEqual(manifest["test"]["sites"], 4)

    def test_a2_is_exact_reverse_of_a1(self) -> None:
        frame = self.split_frame()
        a1_train, a1_test, _ = self.m.location_split_masks(
            frame,
            "a1_even_to_odd",
        )
        a2_train, a2_test, _ = self.m.location_split_masks(
            frame,
            "a2_odd_to_even",
        )
        np.testing.assert_array_equal(a2_train, a1_test)
        np.testing.assert_array_equal(a2_test, a1_train)

    def test_a5_oracle_is_building_disjoint_inside_one_site(self) -> None:
        frame = self.split_frame()
        train, test, manifest = self.m.in_site_oracle_masks(frame, 11)
        self.assertEqual(set(frame.loc[train | test, "site_id"]), {11})
        self.assertEqual(manifest["site_overlap"], [11])
        self.assertEqual(manifest["building_overlap"], [])
        self.assertEqual(manifest["target_label_access"], "oracle_train_only")

    def test_b1_meter_prefix_is_nested_and_covers_all_source_sites(self) -> None:
        frame = self.split_frame()
        source, _, _ = self.m.location_split_masks(frame, "a1_even_to_odd")
        small = self.m.meter_budget_manifest(frame, source, budget=8, seed=42)
        large = self.m.meter_budget_manifest(frame, source, budget=16, seed=42)
        small_keys = {
            tuple(row[column] for column in self.m.METER_KEY_COLUMNS)
            for row in small["selected_meters"]
        }
        large_keys = {
            tuple(row[column] for column in self.m.METER_KEY_COLUMNS)
            for row in large["selected_meters"]
        }
        self.assertTrue(small["all_source_sites_covered"])
        self.assertEqual(len(small["site_allocation"]), 8)
        self.assertTrue(small_keys < large_keys)
        selected_rows = self.m.meter_manifest_mask(frame, small)
        self.assertEqual(set(frame.loc[selected_rows, "site_id"]), set(range(0, 16, 2)))

    def test_b1_rejects_budget_that_drops_source_sites(self) -> None:
        frame = self.split_frame()
        source, _, _ = self.m.location_split_masks(frame, "a1_even_to_odd")
        with self.assertRaisesRegex(ValueError, "cannot cover all 8 source sites"):
            self.m.meter_budget_manifest(frame, source, budget=7, seed=42)

    def test_b2_matches_unique_anomalies_and_keeps_m3_shape(self) -> None:
        y = pd.Series([0] * 40 + [1] * 12)
        fit_index, manifest = self.m.matched_anomaly_fit_indices(
            y,
            positive_budget=5,
            selection_seed=123,
        )
        self.assertEqual(len(fit_index), 20)
        self.assertEqual(int(y.loc[fit_index].sum()), 10)
        self.assertEqual(len(np.unique(fit_index[5:10])), 5)
        np.testing.assert_array_equal(fit_index[5:10], fit_index[15:20])
        self.assertEqual(manifest["unique_anomaly_rows"], 5)
        self.assertEqual(manifest["fit_rows"], 20)
        self.assertEqual(manifest["effective_fit_anomaly_rate"], 0.5)

    def test_grouped_support_records_plot_denominators(self) -> None:
        frame = self.split_frame()
        source, _, _ = self.m.location_split_masks(frame, "a1_even_to_odd")
        support = self.m.grouped_support(frame, source)
        self.assertEqual(set(support), {str(site) for site in range(0, 16, 2)})
        self.assertEqual(support["0"]["buildings"], 2)
        self.assertEqual(support["0"]["meter_series"], 4)
        self.assertIn("anomaly_rate", support["0"])

    def test_source_calibration_is_building_disjoint_inside_source_sites(self) -> None:
        frame = pd.DataFrame(
            {
                "site_id": [0, 0, 0, 0, 2, 2, 2, 2],
                "building_id": [4, 4, 5, 5, 9, 9, 10, 10],
                "meter": [0] * 8,
                "anomaly": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        source = np.ones(len(frame), dtype=bool)
        fit, calibration, manifest = self.m.source_building_calibration_masks(
            frame,
            source,
        )
        self.assertEqual(set(frame.loc[calibration, "building_id"]), {4, 9})
        self.assertEqual(set(frame.loc[fit, "building_id"]), {5, 10})
        self.assertEqual(manifest["building_overlap"], [])


class TestM6RunnerIsolation(unittest.TestCase):
    def test_runner_only_imports_public_frozen_lead_api(self) -> None:
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
        imported_from_lead: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "lead":
                imported_from_lead.update(alias.name for alias in node.names)
        import lead

        self.assertTrue(imported_from_lead)
        self.assertTrue(imported_from_lead <= set(lead.__all__))

    def test_runner_declares_additive_cells_without_a1_overwrite(self) -> None:
        runner = load_script("run_m6_site_transfer", RUNNER)
        self.assertEqual(
            runner.SUPPORTED_EXPERIMENTS,
            ("a2", "a3", "a4", "a5", "b1", "b2"),
        )
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"src_lead_modified": False', source)
        self.assertNotIn('m3_full_site_transfer.json"', source)

    def test_plot_helpers_keep_macro_dispersion_and_exact_histogram_counts(
        self,
    ) -> None:
        runner = load_script("run_m6_site_transfer_plot_helpers", RUNNER)
        slices = {
            "1": {
                "models": {
                    "ensemble": {"roc_auc": 0.9, "pr_auc": 0.2},
                }
            },
            "3": {
                "models": {
                    "ensemble": {"roc_auc": 1.0, "pr_auc": 0.8},
                }
            },
        }
        macro = runner.macro_site_summary(slices)
        self.assertAlmostEqual(macro["ensemble"]["pr_auc"]["mean"], 0.5)
        self.assertEqual(macro["ensemble"]["pr_auc"]["n_scorable"], 2)
        hist = runner.score_histograms(
            np.array([0, 1, 0, 1]),
            np.array([1, 1, 3, 3]),
            {"ensemble": np.array([0.1, 0.8, 0.2, 0.9])},
            bins=10,
        )
        self.assertEqual(sum(hist["overall"]["ensemble"]["normal"]), 2)
        self.assertEqual(sum(hist["overall"]["ensemble"]["anomaly"]), 2)
        self.assertEqual(
            sum(hist["by_site_id"]["1"]["ensemble"]["anomaly"]),
            1,
        )

    def test_fixed_recall_threshold_uses_calibration_positive_scores(self) -> None:
        runner = load_script("run_m6_site_transfer_threshold", RUNNER)
        y = np.array([1, 1, 1, 1, 1, 0, 0])
        prediction = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.65, 0.1])
        threshold = runner.fixed_recall_threshold(
            y,
            prediction,
            target_recall=0.8,
        )
        self.assertEqual(threshold, 0.6)
        metrics = runner.evaluation_at_threshold(y, prediction, threshold)
        self.assertEqual(metrics["recall"], 0.8)
        self.assertEqual(metrics["confusion_matrix"]["fp"], 1)


class TestM6OracleComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_script("compare_m6_site_oracle", COMPARATOR)

    @staticmethod
    def artifacts() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        cross = {
            "site_id": np.array([1, 1, 1, 3]),
            "building_id": np.array([11, 11, 13, 31]),
            "meter": np.array([0, 0, 0, 0]),
            "timestamp_ns": np.array([100, 200, 100, 100]),
            "anomaly": np.array([0, 1, 0, 1]),
            "ensemble": np.array([0.4, 0.6, 0.7, 0.8]),
        }
        oracle = {
            "site_id": np.array([1, 1]),
            "building_id": np.array([11, 11]),
            "meter": np.array([0, 0]),
            "timestamp_ns": np.array([100, 200]),
            "anomaly": np.array([0, 1]),
            "ensemble": np.array([0.1, 0.9]),
        }
        return cross, oracle

    def test_oracle_comparison_uses_identical_ordered_rows(self) -> None:
        cross, oracle = self.artifacts()
        result = self.m.paired_oracle_comparison(cross, oracle)
        self.assertEqual(result["site_id"], 1)
        self.assertEqual(result["n_rows"], 2)
        self.assertEqual(
            result["identity_basis"],
            "ordered_site_building_meter_timestamp_label",
        )
        self.assertEqual(result["models"]["ensemble"]["cross_site"]["pr_auc"], 1.0)
        self.assertEqual(
            result["models"]["ensemble"]["in_site_oracle"]["pr_auc"],
            1.0,
        )

    def test_oracle_comparison_rejects_row_identity_drift(self) -> None:
        cross, oracle = self.artifacts()
        oracle["timestamp_ns"][1] = 999
        with self.assertRaisesRegex(AssertionError, "same ordered rows"):
            self.m.paired_oracle_comparison(cross, oracle)


class TestM6PlotDataAggregation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_script("aggregate_m6_site_transfer", AGGREGATOR)

    @staticmethod
    def cell_payload(cell: str) -> dict:
        return {
            "status": "completed",
            "cell": cell,
            "split": {"name": "a1_even_to_odd", "fold": 0},
            "selection": (
                {"budget": 50, "seed": 42, "site_allocation": {"0": 7}}
                if cell == "b1"
                else None
            ),
            "fit": {
                "unique_anomaly_rows": 10,
                "fit_rows": 40,
                "selection_seed": 42,
                "model_fit_seconds": {"ensemble": 1.0},
            },
            "metrics": {
                "ensemble": {
                    "roc_auc": 0.9,
                    "pr_auc": 0.6,
                    "threshold_0_5": {
                        "precision": 0.5,
                        "recall": 0.8,
                        "f1": 0.62,
                        "confusion_matrix": {"tn": 8, "fp": 2, "fn": 1, "tp": 4},
                    },
                }
            },
            "curves": {
                "ensemble": {
                    "roc": {"x": [0, 1], "y": [0, 1]},
                    "precision_recall": {"x": [0, 1], "y": [1, 0.5]},
                }
            },
            "slices": {
                "by_site_id": {
                    "1": {
                        "models": {
                            "ensemble": {
                                "n_rows": 15,
                                "n_anomalies": 5,
                                "anomaly_rate": 1 / 3,
                                "roc_auc": 0.9,
                                "pr_auc": 0.6,
                                "threshold_0_5": None,
                            }
                        }
                    }
                }
            },
            "macro_site_metrics": {"ensemble": {"pr_auc": {"mean": 0.6}}},
            "score_histograms": {"bin_edges": [0, 1]},
            "operating_points": {
                "source_calibrated_recall_0_90": {
                    "ensemble": {
                        "threshold": 0.3,
                        "test": {"recall": 0.9},
                    }
                }
            },
            "artifacts": {"predictions": "data/processed/cell.npz"},
            "elapsed_seconds": 2.0,
            "timing_breakdown": {"train_feature_seconds": 0.5},
            "prediction_seconds": {"ensemble_probability_combine": 0.01},
            "matrix_profile": {"fit_shape": [40, 137]},
        }

    def test_aggregate_exposes_every_plot_family_without_copying_predictions(
        self,
    ) -> None:
        result = self.m.aggregate_payloads(
            [("cell.json", self.cell_payload("b1"))],
            [],
        )
        self.assertEqual(len(result["model_metrics"]), 1)
        self.assertEqual(len(result["curves"]), 2)
        self.assertEqual(len(result["site_metrics"]), 1)
        self.assertEqual(len(result["fixed_recall_0_90"]), 1)
        self.assertEqual(result["learning_curves"][0]["budget"], 50)
        self.assertEqual(result["cells"][0]["predictions"], "data/processed/cell.npz")
        families = result["plot_data_contract"]["families"]
        self.assertIn("per_site_forest", families)
        self.assertIn("runtime_and_matrix_cost", families)


if __name__ == "__main__":
    unittest.main()
