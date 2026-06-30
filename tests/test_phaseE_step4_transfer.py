from __future__ import annotations

import ast
import json
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
HELPER = ROOT / "scripts" / "phaseE_transfer.py"
STEP4A = ROOT / "scripts" / "run_phaseE_step4a_bdg2_transfer.py"
STEP4B = ROOT / "scripts" / "run_phaseE_step4b_tabpfn_vs_gbdt_bdg2.py"
STEP4C = ROOT / "scripts" / "run_phaseE_step4c_pooled_powered_fallback.py"


def load_module(path: Path, name: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestPhaseEStep4Transfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.helper = load_module(HELPER, "phaseE_transfer")
        cls.step4a = load_module(STEP4A, "phaseE_step4a")
        cls.step4b = load_module(STEP4B, "phaseE_step4b")
        cls.step4c = load_module(STEP4C, "phaseE_step4c")

    def test_pilot_sites_include_fox_and_bdg2_only_rich_site(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "site_id": "Fox",
                    "buildings": 101,
                    "bdg2_only_buildings": 2,
                    "gepiii_overlap_buildings": 99,
                },
                {
                    "site_id": "Bear",
                    "buildings": 80,
                    "bdg2_only_buildings": 15,
                    "gepiii_overlap_buildings": 65,
                },
                {
                    "site_id": "Wolf",
                    "buildings": 90,
                    "bdg2_only_buildings": 3,
                    "gepiii_overlap_buildings": 87,
                },
            ]
        )
        with mock.patch.object(
            self.helper, "site_building_summary", return_value=summary
        ):
            self.assertEqual(
                self.helper.pilot_sites(Path("unused"), meter="chilledwater"),
                ["Fox", "Bear"],
            )

    def test_pilot_gate_rejects_plumbing_only_bdg2_rows(self) -> None:
        plumbing_only = [
            {
                "variant": "raw",
                "site_id": "Fox",
                "stratified": {
                    "all": {"score_summary": {"score_coverage": 1.0}},
                    "completeness_strata": {
                        "bdg2_only__sufficient_obs": {
                            "rows": 0,
                            "buildings": 0,
                            "score_summary": {"rows": 0},
                        },
                        "gepiii_overlap__sufficient_obs": {
                            "rows": 100,
                            "buildings": 10,
                            "score_summary": {"rows": 100},
                        },
                    },
                },
            }
        ]
        gate = self.step4a.pilot_gate(plumbing_only)
        self.assertEqual(gate["status"], "failed")
        self.assertEqual(gate["verdict"], "no_bdg2_only_sufficient_obs")
        self.assertEqual(gate["allowed_next_step"], "stop_and_report")

        failing = [
            {
                "variant": "raw",
                "site_id": "Fox",
                "stratified": {
                    "all": {"score_summary": {"score_coverage": 0.99}},
                    "completeness_strata": {
                        "bdg2_only__sufficient_obs": {
                            "rows": 0,
                            "buildings": 0,
                            "score_summary": {"rows": 0},
                        },
                        "gepiii_overlap__sufficient_obs": {
                            "rows": 0,
                            "buildings": 0,
                            "score_summary": {"rows": 0},
                        },
                    },
                },
            }
        ]
        gate = self.step4a.pilot_gate(failing)
        self.assertEqual(gate["status"], "failed")
        self.assertEqual(gate["allowed_next_step"], "stop_and_diagnose")

    def test_pilot_gate_allows_single_bdg2_only_building(self) -> None:
        single_building = [
            {
                "variant": "raw",
                "site_id": "Fox",
                "stratified": {
                    "all": {"score_summary": {"score_coverage": 1.0}},
                    "completeness_strata": {
                        "bdg2_only__sufficient_obs": {
                            "rows": 17_544,
                            "buildings": 1,
                            "score_summary": {"rows": 17_544, "score_median": 0.1},
                            "ood_summary": {
                                "square_feet_distribution": {"median": 1.0},
                                "meter_reading_distribution": {"median": 1.0},
                                "model_feature_missing_rate": 0.0,
                                "primary_use_unseen_rate": 0.0,
                            },
                        },
                        "gepiii_overlap__sufficient_obs": {
                            "rows": 100_000,
                            "buildings": 10,
                            "score_summary": {"rows": 100_000, "score_median": 0.1},
                            "ood_summary": {
                                "square_feet_distribution": {"median": 1.0},
                                "meter_reading_distribution": {"median": 1.0},
                                "model_feature_missing_rate": 0.0,
                                "primary_use_unseen_rate": 0.0,
                            },
                        },
                    },
                },
            }
        ]
        gate = self.step4a.pilot_gate(single_building)
        self.assertEqual(gate["status"], "passed")
        self.assertEqual(gate["verdict"], "within_context_evidence_available")
        self.assertEqual(gate["allowed_next_step"], "within_context_packet_path")
        stability = gate["multi_building_transfer_stability"]["Fox"][
            "bdg2_only__sufficient_obs"
        ]
        self.assertFalse(stability["powered"])
        self.assertEqual(stability["buildings"], 1)

    def test_pilot_gate_allows_missing_overlap_baseline(self) -> None:
        isolated = [
            {
                "variant": "raw",
                "site_id": "Swan",
                "stratified": {
                    "all": {"score_summary": {"score_coverage": 1.0}},
                    "completeness_strata": {
                        "bdg2_only__sufficient_obs": {
                            "rows": 20_000,
                            "buildings": 5,
                            "score_summary": {"rows": 20_000, "score_median": 0.1},
                        },
                        "gepiii_overlap__sufficient_obs": {
                            "rows": 0,
                            "buildings": 0,
                            "score_summary": {"rows": 0},
                        },
                    },
                },
            }
        ]
        gate = self.step4a.pilot_gate(isolated)
        self.assertEqual(gate["status"], "passed")
        self.assertEqual(gate["verdict"], "within_context_evidence_available")
        self.assertEqual(gate["allowed_next_step"], "within_context_packet_path")

    def test_multi_building_transfer_stability_flag_true_and_false(self) -> None:
        weak = self.helper.multi_building_transfer_stability(
            {"rows": 17_544, "buildings": 1, "score_summary": {"rows": 17_544}}
        )
        strong = self.helper.multi_building_transfer_stability(
            {"rows": 100_000, "buildings": 5, "score_summary": {"rows": 100_000}}
        )
        self.assertFalse(weak["powered"])
        self.assertTrue(strong["powered"])
        self.assertEqual(strong["min_buildings"], 5)
        self.assertEqual(strong["min_rows"], 17_544)

    def test_pilot_gate_detects_ood_not_missingness(self) -> None:
        def stratum(rows, buildings, median, square_feet, reading):
            return {
                "rows": rows,
                "buildings": buildings,
                "score_summary": {"rows": rows, "score_median": median},
                "ood_summary": {
                    "square_feet_distribution": {"median": square_feet},
                    "meter_reading_distribution": {"median": reading},
                    "model_feature_missing_rate": 0.0,
                    "primary_use_unseen_rate": 0.0,
                },
            }

        result = [
            {
                "variant": "raw",
                "site_id": "Pooled",
                "stratified": {
                    "all": {"score_summary": {"score_coverage": 1.0}},
                    "completeness_strata": {
                        "bdg2_only__sufficient_obs": stratum(
                            100_000, 5, 0.40, 300_000, 500
                        ),
                        "gepiii_overlap__sufficient_obs": stratum(
                            100_000, 20, 0.04, 80_000, 450
                        ),
                    },
                },
            }
        ]
        gate = self.step4a.pilot_gate(result)
        self.assertEqual(gate["status"], "passed")
        self.assertEqual(
            gate["verdict"], "within_context_evidence_available_with_ood_signal"
        )
        self.assertEqual(gate["allowed_next_step"], "within_context_packet_path")
        self.assertTrue(
            gate["sufficient_obs_comparisons"][0]["ood_evidence"]["ood_signal"]
        )

    def test_step4a_contract_does_not_emit_ground_truth_metrics(self) -> None:
        tree = ast.parse(STEP4A.read_text(encoding="utf-8"), filename=str(STEP4A))
        names = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("bdg2_ground_truth_metrics_reported", names)
        self.assertNotIn("pseudo-label ROC-AUC", names)
        self.assertNotIn("BDG2 ground-truth ROC-AUC", names)

    def test_rank_agreement_reports_spearman_and_top_decile_overlap(self) -> None:
        a = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
        b = np.array([0.1, 0.25, 0.35, 0.45, 0.8])
        agreement = self.step4b.rank_agreement(a, b)
        self.assertEqual(agreement["rows"], 5)
        self.assertAlmostEqual(agreement["spearman"], 1.0)
        self.assertAlmostEqual(agreement["top_decile_overlap"], 1.0)

    def test_empty_ood_summary_uses_json_clean_none(self) -> None:
        frame = pd.DataFrame(
            {
                "square_feet": pd.Series(dtype="float64"),
                "meter_reading": pd.Series(dtype="float64"),
                "primary_use_enc": pd.Series(dtype="int16"),
            }
        )
        summary = self.helper.ood_summary(frame, feature_cols=["meter_reading"])
        self.assertIsNone(summary["model_feature_missing_rate"])
        self.assertEqual(summary["feature_missing_rates"], {})
        cleaned = self.helper.json_clean({"x": np.float64(np.nan)})
        self.assertIsNone(cleaned["x"])
        json.dumps(cleaned, allow_nan=False)

    def test_completeness_label_uses_building_meter_missing_rate(self) -> None:
        frame = pd.DataFrame(
            {
                "building_id": ["a", "a", "b", "b"],
                "meter": [1, 1, 1, 1],
                "meter_reading": [1.0, None, None, None],
            }
        )
        labels = self.helper.completeness_label(frame)
        self.assertEqual(labels.iloc[0], "sufficient_obs")
        self.assertEqual(labels.iloc[1], "sufficient_obs")
        self.assertEqual(labels.iloc[2], "high_missing")
        self.assertEqual(labels.iloc[3], "high_missing")

    def test_stratified_delta_is_none_when_overlap_empty(self) -> None:
        frame = pd.DataFrame(
            {
                "building_id": ["a", "b"],
                "meter": [1, 1],
                "is_gepiii_overlap": [False, False],
                "primary_use_enc": [0, 0],
                "meter_reading": [1.0, 2.0],
                "log_square_feet": [1.0, 1.0],
                "square_feet": [10.0, 10.0],
            }
        )
        report = self.helper.stratified_score_report(
            featured=frame,
            scores=np.array([0.1, 0.2]),
            feature_cols=["meter_reading"],
        )
        self.assertIsNone(
            report["overlap_vs_bdg2_only"]["median_score_delta_overlap_minus_bdg2_only"]
        )
        self.assertIn("bdg2_only__sufficient_obs", report["completeness_strata"])

    def test_bounded_query_prefers_both_overlap_strata(self) -> None:
        frame = pd.DataFrame(
            {
                "building_id": [f"b{i}" for i in range(8)],
                "timestamp": pd.date_range("2017-01-01", periods=8, freq="h"),
                "is_gepiii_overlap": [
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                ],
            }
        )
        query = self.step4b.bounded_query(frame, max_rows=4, seed=42)
        self.assertLessEqual(len(query), 4)
        self.assertIn(False, set(query["is_gepiii_overlap"]))
        self.assertIn(True, set(query["is_gepiii_overlap"]))

    def test_rank_agreement_by_stratum_reports_completeness_cells(self) -> None:
        frame = pd.DataFrame(
            {
                "building_id": ["a", "a", "b", "b"],
                "meter": [1, 1, 1, 1],
                "meter_reading": [1.0, 2.0, None, None],
                "is_gepiii_overlap": [False, False, True, True],
            }
        )
        out = self.step4b.rank_agreement_by_stratum(
            frame,
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.1, 0.2, 0.4, 0.3]),
        )
        self.assertIn("bdg2_only__sufficient_obs", out)
        self.assertIn("gepiii_overlap__high_missing", out)

    def test_pooled_fallback_reports_four_cross_strata(self) -> None:
        cells = self.step4c.empty_cells()
        featured = pd.DataFrame(
            {
                "building_id": ["a", "b", "c", "d", "d"],
                "meter": [1, 1, 1, 1, 1],
                "meter_reading": [1.0, None, 3.0, None, None],
                "square_feet": [10.0, 20.0, 30.0, 40.0, 40.0],
                "log_square_feet": [1.0, 2.0, 3.0, 4.0, 4.0],
                "primary_use_enc": [0, 0, -1, 0, 0],
                "is_gepiii_overlap": [True, True, False, False, False],
            }
        )
        self.step4c.add_site_to_pool(
            cells,
            featured=featured,
            scores=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            feature_cols=["meter_reading", "log_square_feet"],
        )
        report = self.step4c.pooled_stratified_report(cells)

        self.assertEqual(
            set(report["completeness_strata"]),
            {
                "gepiii_overlap__sufficient_obs",
                "gepiii_overlap__high_missing",
                "bdg2_only__sufficient_obs",
                "bdg2_only__high_missing",
            },
        )
        self.assertEqual(report["gepiii_overlap__sufficient_obs"]["buildings"], 1)
        self.assertEqual(report["gepiii_overlap__high_missing"]["buildings"], 1)
        self.assertEqual(report["bdg2_only__sufficient_obs"]["buildings"], 1)
        self.assertEqual(report["bdg2_only__high_missing"]["buildings"], 1)

    def test_pooled_gate_underpowered_even_after_pooling(self) -> None:
        def stratum(rows, buildings, median=0.1):
            return {
                "rows": rows,
                "buildings": buildings,
                "score_summary": {"rows": rows, "score_median": median},
                "ood_summary": {
                    "square_feet_distribution": {"median": 1.0},
                    "meter_reading_distribution": {"median": 1.0},
                    "model_feature_missing_rate": 0.0,
                    "primary_use_unseen_rate": 0.0,
                },
            }

        pooled = {
            "all": {"score_summary": {"score_coverage": 1.0}},
            "completeness_strata": {
                "bdg2_only__sufficient_obs": stratum(100_000, 4),
                "gepiii_overlap__sufficient_obs": stratum(100_000, 10),
            },
        }
        gate = self.step4c.pooled_gate(pooled)
        self.assertEqual(gate["status"], "passed")
        self.assertEqual(gate["verdict"], "within_context_evidence_available")
        self.assertEqual(gate["allowed_next_step"], "within_context_packet_path")
        self.assertFalse(
            gate["multi_building_transfer_stability"]["pooled_chilledwater"][
                "bdg2_only__sufficient_obs"
            ]["powered"]
        )

    def test_pooled_gate_detects_powered_ood_without_full_permission(self) -> None:
        def stratum(rows, buildings, median, square_feet, reading):
            return {
                "rows": rows,
                "buildings": buildings,
                "score_summary": {"rows": rows, "score_median": median},
                "ood_summary": {
                    "square_feet_distribution": {"median": square_feet},
                    "meter_reading_distribution": {"median": reading},
                    "model_feature_missing_rate": 0.0,
                    "primary_use_unseen_rate": 0.0,
                },
            }

        pooled = {
            "all": {"score_summary": {"score_coverage": 1.0}},
            "completeness_strata": {
                "bdg2_only__sufficient_obs": stratum(100_000, 5, 0.50, 300_000, 500),
                "gepiii_overlap__sufficient_obs": stratum(
                    100_000, 20, 0.05, 80_000, 450
                ),
            },
        }
        gate = self.step4c.pooled_gate(pooled)
        self.assertEqual(gate["status"], "passed")
        self.assertEqual(
            gate["verdict"], "within_context_evidence_available_with_ood_signal"
        )
        self.assertEqual(gate["allowed_next_step"], "within_context_packet_path")


if __name__ == "__main__":
    unittest.main()
