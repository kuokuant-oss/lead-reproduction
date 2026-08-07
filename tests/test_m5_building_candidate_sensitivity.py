from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_m5_building_candidate_sensitivity import (
    build_sensitivity_audit,
    profiles_from_training_frame,
)
from scripts.m5_building_curve_protocol import (
    build_building_profiles,
    stable_priority,
    validate_ladder,
)
from scripts.run_m5_building_curve_tabpfn_cell import parse_args as tabpfn_args
from scripts.run_m5_building_curve_tree_cell import parse_args as tree_args
from tests.test_m5_building_curve_protocol import synthetic_frame


class TestM5BuildingCandidateSensitivity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = synthetic_frame()
        cls.profiles = build_building_profiles(cls.frame)
        cls.first_dir = tempfile.TemporaryDirectory()
        cls.second_dir = tempfile.TemporaryDirectory()
        cls.first_root = Path(cls.first_dir.name)
        cls.second_root = Path(cls.second_dir.name)
        cls.first_summary = build_sensitivity_audit(cls.profiles, cls.first_root)
        cls.second_summary = build_sensitivity_audit(cls.profiles, cls.second_root)
        cls.manifests = {
            seed: json.loads(
                (cls.first_root / f"building_ladder_seed{seed}.json").read_text(
                    encoding="utf-8"
                )
            )
            for seed in (42, 43, 44)
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.first_dir.cleanup()
        cls.second_dir.cleanup()

    def test_same_seed_artifacts_are_byte_and_digest_identical(self) -> None:
        for seed in (42, 43, 44):
            for suffix in ("csv", "json"):
                name = f"building_ladder_seed{seed}.{suffix}"
                self.assertEqual(
                    (self.first_root / name).read_bytes(),
                    (self.second_root / name).read_bytes(),
                )
            self.assertEqual(
                self.first_summary["ladders"][str(seed)]["csv_sha256"],
                self.second_summary["ladders"][str(seed)]["csv_sha256"],
            )

    def test_three_seeds_are_distinct_and_every_budget_is_a_strict_prefix(self) -> None:
        for budget in (10, 20, 50, 100):
            prefixes = {
                tuple(manifest["cells"][str(budget)]["available_buildings"])
                for manifest in self.manifests.values()
            }
            self.assertEqual(len(prefixes), 3)
        for manifest in self.manifests.values():
            previous_available: set[int] = set()
            previous_fit: set[int] = set()
            previous_es: set[int] = set()
            ladder = pd.read_csv(
                self.first_root / f"building_ladder_seed{manifest['building_seed']}.csv"
            )
            validate_ladder(
                ladder.rename(columns={"tree_role": "role"})[
                    [
                        "position",
                        "building_id",
                        "role",
                        "site_id",
                        "stable_priority",
                        "selection_score",
                        "overall_error_before",
                        "overall_error_after",
                        "marginal_error_reduction",
                        "primary_balance_need_addressed",
                    ]
                ],
                manifest,
            )
            for budget in (10, 20, 50, 100):
                cell = manifest["cells"][str(budget)]
                available = set(cell["available_buildings"])
                fit = set(cell["tree_fit_buildings"])
                early_stop = set(cell["tree_early_stop_buildings"])
                self.assertTrue(
                    previous_available < available if previous_available else True
                )
                self.assertTrue(previous_fit < fit if previous_fit else True)
                self.assertTrue(previous_es < early_stop if previous_es else True)
                previous_available, previous_fit, previous_es = (
                    available,
                    fit,
                    early_stop,
                )

    def test_selected_buildings_are_unique_even_and_holdout_labels_are_ignored(
        self,
    ) -> None:
        for manifest in self.manifests.values():
            selected = manifest["cells"]["100"]["available_buildings"]
            self.assertEqual(len(selected), len(set(selected)))
            self.assertTrue(all(building_id % 2 == 0 for building_id in selected))
            self.assertFalse(manifest["split"]["odd_labels_used_for_selection"])

        odd = self.frame.copy()
        odd["building_id"] += 1
        combined = pd.concat([self.frame, odd], ignore_index=True)
        changed = combined.copy()
        odd_mask = changed["building_id"].mod(2).eq(1)
        changed.loc[odd_mask, "anomaly"] = 1 - changed.loc[odd_mask, "anomaly"]
        first = profiles_from_training_frame(combined)
        second = profiles_from_training_frame(changed)
        pd.testing.assert_frame_equal(first, second)

    def test_quality_composition_and_candidate_limits_pass(self) -> None:
        self.assertTrue(self.first_summary["quality_gate"]["all_passed"])
        self.assertTrue(self.first_summary["meaningful_difference_gate"]["passed"])
        composition = pd.read_csv(self.first_root / "composition_audit.csv")
        self.assertEqual(len(composition), 12)
        self.assertTrue(composition["quality_gate_pass"].all())
        self.assertTrue(
            (
                composition["prefix_discrepancy"]
                <= composition["canonical_best_greedy_discrepancy"] * 1.50 + 1e-12
            ).all()
        )
        self.assertTrue((composition["absolute_degradation"] <= 0.003 + 1e-12).all())
        for seed in (42, 43, 44):
            ladder = pd.read_csv(self.first_root / f"building_ladder_seed{seed}.csv")
            self.assertTrue(
                (ladder["selection_score_ratio_to_best"] <= 1.02 + 1e-12).all()
            )
            self.assertTrue(ladder["selection_rank"].le(4).all())
            required = {
                "site_id",
                "primary_use",
                "tree_role",
                "rows",
                "anomalies",
                "anomaly_rate",
                "meter_presence",
                "size_bin",
                "anomaly_bin",
                "acceptable_candidate_rank",
                "seed_priority",
            }
            self.assertTrue(required.issubset(ladder.columns))

    def test_fixed_row_seed_keeps_priority_policy_independent_of_building_seed(
        self,
    ) -> None:
        raw_rows = np.arange(100, 132, dtype="int64")
        expected = stable_priority(raw_rows, seed=42)
        for manifest in self.manifests.values():
            self.assertEqual(manifest["row_seed"], 42)
            self.assertEqual(manifest["row_selection_seed"], 42)
            self.assertIsNone(manifest["role_seed"])
            np.testing.assert_array_equal(
                stable_priority(raw_rows, seed=manifest["row_seed"]), expected
            )
            self.assertIn("building_row_quotas", manifest)

    def test_runner_default_identity_contains_explicit_building_seed(self) -> None:
        manifest42 = self.first_root / "building_ladder_seed42.json"
        manifest43 = self.first_root / "building_ladder_seed43.json"
        common42 = [
            "--building-manifest",
            str(manifest42),
            "--building-budget",
            "10",
        ]
        common43 = [
            "--building-manifest",
            str(manifest43),
            "--building-budget",
            "10",
        ]
        tree42 = tree_args(common42)
        tree43 = tree_args(common43)
        tabpfn42 = tabpfn_args(common42)
        tabpfn43 = tabpfn_args(common43)
        self.assertIn("building_seed42", str(tree42.out_root))
        self.assertIn("building_seed43", str(tree43.out_root))
        self.assertIn("building_seed42", str(tabpfn42.out_root))
        self.assertIn("building_seed43", str(tabpfn43.out_root))
        self.assertNotEqual(tree42.out_root, tree43.out_root)
        self.assertNotEqual(tabpfn42.out_root, tabpfn43.out_root)
        self.assertEqual(tree42.model_seed, 42)
        self.assertEqual(tabpfn42.model_seed, 42)

    def test_overlap_artifact_has_all_pairwise_rows(self) -> None:
        overlap = pd.read_csv(self.first_root / "building_overlap.csv")
        self.assertEqual(len(overlap), 12)
        self.assertEqual(
            set(map(tuple, overlap[["seed_a", "seed_b"]].drop_duplicates().to_numpy())),
            {(42, 43), (42, 44), (43, 44)},
        )
        self.assertTrue(overlap["jaccard_similarity"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
