from __future__ import annotations

import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_m5_building_candidate_sensitivity import (
    DEFAULT_BUILDING_SEEDS,
    build_sensitivity_audit,
    profiles_from_training_frame,
)
from scripts.m5_building_curve_protocol import (
    LadderInfeasibilityError,
    METER_IDS,
    SAMPLING_PROFILE,
    build_building_profiles,
    build_nested_building_ladder,
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
        cls.seeds = tuple(DEFAULT_BUILDING_SEEDS)
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
            for seed in cls.seeds
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.first_dir.cleanup()
        cls.second_dir.cleanup()

    def test_same_seed_artifacts_are_byte_and_digest_identical(self) -> None:
        for seed in self.seeds:
            for suffix in ("csv", "json"):
                name = f"building_ladder_seed{seed}.{suffix}"
                self.assertEqual(
                    (self.first_root / name).read_bytes(),
                    (self.second_root / name).read_bytes(),
                )
            self.assertEqual(
                self.first_summary["reproducibility_digests"][str(seed)],
                self.second_summary["reproducibility_digests"][str(seed)],
            )

    def test_seeds_are_distinct_and_every_budget_is_a_strict_prefix(self) -> None:
        for budget in (10, 20, 50, 100):
            prefixes = {
                tuple(manifest["cells"][str(budget)]["available_buildings"])
                for manifest in self.manifests.values()
            }
            self.assertEqual(len(prefixes), len(self.seeds))

        for manifest in self.manifests.values():
            previous_available: set[int] = set()
            previous_fit: set[int] = set()
            previous_es: set[int] = set()
            seed = manifest["building_seed"]
            ladder = pd.read_csv(
                self.first_root / f"building_ladder_seed{seed}.csv"
            ).rename(columns={"tree_role": "role"})
            validate_ladder(
                ladder[
                    [
                        "position",
                        "building_id",
                        "role",
                        "site_id",
                        "sampling_attempt",
                        "site_draw_rank",
                        "site_candidate_count",
                        "site_tie_priority",
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

    def test_only_even_unique_buildings_and_holdout_data_are_ignored(self) -> None:
        for manifest in self.manifests.values():
            selected = manifest["cells"]["100"]["available_buildings"]
            self.assertEqual(len(selected), len(set(selected)))
            self.assertTrue(all(building_id % 2 == 0 for building_id in selected))
            self.assertFalse(manifest["split"]["odd_data_used_for_selection"])
            self.assertFalse(manifest["split"]["odd_labels_used_for_selection"])

        odd = self.frame.copy()
        odd["building_id"] += 1
        combined = pd.concat([self.frame, odd], ignore_index=True)
        changed = combined.copy()
        odd_mask = changed["building_id"].mod(2).eq(1)
        changed.loc[odd_mask, "anomaly"] = 1 - changed.loc[odd_mask, "anomaly"]
        changed.loc[odd_mask, "site_id"] = 999
        changed.loc[odd_mask, "meter"] = 3
        changed.loc[odd_mask, "primary_use"] = "Holdout-only mutation"
        pd.testing.assert_frame_equal(
            profiles_from_training_frame(combined),
            profiles_from_training_frame(changed),
        )

    def test_anomaly_diagnostics_cannot_change_building_identity(self) -> None:
        changed = self.profiles.copy()
        changed["anomalies"] = changed["rows"] - changed["anomalies"]
        changed["anomaly_rate"] = 1.0 - changed["anomaly_rate"]
        changed["anomaly_bin"] = "deliberately_changed"
        changed["zero_anomaly"] = 1 - changed["zero_anomaly"]
        changed["anomaly_meter_count"] = 99
        for meter in METER_IDS:
            changed[f"meter_{meter}_row_share"] = (
                1.0 - changed[f"meter_{meter}_row_share"]
            )
        original, _ = build_nested_building_ladder(
            self.profiles, [10, 20, 50, 100], seed=42
        )
        mutated, _ = build_nested_building_ladder(changed, [10, 20, 50, 100], seed=42)
        np.testing.assert_array_equal(
            original["building_id"].to_numpy(),
            mutated["building_id"].to_numpy(),
        )

        minimal_columns = [
            "building_id",
            "site_id",
            *(f"meter_{meter}_present" for meter in METER_IDS),
        ]
        minimal, _ = build_nested_building_ladder(
            self.profiles[minimal_columns], [10, 20, 50, 100], seed=42
        )
        np.testing.assert_array_equal(
            original["building_id"].to_numpy(),
            minimal["building_id"].to_numpy(),
        )

    def test_site_stratification_and_random_provenance_are_recorded(self) -> None:
        for seed, manifest in self.manifests.items():
            self.assertEqual(manifest["sampling_profile"], SAMPLING_PROFILE)
            self.assertEqual(manifest["rng"]["algorithm"], "numpy.random.PCG64")
            self.assertEqual(
                manifest["sampling_method"],
                "seeded_site_stratified_random_without_replacement",
            )
            self.assertFalse(
                manifest["meter_feasibility"][
                    "single_building_swap_or_greedy_correction"
                ]
            )
            for budget in (10, 20, 50, 100):
                cell = manifest["cells"][str(budget)]
                self.assertTrue(cell["site_stratified_sampling_applied"])
                self.assertEqual(sum(cell["site_counts"].values()), budget)
                self.assertLessEqual(
                    cell["site_max_absolute_count_deviation"], 1.0 + 1e-12
                )
            ladder = pd.read_csv(self.first_root / f"building_ladder_seed{seed}.csv")
            self.assertTrue(
                {
                    "sampling_attempt",
                    "site_draw_rank",
                    "site_candidate_count",
                    "site_tie_priority",
                }.issubset(ladder.columns)
            )
            self.assertFalse(
                {
                    "selection_score",
                    "selection_rank",
                    "marginal_error_reduction",
                    "primary_balance_need_addressed",
                }
                & set(ladder.columns)
            )

    def test_all_meter_constraints_and_prefix_artifacts_pass(self) -> None:
        self.assertTrue(self.first_summary["meter_feasibility_gate"]["all_passed"])
        prefix_audit = pd.read_csv(self.first_root / "sampling_prefix_audit.csv")
        self.assertEqual(len(prefix_audit), len(self.seeds) * 4)
        self.assertTrue(prefix_audit["constraint_pass"].all())
        self.assertTrue(prefix_audit["reproducibility_digest"].str.len().eq(64).all())

        for manifest in self.manifests.values():
            previous: dict[int, int] | None = None
            for budget in (10, 20, 50, 100):
                cell = manifest["cells"][str(budget)]
                self.assertTrue(cell["meter_constraint_pass"])
                counts = {
                    meter: cell["meter_source_building_counts"][str(meter)]
                    for meter in METER_IDS
                }
                for meter in METER_IDS:
                    if previous is None:
                        self.assertGreaterEqual(counts[meter], 2)
                    else:
                        self.assertGreaterEqual(counts[meter], previous[meter] + 1)
                previous = counts

    def test_failed_whole_ladder_uses_deterministic_redraw_stream(self) -> None:
        sparse = self.profiles.copy()
        sparse["meter_3_present"] = 0
        sparse.loc[sparse.index % 8 == 0, "meter_3_present"] = 1
        with self.assertRaisesRegex(
            LadderInfeasibilityError, r"building_seed=0 after 1 attempts"
        ):
            build_nested_building_ladder(
                sparse,
                [10, 20, 50, 100],
                seed=0,
                max_sampling_attempts=1,
            )
        first, first_manifest = build_nested_building_ladder(
            sparse,
            [10, 20, 50, 100],
            seed=0,
            max_sampling_attempts=2,
        )
        second, second_manifest = build_nested_building_ladder(
            sparse,
            [10, 20, 50, 100],
            seed=0,
            max_sampling_attempts=2,
        )
        self.assertEqual(first_manifest["sampling_attempt"], 1)
        self.assertEqual(first_manifest["attempts_used"], 2)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first_manifest["cells"], second_manifest["cells"])

    def test_explicit_infeasibility_error_does_not_relax_constraints(self) -> None:
        infeasible = self.profiles.copy()
        infeasible["meter_3_present"] = 0
        infeasible.loc[infeasible.index[:4], "meter_3_present"] = 1
        with self.assertRaisesRegex(
            LadderInfeasibilityError,
            r"meter=3 K=100 available=4 required_at_least=5",
        ):
            build_nested_building_ladder(
                infeasible,
                [10, 20, 50, 100],
                seed=42,
                max_sampling_attempts=3,
            )

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
        expected_pairs = set(combinations(self.seeds, 2))
        self.assertEqual(len(overlap), len(expected_pairs) * 4)
        self.assertEqual(
            set(map(tuple, overlap[["seed_a", "seed_b"]].drop_duplicates().to_numpy())),
            expected_pairs,
        )
        self.assertTrue(overlap["jaccard_similarity"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
