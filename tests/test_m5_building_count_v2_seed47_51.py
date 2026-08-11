from __future__ import annotations

import unittest

from scripts.prepare_m5_building_count_v2_seed47_51 import (
    BUNDLED_CANONICAL_HOLDOUT,
    EXPECTED_BUDGETS,
    EXPECTED_SEEDS,
    validate_audit_bundle,
    validate_canonical_holdout,
)
from scripts.run_m5_building_count_v2 import budget_major_seed_pairs
from scripts.run_m5_building_count_v2_seed47_51 import (
    command_for_args,
    parse_args,
)


class TestM5BuildingCountV2Seed47To51(unittest.TestCase):
    def test_portable_audit_bundle_is_complete(self) -> None:
        summary = validate_audit_bundle()
        self.assertEqual(summary["building_seeds"], list(EXPECTED_SEEDS))
        self.assertEqual(summary["budgets"], list(EXPECTED_BUDGETS))
        self.assertEqual(summary["profile_source"], "candidate_building_profiles.csv")
        validate_canonical_holdout(BUNDLED_CANONICAL_HOLDOUT)

    def test_formal_order_is_budget_major(self) -> None:
        summary = validate_audit_bundle()
        expected = [
            (seed, budget)
            for budget in EXPECTED_BUDGETS
            for seed in EXPECTED_SEEDS
        ]
        self.assertEqual(budget_major_seed_pairs(summary), expected)

    def test_launcher_defaults_to_non_launching_plan(self) -> None:
        args = parse_args([])
        self.assertEqual(args.mode, "plan")
        command = command_for_args(args)
        self.assertIn("--pair-order", command)
        self.assertEqual(command[command.index("--pair-order") + 1], "budget-major")
        self.assertNotIn("--publish-results", command)

    def test_validation_is_bounded_and_isolated(self) -> None:
        args = parse_args(["--mode", "validation"])
        command = command_for_args(args)
        self.assertIn("--validation-context-rows", command)
        self.assertIn("--validation-holdout-rows", command)
        self.assertIn("NON_SCIENTIFIC_VALIDATION_seed47_51", " ".join(command))


if __name__ == "__main__":
    unittest.main()
