from __future__ import annotations

import json
import unittest
from pathlib import Path


POLICY = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "reports"
    / "m5-tabpfn-repeated-inference-policy.json"
)


class TestM5RepeatedInferencePlan(unittest.TestCase):
    def test_fixed_version_and_diagnostic_boundary(self) -> None:
        policy = json.loads(POLICY.read_text(encoding="utf-8"))
        self.assertEqual(policy["scientific_tabpfn_version"], "8.0.8")
        self.assertEqual(policy["diagnostic_only_versions"], ["8.1.0"])
        self.assertIn(
            "factorial_estimation",
            policy["tabpfn_8_1_0_scope"]["excluded_from"],
        )

    def test_pilot_is_bounded_and_does_not_score_independent_query(self) -> None:
        policy = json.loads(POLICY.read_text(encoding="utf-8"))
        pilot = policy["variance_pilot"]
        self.assertEqual(pilot["status"], "designed_not_running")
        self.assertEqual(pilot["initial_replicates_per_cell"], 8)
        self.assertEqual(pilot["maximum_replicates_per_cell"], 40)
        self.assertEqual(pilot["fits_per_cell"], 1)
        self.assertEqual(pilot["cells"], ["11", "10", "01", "00"])
        self.assertIn("192_row_query_scoring", policy["forbidden_this_round"])

    def test_plan_removes_bit_stability_as_scientific_gate(self) -> None:
        plan = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "plans"
            / "m5-context-construction-paper-plan.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Bit-identical probabilities are not a", plan)
        self.assertIn("engineering diagnostics only", plan)
        self.assertIn("frozen and unscored", plan)


if __name__ == "__main__":
    unittest.main()
