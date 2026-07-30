from __future__ import annotations

import unittest
from pathlib import Path

from scripts.check_long_running_timeout_policy import (
    Finding,
    ROOT,
    classify,
    markdown,
    scan,
    violations,
)


class TestLongRunningTimeoutPolicy(unittest.TestCase):
    def test_research_process_timeout_is_a_violation(self) -> None:
        category, _, _ = classify(
            Path("scripts/new_research.py"),
            "subprocess.run(command, timeout=10)",
            blocked=False,
        )
        self.assertEqual(category, "A")
        self.assertEqual(
            len(
                violations(
                    [
                        Finding(
                            "scripts/new_research.py",
                            1,
                            "timeout=10",
                            category,
                            "x",
                            "x",
                        )
                    ]
                )
            ),
            1,
        )

    def test_network_cleanup_and_test_timeouts_are_not_violations(self) -> None:
        self.assertEqual(
            classify(
                Path("scripts/supervise_m5_tabpfn_recovery.py"),
                "--timeout",
                blocked=False,
            )[0],
            "B",
        )
        self.assertEqual(
            classify(
                Path("scripts/m6_tabpfn_context_protocol.py"),
                "thread.join(timeout=2)",
                blocked=False,
            )[0],
            "C",
        )
        self.assertEqual(
            classify(Path("tests/test_example.py"), "timeout=1", blocked=False)[0], "D"
        )

    def test_blocked_legacy_path_is_reported_without_override(self) -> None:
        category, _, action = classify(
            Path("scripts/run_m5_tabpfn_single_context_scaling.py"),
            "worker.wait(timeout=10)",
            blocked=True,
        )
        self.assertEqual(category, "E")
        self.assertIn("Launch blocked", action)

    def test_report_contains_each_finding_and_certification(self) -> None:
        report = markdown(
            [Finding("scripts/x.py", 5, "timeout=1", "E", "legacy", "migrate")]
        )
        self.assertIn("scripts/x.py:5", report)
        self.assertIn("No category-A", report)

    def test_repository_has_no_active_research_timeout_and_blocks_legacy_paths(
        self,
    ) -> None:
        findings = scan(ROOT)
        self.assertEqual(violations(findings), [])
        blocked = {finding.path for finding in findings if finding.category == "E"}
        self.assertTrue(
            {
                "scripts/run_m5_tabpfn_137_batches.ps1",
                "scripts/run_m5_tabpfn_canonical_full_test.py",
                "scripts/run_m5_tabpfn_single_context_scaling.py",
            }.issubset(blocked)
        )


if __name__ == "__main__":
    unittest.main()
