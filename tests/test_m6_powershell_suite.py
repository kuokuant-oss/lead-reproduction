from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "scripts" / "run_m6_site_transfer_suite.ps1"


class TestM6PowerShellSuite(unittest.TestCase):
    def test_python_stdout_is_consumed_before_budget_function_returns(self) -> None:
        """Python progress logs must not join Get-B2CommonPositiveBudget output."""
        source = SUITE.read_text(encoding="utf-8")
        self.assertRegex(
            source,
            re.compile(r"&\s+\$Python\s+@Arguments\s*\|\s*Out-Host"),
        )


if __name__ == "__main__":
    unittest.main()
