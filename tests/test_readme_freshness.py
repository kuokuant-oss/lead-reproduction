from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
M4_PLAN = ROOT / "docs" / "plans" / "m4-plan.md"
ADR_DIR = ROOT / "docs" / "adr"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def m4_plan_statuses() -> dict[str, str]:
    text = read_text(M4_PLAN)
    statuses: dict[str, str] = {}
    pattern = re.compile(
        r"^### M4\.(?P<slice>\d):.*?^\*\*Status\*\*: (?P<status>\w+)",
        re.MULTILINE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        statuses[f"M4.{match.group('slice')}"] = match.group("status")
    return statuses


def adr_status(adr_number: str) -> str:
    matches = sorted(ADR_DIR.glob(f"{adr_number}-*.md"))
    if not matches:
        raise AssertionError(f"ADR {adr_number} not found")

    text = read_text(matches[0])
    match = re.search(r"^## Status\s+^(?P<status>[^\n]+)", text, re.MULTILINE)
    if not match:
        raise AssertionError(f"ADR {adr_number} has no status block")
    return match.group("status").strip()


def readme_lines_with(token: str) -> list[str]:
    return [line for line in read_text(README).splitlines() if token in line]


class TestReadmeFreshness(unittest.TestCase):
    def test_readme_m4_status_matches_plan(self) -> None:
        statuses = m4_plan_statuses()
        done_slices = [
            slice_name for slice_name, status in statuses.items() if status == "Done"
        ]
        self.assertTrue(done_slices, "M4 plan has no completed slices")

        latest_done = sorted(done_slices, key=lambda value: int(value.split(".")[1]))[
            -1
        ]
        readme = read_text(README)

        self.assertNotIn(
            "M4.2 與 M4.3 尚未執行",
            readme,
            "README says M4.2/M4.3 are not executed, but the M4 plan marks them Done",
        )
        self.assertIn(
            f"M4.0-{latest_done} complete",
            readme,
            f"README should report completion through {latest_done}",
        )

    def test_readme_adr_status_matches_adr_files(self) -> None:
        for adr_number in ("0010", "0011"):
            actual_status = adr_status(adr_number)
            lines = readme_lines_with(f"ADR {adr_number}")
            self.assertTrue(lines, f"README does not mention ADR {adr_number}")

            for line in lines:
                self.assertNotIn(
                    "Proposed",
                    line,
                    f"README claims ADR {adr_number} is Proposed, actual status is {actual_status}",
                )

            self.assertTrue(
                any(actual_status.split()[0] in line for line in lines),
                f"README should mention ADR {adr_number} status {actual_status}",
            )


if __name__ == "__main__":
    unittest.main()
