from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
M4_PLAN = ROOT / "docs" / "plans" / "m4-plan.md"
ADR_DIR = ROOT / "docs" / "adr"
M4_GOVERNANCE_ISSUE = "24"


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


def latest_done_m4_slice() -> str:
    statuses = m4_plan_statuses()
    done_slices = [
        slice_name for slice_name, status in statuses.items() if status == "Done"
    ]
    if not done_slices:
        raise AssertionError("M4 plan has no completed slices")
    return sorted(done_slices, key=lambda value: int(value.split(".")[1]))[-1]


def readme_milestone_status(milestone: str) -> str:
    readme = read_text(README)
    pattern = re.compile(
        rf"^\| \*\*{re.escape(milestone)}\*\* \| [^|]+ \| (?P<status>[^|]+) \|",
        re.MULTILINE,
    )
    match = pattern.search(readme)
    if not match:
        raise AssertionError(f"README milestone table has no {milestone} row")
    return match.group("status").strip()


def readme_m4_section_claim() -> str:
    readme = read_text(README)
    match = re.search(r"^M4\.0-M4\.(?P<slice>\d) complete", readme, re.MULTILINE)
    if not match:
        raise AssertionError("README M4 section has no completed-through claim")
    return f"M4.{match.group('slice')}"


def m4_tracker_rows() -> dict[str, dict[str, str]]:
    text = read_text(M4_PLAN)
    rows: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        if not line.startswith("| M4"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] == "Slice":
            continue
        rows[cells[0]] = {"issue": cells[1], "status": cells[2]}
    return rows


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


def actual_adr_count() -> int:
    return len(list(ADR_DIR.glob("*.md")))


class TestReadmeFreshness(unittest.TestCase):
    def test_readme_m4_status_matches_plan(self) -> None:
        latest_done = latest_done_m4_slice()

        self.assertEqual(
            readme_milestone_status("M4"),
            f"M4.0-{latest_done} complete",
            "README M4 milestone table status is stale",
        )
        self.assertEqual(
            readme_m4_section_claim(),
            latest_done,
            "README M4 section completed-through claim is stale",
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

    def test_readme_adr_count_matches_adr_directory(self) -> None:
        readme = read_text(README)
        match = re.search(r"目前共有 (?P<count>\d+) 份 ADR", readme)
        if not match:
            raise AssertionError("README does not state the current ADR count")

        self.assertEqual(
            int(match.group("count")),
            actual_adr_count(),
            "README ADR count is stale",
        )

    def test_m4_tracker_closed_governance_issue_is_done(self) -> None:
        row = m4_tracker_rows().get("M4 governance hardening")
        self.assertIsNotNone(row, "M4 tracker is missing governance row")
        assert row is not None
        self.assertIn(f"issues/{M4_GOVERNANCE_ISSUE}", row["issue"])
        self.assertNotIn("TBD", row["issue"])
        self.assertEqual(
            row["status"],
            "Done",
            f"M4 governance issue #{M4_GOVERNANCE_ISSUE} is closed; tracker must be Done",
        )


if __name__ == "__main__":
    unittest.main()
