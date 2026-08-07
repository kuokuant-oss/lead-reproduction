"""Audit research launch paths for prohibited wall-clock termination.

Network request bounds, thread cleanup waits, and test-fixture timeouts are
reported but are not research-process timeout violations. Legacy research
launchers with a documented unconditional launch block are reported as blocked
until their checkpoint/resume migration is complete.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".py", ".ps1", ".sh"}
PATTERN = re.compile(r"(?i)\btimeout\b|timeout=|--timeout|Start-Job")
BLOCK_MARKER = "LONG_RUNNING_RESEARCH_LAUNCH_BLOCKED"


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    text: str
    category: str
    disposition: str
    action: str


def classify(path: Path, text: str, *, blocked: bool) -> tuple[str, str, str]:
    name = path.as_posix()
    if name.startswith("tests/"):
        return (
            "D",
            "test_fixture_or_test_timeout",
            "Not a research launch; retain only as a bounded test fixture.",
        )
    lowered = text.lower()
    if name in {
        "scripts/run_m5_tabpfn_portable_shard.py",
        "scripts/run_m6_tabpfn_context_suite.ps1",
    }:
        return (
            "C",
            "explicit_no_timeout_execution",
            "Reviewed no-timeout path; the match is explanatory text, not a kill control.",
        )
    if (
        "no wall-time timeout" in lowered
        or "no wall-clock timeout" in lowered
        or "no timeout" in lowered
    ):
        return (
            "C",
            "explicit_no_timeout_declaration",
            "Documentation/observability only; it does not terminate research work.",
        )
    if name in {
        "scripts/run_m5_building_curve_overnight.py",
        "scripts/run_m5_building_candidate_sensitivity_overnight.py",
    }:
        return (
            "B",
            "git_push_request_timeout",
            "Network publication bound; scientific child processes have no wall-clock termination.",
        )
    if "deploy_m5_tabpfn" in name or "supervise_m5_tabpfn_recovery" in name:
        return (
            "B",
            "external_service_request_timeout",
            "External setup/request bound; not a scientific-process kill.",
        )
    if ".join(timeout" in text or "_thread.join(timeout" in text:
        return "C", "cleanup_wait", "Cleanup wait; it does not terminate research work."
    if blocked:
        return (
            "E",
            "legacy_long_running_path",
            "Launch blocked until checkpoint/resume migration removes termination logic.",
        )
    return (
        "A",
        "research_process_timeout_or_auto_kill",
        "Remove timeout/auto-kill or install an unconditional launch block.",
    )


def scan(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for directory in (root / "scripts", root / "tests"):
        if not directory.exists():
            continue
        for path in sorted(
            candidate
            for candidate in directory.rglob("*")
            if candidate.suffix.lower() in SOURCE_SUFFIXES
        ):
            if path.name in {
                "check_long_running_timeout_policy.py",
                "probe_foreground_persistent_session.py",
                "run_m5_e0_stage1_validation.ps1",
            }:
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            blocked = BLOCK_MARKER in source
            for line_number, line in enumerate(source.splitlines(), start=1):
                if PATTERN.search(line):
                    category, disposition, action = classify(
                        path.relative_to(root), line, blocked=blocked
                    )
                    findings.append(
                        Finding(
                            path.relative_to(root).as_posix(),
                            line_number,
                            line.strip(),
                            category,
                            disposition,
                            action,
                        )
                    )
    return findings


def violations(findings: list[Finding]) -> list[Finding]:
    return [finding for finding in findings if finding.category == "A"]


def markdown(findings: list[Finding]) -> str:
    lines = [
        "# Repository long-running timeout audit",
        "",
        "This static audit distinguishes research-process wall-clock termination from external-service bounds, cleanup waits, and test fixtures.",
        "",
        "| Category | Path:line | Context | Disposition | Required action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in findings:
        context = item.text.replace("|", "\\|").replace("`", "'")
        lines.append(
            f"| {item.category} | `{item.path}:{item.line}` | `{context}` | {item.disposition} | {item.action} |"
        )
    counts = {
        category: sum(item.category == category for item in findings)
        for category in "ABCDE"
    }
    lines.extend(
        [
            "",
            "## Certification",
            "",
            "Findings: "
            + ", ".join(f"{key}={value}" for key, value in counts.items())
            + ".",
            "No category-A research-process timeout/auto-kill finding may remain. Category-E paths are not launchable and require migration before use. Network/request timeouts, cleanup waits, and tests are explicitly excluded from the research-process prohibition.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args(argv)
    findings = scan(args.root)
    report = markdown(findings)
    if args.write_report:
        args.write_report.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    bad = violations(findings)
    if bad:
        print(
            f"timeout policy failed: {len(bad)} research-process violation(s)",
            file=sys.stderr,
        )
        return 1
    print(
        "timeout policy passed: no active research-process wall-clock timeout",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
