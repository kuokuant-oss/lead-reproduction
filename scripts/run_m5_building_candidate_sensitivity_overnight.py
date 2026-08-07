"""Persistent bounded-retry supervisor for the M5 building sensitivity sweep.

The scientific cell runners retain ownership of checkpointing and resume. This
supervisor only schedules cells, persists attempt counts, skips exhausted units,
finalizes complete sweeps, and optionally publishes the generated report section.
No scientific subprocess is given a wall-clock timeout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from lead import PROC, ROOT
from run_m5_building_candidate_sensitivity_models import _complete, _units

DEFAULT_AUDIT_ROOT = (
    PROC / "m5_building_curve" / "sensitivity" / "building_candidate_pilot"
)
DEFAULT_REPORT = ROOT / "docs" / "reports" / "m5-building-count-experiment.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--mode", choices=("plan", "formal"), default="plan")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--retry-delay", type=int, default=120)
    parser.add_argument("--unit-retries", type=int, default=2)
    parser.add_argument("--finalize-retries", type=int, default=2)
    parser.add_argument("--push-retries", type=int, default=5)
    parser.add_argument("--git-push-timeout", type=int, default=120)
    parser.add_argument("--gpu-wait-checks", type=int, default=30)
    parser.add_argument("--publish-results", action="store_true")
    args = parser.parse_args(argv)
    if args.retry_delay < 0 or args.unit_retries < 0 or args.finalize_retries < 0:
        raise ValueError("retry delay/counts must be non-negative")
    if args.push_retries < 1 or args.git_push_timeout < 1 or args.gpu_wait_checks < 1:
        raise ValueError("push bounds and GPU wait checks must be positive")
    return args


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _event(supervisor_root: Path, event: str, **values: Any) -> None:
    supervisor_root.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.time(), "event": event, **values}
    with (supervisor_root / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(payload, sort_keys=True), flush=True)


def _unit_name(unit: dict[str, Any]) -> str:
    identity = unit["identity"]
    return (
        f"building_seed{identity['building_seed']}_k{identity['K']}_"
        f"f{identity['features']}_{identity['model']}"
    )


def _attempt_path(supervisor_root: Path, stage: str) -> Path:
    return supervisor_root / "attempts" / f"{stage}.json"


def _failure_path(supervisor_root: Path, stage: str) -> Path:
    return supervisor_root / "failed_stages" / f"{stage}.json"


def _attempt_count(supervisor_root: Path, stage: str) -> int:
    path = _attempt_path(supervisor_root, stage)
    if not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return max(0, int(payload.get("attempt", 0)))
    except (OSError, TypeError, ValueError):
        return 0


def _record_attempt(supervisor_root: Path, stage: str, attempt: int) -> None:
    _atomic_json(
        _attempt_path(supervisor_root, stage),
        {"stage": stage, "attempt": attempt, "timestamp": time.time()},
    )


def _mark_stage_failed(
    supervisor_root: Path,
    stage: str,
    *,
    attempts: int,
    reason: str,
    returncode: int | None = None,
) -> None:
    _atomic_json(
        _failure_path(supervisor_root, stage),
        {
            "status": "stage_failed",
            "requires_review": True,
            "stage": stage,
            "attempts": attempts,
            "reason": reason,
            "returncode": returncode,
            "timestamp": time.time(),
        },
    )


def _active_gpu_processes() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return ["nvidia-smi unavailable"]
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _wait_for_idle_gpu(
    supervisor_root: Path,
    stage: str,
    *,
    checks: int,
    delay: int,
) -> bool:
    for check in range(1, checks + 1):
        active = _active_gpu_processes()
        if not active:
            return True
        _event(
            supervisor_root,
            "gpu_busy",
            stage=stage,
            check=check,
            max_checks=checks,
            active_gpu_processes=active,
        )
        if check < checks:
            time.sleep(delay)
    return False


def _status_payload(
    units: list[dict[str, Any]],
    *,
    status: str,
    failed_stages: list[str],
    **values: Any,
) -> dict[str, Any]:
    completed = sum(_complete(unit) for unit in units)
    return {
        "mode": "formal",
        "status": status,
        "units": len(units),
        "completed": completed,
        "pending": len(units) - completed,
        "failed_stages": failed_stages,
        "unit_identities": [unit["identity"] for unit in units],
        "timestamp": time.time(),
        **values,
    }


def _write_status(
    audit_root: Path,
    supervisor_root: Path,
    payload: dict[str, Any],
) -> None:
    _atomic_json(audit_root / "model_sweep_status.json", payload)
    _atomic_json(supervisor_root / "status.json", payload)


def _run_units(
    args: argparse.Namespace,
    units: list[dict[str, Any]],
    supervisor_root: Path,
) -> list[str]:
    max_attempts = 1 + args.unit_retries
    failed: list[str] = []
    for index, unit in enumerate(units, start=1):
        stage = _unit_name(unit)
        failed_marker = _failure_path(supervisor_root, stage)
        if _complete(unit):
            _event(supervisor_root, "unit_reused", stage=stage)
            continue
        if failed_marker.is_file():
            failed.append(stage)
            _event(
                supervisor_root,
                "unit_skipped_failed_marker",
                stage=stage,
                marker=str(failed_marker),
            )
            continue

        attempts = _attempt_count(supervisor_root, stage)
        last_returncode: int | None = None
        while not _complete(unit) and attempts < max_attempts:
            attempts += 1
            _record_attempt(supervisor_root, stage, attempts)
            _write_status(
                args.audit_root,
                supervisor_root,
                _status_payload(
                    units,
                    status="running",
                    failed_stages=failed,
                    current_unit=index,
                    current_identity=unit["identity"],
                    attempt=attempts,
                    max_attempts=max_attempts,
                ),
            )
            if unit["identity"]["model"] == "tabpfn" and not _wait_for_idle_gpu(
                supervisor_root,
                stage,
                checks=args.gpu_wait_checks,
                delay=args.retry_delay,
            ):
                last_returncode = 75
                _event(
                    supervisor_root,
                    "unit_gpu_wait_exhausted",
                    stage=stage,
                    attempt=attempts,
                )
            else:
                _event(
                    supervisor_root,
                    "unit_start",
                    stage=stage,
                    attempt=attempts,
                    command=unit["command"],
                )
                result = subprocess.run(unit["command"], cwd=ROOT)
                last_returncode = result.returncode
                _event(
                    supervisor_root,
                    "unit_end",
                    stage=stage,
                    attempt=attempts,
                    returncode=result.returncode,
                    complete=_complete(unit),
                )
            if _complete(unit):
                break
            if attempts < max_attempts:
                _event(
                    supervisor_root,
                    "unit_retry",
                    stage=stage,
                    attempt=attempts,
                    delay_seconds=args.retry_delay,
                )
                time.sleep(args.retry_delay)

        if _complete(unit):
            _event(supervisor_root, "unit_complete", stage=stage, attempts=attempts)
            continue
        _mark_stage_failed(
            supervisor_root,
            stage,
            attempts=attempts,
            reason="unit_retry_limit_exhausted",
            returncode=last_returncode,
        )
        failed.append(stage)
        _event(
            supervisor_root,
            "unit_failed",
            stage=stage,
            attempts=attempts,
            returncode=last_returncode,
        )
    return failed


def _bounded_stage(
    args: argparse.Namespace,
    supervisor_root: Path,
    *,
    stage: str,
    action: Callable[[], int],
    valid: Callable[[], bool],
    retries: int | None = None,
) -> bool:
    if valid():
        _event(supervisor_root, "finalize_stage_reused", stage=stage)
        return True
    failed_marker = _failure_path(supervisor_root, stage)
    if failed_marker.is_file():
        _event(supervisor_root, "finalize_stage_skipped", stage=stage)
        return False
    attempts = _attempt_count(supervisor_root, stage)
    max_attempts = 1 + (args.finalize_retries if retries is None else retries)
    last_returncode: int | None = None
    while attempts < max_attempts and not valid():
        attempts += 1
        _record_attempt(supervisor_root, stage, attempts)
        last_returncode = action()
        if last_returncode == 0 and valid():
            _event(
                supervisor_root,
                "finalize_stage_complete",
                stage=stage,
                attempts=attempts,
            )
            return True
        _event(
            supervisor_root,
            "finalize_stage_retry",
            stage=stage,
            attempt=attempts,
            returncode=last_returncode,
        )
        if attempts < max_attempts:
            time.sleep(args.retry_delay)
    _mark_stage_failed(
        supervisor_root,
        stage,
        attempts=attempts,
        reason="finalize_retry_limit_exhausted",
        returncode=last_returncode,
    )
    return False


def _aggregate_paths(audit_root: Path, seeds: list[int]) -> tuple[Path, Path]:
    tag = "building_seed_sweep_" + "-".join(str(seed) for seed in seeds)
    root = audit_root / "model_results" / tag
    return root, root / "summary.json"


def _aggregate_action(
    audit_root: Path,
    units: list[dict[str, Any]],
    seeds: list[int],
) -> Callable[[], int]:
    aggregate_root, _ = _aggregate_paths(audit_root, seeds)
    command = [
        sys.executable,
        "scripts/report_m5_building_curve.py",
        *[str(Path(unit["output"]) / "cell.json") for unit in units],
        "--out-root",
        str(aggregate_root),
    ]

    def run() -> int:
        return subprocess.run(command, cwd=ROOT).returncode

    return run


def _report_action(args: argparse.Namespace) -> Callable[[], int]:
    command = [
        sys.executable,
        "scripts/update_m5_building_candidate_sensitivity_report.py",
        "--audit-root",
        str(args.audit_root),
        "--report",
        str(args.report),
    ]

    def run() -> int:
        return subprocess.run(command, cwd=ROOT).returncode

    return run


def _report_complete(report: Path) -> bool:
    if not report.is_file():
        return False
    text = report.read_text(encoding="utf-8")
    return (
        "<!-- BEGIN M5 BUILDING CANDIDATE SENSITIVITY PILOT -->" in text
        and "<!-- END M5 BUILDING CANDIDATE SENSITIVITY PILOT -->" in text
    )


def _publication_action(args: argparse.Namespace) -> Callable[[], int]:
    relative_report = str(args.report.relative_to(ROOT))

    def run() -> int:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        dirty = {line[3:] for line in status.splitlines() if line.strip()}
        if dirty - {relative_report}:
            _event(
                args.audit_root / "overnight",
                "publication_dirty_gate_failed",
                dirty=sorted(dirty),
            )
            return 3
        subprocess.run(["git", "add", relative_report], cwd=ROOT, check=True)
        changed = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT
        ).returncode
        if changed:
            result = subprocess.run(
                ["git", "commit", "-m", "Record M5 building sensitivity results"],
                cwd=ROOT,
            )
            if result.returncode:
                return result.returncode
        try:
            return subprocess.run(
                ["git", "push", "origin", "HEAD:main"],
                cwd=ROOT,
                timeout=args.git_push_timeout,
            ).returncode
        except subprocess.TimeoutExpired:
            return 124

    return run


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads((args.audit_root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "audit_passed_ready_for_model_evaluation":
        raise SystemExit("selection audit did not pass; overnight sweep is blocked")
    unit_args = argparse.Namespace(
        validation_fit_rows=200,
        validation_early_stop_rows=100,
        validation_context_rows=200,
        validation_holdout_rows=200,
    )
    units = _units(
        args.audit_root,
        summary,
        families=["tree", "tabpfn"],
        mode="formal",
        model_seed=args.model_seed,
        args=unit_args,
    )
    if args.mode == "plan":
        completed = sum(_complete(unit) for unit in units)
        print(
            json.dumps(
                {
                    "mode": "plan",
                    "units": len(units),
                    "completed": completed,
                    "pending": len(units) - completed,
                    "unit_identities": [unit["identity"] for unit in units],
                },
                indent=2,
            )
        )
        return 0
    supervisor_root = args.audit_root / "overnight"
    fatal_marker = supervisor_root / "FAILED.json"
    if fatal_marker.is_file():
        _event(supervisor_root, "queue_blocked_by_fatal_marker", path=str(fatal_marker))
        return 2

    failed = _run_units(args, units, supervisor_root)
    seeds = [int(seed) for seed in summary["building_seeds"]]
    aggregate_root, aggregate_summary = _aggregate_paths(args.audit_root, seeds)
    if not failed:
        aggregate_ok = _bounded_stage(
            args,
            supervisor_root,
            stage="aggregate",
            action=_aggregate_action(args.audit_root, units, seeds),
            valid=lambda: (
                aggregate_summary.is_file()
                and (aggregate_root / "building_seed_summary.csv").is_file()
            ),
        )
        if not aggregate_ok:
            failed.append("aggregate")
    if not failed:
        report_ok = _bounded_stage(
            args,
            supervisor_root,
            stage="report",
            action=_report_action(args),
            valid=lambda: _report_complete(args.report),
        )
        if not report_ok:
            failed.append("report")
    if not failed and args.publish_results:
        published_marker = supervisor_root / "PUBLISHED.json"

        def publish() -> int:
            returncode = _publication_action(args)()
            if returncode == 0:
                _atomic_json(
                    published_marker,
                    {"status": "published", "timestamp": time.time()},
                )
            return returncode

        publish_ok = _bounded_stage(
            args,
            supervisor_root,
            stage="publish",
            action=publish,
            valid=published_marker.is_file,
            retries=args.push_retries - 1,
        )
        if not publish_ok:
            failed.append("publish")

    status = "completed_with_failures" if failed else "completed"
    payload = _status_payload(
        units,
        status=status,
        failed_stages=failed,
        aggregate_root=str(aggregate_root) if aggregate_summary.is_file() else None,
        report=str(args.report) if _report_complete(args.report) else None,
    )
    _write_status(args.audit_root, supervisor_root, payload)
    _atomic_json(supervisor_root / "COMPLETE.json", payload)
    _event(supervisor_root, "queue_complete", status=status, failed_stages=failed)
    return 0


def _entrypoint() -> int:
    try:
        return main()
    except Exception as error:
        root = DEFAULT_AUDIT_ROOT / "overnight"
        payload = {
            "status": "failed",
            "reason": "non_retryable_supervisor_failure",
            "error": repr(error),
            "timestamp": time.time(),
        }
        _atomic_json(root / "FAILED.json", payload)
        _event(root, "queue_failed", error=repr(error))
        raise


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
