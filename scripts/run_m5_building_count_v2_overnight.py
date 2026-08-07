"""Persistent bounded-retry supervisor for the M5 building-count V2 sweep.

Execution order is scientific protocol: finish the first building seed across all
K budgets, then sweep the remaining seeds one K budget at a time. Each seed/K
pair contains a frozen no-early-stopping tree cell and a TabPFN cell. Checkpoint
and resume remain owned by the cell runners.
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

from lead import ROOT
from run_m5_building_count_v2 import (
    DEFAULT_AUDIT_ROOT,
    DEFAULT_OUT_ROOT,
    _complete,
    build_units,
    matched_context_gate,
    ordered_seed_budget_pairs,
)

DEFAULT_REPORT = ROOT / "docs" / "reports" / "m5-building-count-experiment_V2.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
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
    if args.push_retries < 1 or args.git_push_timeout < 1:
        raise ValueError("push retries and timeout must be positive")
    if args.gpu_wait_checks < 1:
        raise ValueError("GPU wait checks must be positive")
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


def _sweep_root(args: argparse.Namespace, summary: dict[str, Any]) -> Path:
    tag = "building_seed_sweep_" + "-".join(
        str(seed) for seed in summary["building_seeds"]
    )
    return args.out_root / tag


def _unit_stage(unit: dict[str, Any]) -> str:
    identity = unit["identity"]
    return (
        f"building_seed{identity['building_seed']}_k{identity['K']}_{identity['model']}"
    )


def _pair_stage(pair: tuple[int, int]) -> str:
    return f"building_seed{pair[0]}_k{pair[1]}"


def _attempt_path(supervisor_root: Path, stage: str) -> Path:
    return supervisor_root / "attempts" / f"{stage}.json"


def _failure_path(supervisor_root: Path, stage: str) -> Path:
    return supervisor_root / "failed_stages" / f"{stage}.json"


def _publication_failure_path(supervisor_root: Path, stage: str) -> Path:
    return supervisor_root / "failed_publications" / f"{stage}.json"


def _published_path(supervisor_root: Path, stage: str) -> Path:
    return supervisor_root / "published" / f"{stage}.json"


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


def _mark_failure(
    path: Path,
    *,
    stage: str,
    attempts: int,
    reason: str,
    returncode: int | None,
) -> None:
    _atomic_json(
        path,
        {
            "status": "failed",
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
    failed_units: list[str],
    failed_publications: list[str],
    **values: Any,
) -> dict[str, Any]:
    completed = sum(_complete(unit) for unit in units)
    return {
        "experiment_version": "m5_building_count_v2",
        "mode": "formal",
        "status": status,
        "units": len(units),
        "completed": completed,
        "pending": len(units) - completed,
        "failed_units": failed_units,
        "failed_publications": failed_publications,
        "timestamp": time.time(),
        **values,
    }


def _write_status(
    audit_root: Path,
    supervisor_root: Path,
    payload: dict[str, Any],
) -> None:
    _atomic_json(audit_root / "model_sweep_v2_status.json", payload)
    _atomic_json(supervisor_root / "status.json", payload)


def _tracked_dirty_paths() -> set[str]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {line[3:] for line in status.splitlines() if line.strip()}


def _clean_gate(report: Path) -> None:
    relative_report = str(report.relative_to(ROOT))
    dirty = _tracked_dirty_paths()
    if dirty - {relative_report}:
        raise RuntimeError(
            f"V2 overnight found unexpected tracked changes: {sorted(dirty)}"
        )


def _run_unit(
    args: argparse.Namespace,
    unit: dict[str, Any],
    supervisor_root: Path,
    units: list[dict[str, Any]],
    *,
    failed_units: list[str],
    failed_publications: list[str],
    pair_index: int,
    pair_count: int,
) -> bool:
    stage = _unit_stage(unit)
    failure = _failure_path(supervisor_root, stage)
    if _complete(unit):
        _event(supervisor_root, "unit_reused", stage=stage)
        return True
    if failure.is_file():
        if stage not in failed_units:
            failed_units.append(stage)
        _event(supervisor_root, "unit_skipped_failed_marker", stage=stage)
        return False

    attempts = _attempt_count(supervisor_root, stage)
    max_attempts = 1 + args.unit_retries
    last_returncode: int | None = None
    while attempts < max_attempts and not _complete(unit):
        attempts += 1
        _record_attempt(supervisor_root, stage, attempts)
        _write_status(
            args.audit_root,
            supervisor_root,
            _status_payload(
                units,
                status="running",
                failed_units=failed_units,
                failed_publications=failed_publications,
                current_pair=pair_index,
                pair_count=pair_count,
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
        if not _complete(unit) and attempts < max_attempts:
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
        return True
    _mark_failure(
        failure,
        stage=stage,
        attempts=attempts,
        reason="unit_retry_limit_exhausted",
        returncode=last_returncode,
    )
    failed_units.append(stage)
    _event(
        supervisor_root,
        "unit_failed",
        stage=stage,
        attempts=attempts,
        returncode=last_returncode,
    )
    return False


def _run_command(command: list[str]) -> int:
    return subprocess.run(command, cwd=ROOT).returncode


def _bounded_action(
    args: argparse.Namespace,
    supervisor_root: Path,
    *,
    stage: str,
    action: Callable[[], int],
    valid: Callable[[], bool],
    attempts_root: str = "attempts",
) -> bool:
    if valid():
        return True
    attempt_path = supervisor_root / attempts_root / f"{stage}.json"
    attempts = 0
    if attempt_path.is_file():
        try:
            attempts = max(
                0,
                int(json.loads(attempt_path.read_text(encoding="utf-8"))["attempt"]),
            )
        except (OSError, KeyError, TypeError, ValueError):
            attempts = 0
    max_attempts = 1 + args.finalize_retries
    last_returncode: int | None = None
    while attempts < max_attempts and not valid():
        attempts += 1
        _atomic_json(
            attempt_path,
            {"stage": stage, "attempt": attempts, "timestamp": time.time()},
        )
        last_returncode = action()
        if last_returncode == 0 and valid():
            return True
        _event(
            supervisor_root,
            "bounded_action_retry",
            stage=stage,
            attempt=attempts,
            returncode=last_returncode,
        )
        if attempts < max_attempts:
            time.sleep(args.retry_delay)
    _mark_failure(
        _failure_path(supervisor_root, stage),
        stage=stage,
        attempts=attempts,
        reason="bounded_action_retry_limit_exhausted",
        returncode=last_returncode,
    )
    return False


def _push_with_retries(
    args: argparse.Namespace,
    supervisor_root: Path,
    stage: str,
) -> bool:
    for attempt in range(1, args.push_retries + 1):
        try:
            result = subprocess.run(
                ["git", "push", "origin", "HEAD:main"],
                cwd=ROOT,
                timeout=args.git_push_timeout,
            )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
        if returncode == 0:
            _event(
                supervisor_root,
                "push_complete",
                stage=stage,
                attempt=attempt,
            )
            return True
        _event(
            supervisor_root,
            "push_retry",
            stage=stage,
            attempt=attempt,
            returncode=returncode,
        )
        if attempt < args.push_retries:
            time.sleep(min(args.retry_delay * attempt, 600))
    _mark_failure(
        _publication_failure_path(supervisor_root, stage),
        stage=stage,
        attempts=args.push_retries,
        reason="git_push_retry_limit_exhausted",
        returncode=returncode,
    )
    return False


def _update_progress(
    args: argparse.Namespace,
    supervisor_root: Path,
    pair: tuple[int, int],
) -> int:
    return _run_command(
        [
            sys.executable,
            "scripts/update_m5_building_count_v2_progress.py",
            "--audit-root",
            str(args.audit_root),
            "--out-root",
            str(args.out_root),
            "--supervisor-root",
            str(supervisor_root),
            "--report",
            str(args.report),
            "--last-pair",
            _pair_stage(pair),
        ]
    )


def _commit_and_push_report(
    args: argparse.Namespace,
    supervisor_root: Path,
    pair: tuple[int, int],
) -> bool:
    stage = _pair_stage(pair)
    published = _published_path(supervisor_root, stage)
    if published.is_file():
        _event(supervisor_root, "pair_publication_reused", stage=stage)
        return True
    if _update_progress(args, supervisor_root, pair):
        return False
    _clean_gate(args.report)
    relative_report = str(args.report.relative_to(ROOT))
    if relative_report in _tracked_dirty_paths():
        if subprocess.run(["git", "add", relative_report], cwd=ROOT).returncode:
            raise RuntimeError(f"failed to stage V2 progress report for {stage}")
        message = f"Record M5 V2 building seed {pair[0]} K{pair[1]}"
        if subprocess.run(["git", "commit", "-m", message], cwd=ROOT).returncode:
            raise RuntimeError(f"failed to commit V2 progress report for {stage}")
    if not _push_with_retries(args, supervisor_root, stage):
        return False
    _atomic_json(
        published,
        {"stage": stage, "commit": _git_head(), "timestamp": time.time()},
    )
    return True


def _recover_dirty_report(
    args: argparse.Namespace,
    supervisor_root: Path,
) -> bool:
    """Commit an atomic report update left between write and commit."""
    relative_report = str(args.report.relative_to(ROOT))
    if relative_report not in _tracked_dirty_paths():
        return True
    if not args.publish_results:
        raise RuntimeError("dirty V2 report requires --publish-results recovery")
    if subprocess.run(["git", "add", relative_report], cwd=ROOT).returncode:
        raise RuntimeError("failed to stage recovered V2 progress report")
    if subprocess.run(
        ["git", "commit", "-m", "Recover M5 V2 overnight progress"],
        cwd=ROOT,
    ).returncode:
        raise RuntimeError("failed to commit recovered V2 progress report")
    return _push_with_retries(
        args,
        supervisor_root,
        "recovered_progress_report",
    )


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _pair_gate(
    supervisor_root: Path,
    pair: tuple[int, int],
    pair_units: list[dict[str, Any]],
) -> None:
    records = matched_context_gate(pair_units)
    if len(records) != 1 or not records[0]["passed"]:
        raise AssertionError(f"matched-context gate failed for {_pair_stage(pair)}")
    _atomic_json(
        supervisor_root / "pair_gates" / f"{_pair_stage(pair)}.json",
        {"passed": True, "records": records, "timestamp": time.time()},
    )


def _finalize(
    args: argparse.Namespace,
    supervisor_root: Path,
    summary: dict[str, Any],
) -> bool:
    stage = "final_aggregate_report"
    marker = supervisor_root / "FINALIZED.json"

    def action() -> int:
        result = _run_command(
            [
                sys.executable,
                "scripts/run_m5_building_count_v2.py",
                "--audit-root",
                str(args.audit_root),
                "--out-root",
                str(args.out_root),
                "--report",
                str(args.report),
                "--mode",
                "formal",
            ]
        )
        if result:
            return result
        relative_report = str(args.report.relative_to(ROOT))
        if relative_report in _tracked_dirty_paths():
            if subprocess.run(["git", "add", relative_report], cwd=ROOT).returncode:
                return 2
            if subprocess.run(
                ["git", "commit", "-m", "Finalize M5 building-count V2 report"],
                cwd=ROOT,
            ).returncode:
                return 2
        if args.publish_results and not _push_with_retries(
            args, supervisor_root, stage
        ):
            return 3
        _atomic_json(
            marker,
            {
                "status": "finalized",
                "commit": _git_head(),
                "building_seeds": summary["building_seeds"],
                "budgets": summary["budgets"],
                "timestamp": time.time(),
            },
        )
        return 0

    return _bounded_action(
        args,
        supervisor_root,
        stage=stage,
        action=action,
        valid=marker.is_file,
        attempts_root="finalize_attempts",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads((args.audit_root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "audit_passed_ready_for_model_evaluation":
        raise SystemExit("V2 sampling audit did not pass")
    units = build_units(
        args.audit_root,
        args.out_root,
        summary,
        families=["tree", "tabpfn"],
        mode="formal",
        model_seed=args.model_seed,
        validation_context_rows=200,
        validation_holdout_rows=200,
    )
    pairs = ordered_seed_budget_pairs(summary)
    pair_units: dict[tuple[int, int], list[dict[str, Any]]] = {
        pair: [
            unit
            for unit in units
            if (
                unit["identity"]["building_seed"],
                unit["identity"]["K"],
            )
            == pair
        ]
        for pair in pairs
    }
    if args.mode == "plan":
        print(
            json.dumps(
                {
                    "mode": "plan",
                    "pair_order": [
                        {
                            "order": index,
                            "building_seed": pair[0],
                            "K": pair[1],
                        }
                        for index, pair in enumerate(pairs, start=1)
                    ],
                    "units": len(units),
                    "completed": sum(_complete(unit) for unit in units),
                    "pending": sum(not _complete(unit) for unit in units),
                    "publish_after_each_pair": args.publish_results,
                },
                indent=2,
            )
        )
        return 0

    sweep_root = _sweep_root(args, summary)
    supervisor_root = sweep_root / "overnight"
    fatal_marker = supervisor_root / "FAILED.json"
    if fatal_marker.is_file():
        _event(
            supervisor_root,
            "queue_blocked_by_fatal_marker",
            path=str(fatal_marker),
        )
        return 2
    _clean_gate(args.report)

    failed_units: list[str] = []
    failed_publications: list[str] = []
    if not _recover_dirty_report(args, supervisor_root):
        failed_publications.append("recovered_progress_report")
        _event(
            supervisor_root,
            "recovered_progress_push_failed",
        )
    for pair_index, pair in enumerate(pairs, start=1):
        current_units = pair_units[pair]
        _event(
            supervisor_root,
            "pair_start",
            pair=_pair_stage(pair),
            pair_index=pair_index,
            pair_count=len(pairs),
        )
        pair_ok = True
        for unit in current_units:
            if not _run_unit(
                args,
                unit,
                supervisor_root,
                units,
                failed_units=failed_units,
                failed_publications=failed_publications,
                pair_index=pair_index,
                pair_count=len(pairs),
            ):
                pair_ok = False
        if pair_ok:
            _pair_gate(supervisor_root, pair, current_units)
        if args.publish_results:
            if not _commit_and_push_report(args, supervisor_root, pair):
                stage = _pair_stage(pair)
                failed_publications.append(stage)
                _event(
                    supervisor_root,
                    "pair_publication_failed",
                    stage=stage,
                )
        _write_status(
            args.audit_root,
            supervisor_root,
            _status_payload(
                units,
                status="running",
                failed_units=failed_units,
                failed_publications=failed_publications,
                completed_pair=_pair_stage(pair),
                completed_pair_index=pair_index,
                pair_count=len(pairs),
            ),
        )
        _event(
            supervisor_root,
            "pair_end",
            pair=_pair_stage(pair),
            complete=pair_ok,
        )

    finalized = False
    if not failed_units:
        finalized = _finalize(args, supervisor_root, summary)
        if not finalized:
            failed_publications.append("final_aggregate_report")

    status = (
        "completed"
        if not failed_units and not failed_publications and finalized
        else "completed_with_failures"
    )
    payload = _status_payload(
        units,
        status=status,
        failed_units=failed_units,
        failed_publications=failed_publications,
        finalized=finalized,
        pair_count=len(pairs),
    )
    _write_status(args.audit_root, supervisor_root, payload)
    _atomic_json(supervisor_root / "COMPLETE.json", payload)
    _event(supervisor_root, "queue_complete", **payload)
    return 0


def _entrypoint() -> int:
    args = parse_args()
    try:
        return main(sys.argv[1:])
    except Exception as error:
        try:
            summary = json.loads(
                (args.audit_root / "summary.json").read_text(encoding="utf-8")
            )
            root = _sweep_root(args, summary) / "overnight"
        except Exception:
            root = args.out_root / "overnight"
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
