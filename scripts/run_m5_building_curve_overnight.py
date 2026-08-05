"""Persistent, resumable supervisor for the authorized M5 overnight queue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from lead import PROC, ROOT

PROTOCOL_ROOT = PROC / "m5_building_curve" / "protocol"
FULL_PROTOCOL_ROOT = PROC / "m5_building_curve" / "protocol_full"
MANIFEST = PROTOCOL_ROOT / "representative" / "seed42" / "building_ladder.json"
FULL_MANIFEST = (
    FULL_PROTOCOL_ROOT / "representative" / "seed42" / "building_ladder.json"
)
FORMAL_ROOT = PROC / "m5_building_curve" / "formal"
SUPERVISOR_ROOT = PROC / "m5_building_curve" / "supervisor"
REPORT = ROOT / "docs" / "reports" / "m5-building-count-experiment.md"
FAILED_MARKER = SUPERVISOR_ROOT / "FAILED.json"
PUBLISHED_ROOT = SUPERVISOR_ROOT / "published"
STAGE_FAILED_ROOT = SUPERVISOR_ROOT / "failed_stages"


class GitPushFailed(RuntimeError):
    """A retryable git-push failure after one bounded attempt group."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-delay", type=int, default=120)
    parser.add_argument("--stage-retries", type=int, default=2)
    parser.add_argument("--push-retries", type=int, default=5)
    parser.add_argument("--git-push-timeout", type=int, default=120)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _event(event: str, **values: Any) -> None:
    SUPERVISOR_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.time(), "event": event, **values}
    with (SUPERVISOR_ROOT / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + chr(10))
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(payload, sort_keys=True), flush=True)


def _mark_failed(state_path: Path, **values: Any) -> None:
    payload = {"status": "failed", "timestamp": time.time(), **values}
    _atomic_json(state_path, payload)
    _atomic_json(FAILED_MARKER, payload)


def _stage_failed_marker(stage_name: str) -> Path:
    return STAGE_FAILED_ROOT / f"{stage_name}.json"


def _mark_stage_failed(state_path: Path, **values: Any) -> None:
    payload = {
        "status": "stage_failed",
        "requires_review": True,
        "timestamp": time.time(),
        **values,
    }
    _atomic_json(state_path, payload)
    _atomic_json(_stage_failed_marker(str(values["stage"])), payload)


def _prior_stage_attempts(state_path: Path, stage_name: str) -> int:
    """Recover an in-flight attempt count after supervisor/watchdog restart."""
    if not state_path.is_file():
        return 0
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if state.get("status") != "running" or state.get("stage") != stage_name:
        return 0
    return max(0, int(state.get("attempt", 0)))


def _published_marker(stage_name: str) -> Path:
    return PUBLISHED_ROOT / f"{stage_name}.json"


def _command(
    command: list[str],
    *,
    check: bool = True,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    _event("command_start", command=command)
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        _event(
            "command_timeout",
            command=command,
            timeout_seconds=timeout_seconds,
        )
        result = subprocess.CompletedProcess(command, 124)
    _event("command_end", command=command, returncode=result.returncode)
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result


def _wait_for_idle_gpu(stage: str, state_path: Path, delay: int) -> None:
    """Do not start a TabPFN cell while another compute process owns the GPU."""
    while True:
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
        active = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode == 0 and not active:
            return
        _atomic_json(
            state_path,
            {
                "status": "waiting_for_idle_gpu",
                "stage": stage,
                "active_gpu_processes": active,
                "timestamp": time.time(),
            },
        )
        _event("gpu_wait", stage=stage, active_gpu_processes=active)
        time.sleep(delay)


def _clean_gate() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty = {line[3:] for line in status.splitlines() if line.strip()}
    allowed = {str(REPORT.relative_to(ROOT))}
    if dirty - allowed:
        raise RuntimeError(
            f"overnight formal queue found unexpected dirty paths: {sorted(dirty - allowed)}"
        )


def _ensure_protocols() -> None:
    requests = (
        (
            MANIFEST,
            [
                sys.executable,
                "scripts/prepare_m5_building_curve.py",
                "--budgets",
                "10",
                "20",
                "50",
                "100",
                "--row-policy",
                "average_building_cap",
                "--average-building-rows",
                "500",
                "--max-context-rows",
                "50000",
                "--out-root",
                str(PROTOCOL_ROOT),
            ],
        ),
        (
            FULL_MANIFEST,
            [
                sys.executable,
                "scripts/prepare_m5_building_curve.py",
                "--budgets",
                "725",
                "--row-policy",
                "all_rows",
                "--out-root",
                str(FULL_PROTOCOL_ROOT),
            ],
        ),
    )
    for path, command in requests:
        if path.is_file():
            _event("protocol_reused", path=str(path))
        else:
            _command(command)
    primary = json.loads(MANIFEST.read_text(encoding="utf-8"))
    full = json.loads(FULL_MANIFEST.read_text(encoding="utf-8"))
    if primary["budgets"] != [10, 20, 50, 100]:
        raise AssertionError("primary manifest budgets drifted")
    if primary["row_policy"] != "average_building_cap":
        raise AssertionError("primary manifest row policy drifted")
    if int(primary["average_rows_per_building_limit"]) != 500:
        raise AssertionError("average building row cap drifted")
    if int(primary["max_context_rows"]) != 50_000:
        raise AssertionError("TabPFN context exceeds or differs from 50K")
    row_counts = [
        int(primary["cells"][str(budget)]["available_rows"])
        for budget in primary["budgets"]
    ]
    if any(rows > budget * 500 for rows, budget in zip(row_counts, primary["budgets"])):
        raise AssertionError("a K cell exceeds its per-building row upper bound")
    if any(left >= right for left, right in zip(row_counts, row_counts[1:])):
        raise AssertionError("K row sets are not strict nested growth")
    if full["budgets"] != [725] or full["row_policy"] != "all_rows":
        raise AssertionError("full-building baseline manifest drifted")


def _stages() -> list[dict[str, Any]]:
    tree_script = "scripts/run_m5_building_curve_tree_cell.py"
    tabpfn_script = "scripts/run_m5_building_curve_tabpfn_cell.py"

    def tree(
        name: str, manifest: Path, budget: int, features: int, publish: bool
    ) -> dict[str, Any]:
        out = FORMAL_ROOT / name
        return {
            "name": name,
            "out": out,
            "publish": publish,
            "command": [
                sys.executable,
                tree_script,
                "--building-manifest",
                str(manifest),
                "--building-budget",
                str(budget),
                "--features",
                str(features),
                "--mode",
                "formal",
                "--resume",
                "--out-root",
                str(out),
            ],
        }

    def tabpfn(name: str, budget: int, publish: bool) -> dict[str, Any]:
        out = FORMAL_ROOT / name
        return {
            "name": name,
            "out": out,
            "publish": publish,
            "command": [
                sys.executable,
                tabpfn_script,
                "--building-manifest",
                str(MANIFEST),
                "--building-budget",
                str(budget),
                "--features",
                "137",
                "--n-estimators",
                "8",
                "--mode",
                "formal",
                "--resume",
                "--out-root",
                str(out),
            ],
        }

    stages = [
        tree("tree_full_f17", FULL_MANIFEST, 725, 17, True),
        tree("tree_full_f137", FULL_MANIFEST, 725, 137, True),
    ]
    for budget in (10, 20, 50, 100):
        stages.append(tree(f"tree_k{budget}_f137", MANIFEST, budget, 137, False))
        stages.append(tabpfn(f"tabpfn_k{budget}_f137", budget, True))
    return stages


def _valid_tree_contract(metadata: dict[str, Any], stored: Any) -> bool:
    """Reject completed artifacts created by an obsolete tree protocol."""
    if (
        metadata.get("training_sampling")
        != "M3 post-feature-sort:[negs1,pos,negs2,pos]"
    ):
        return False
    if metadata.get("training_sampling_seeds") != [10, 20]:
        return False
    if metadata.get("training_sampling_order") != ["building_id", "timestamp"]:
        return False
    if metadata.get("matrix_dtype") != "float64":
        return False
    if metadata.get("prediction_dtype") != "float64":
        return False
    if metadata.get("early_stopping_metric") != "pr_auc":
        return False
    contract = metadata.get("fit", {}).get("model_contract", {})
    score_names = metadata.get("score_names", [])
    if not contract or any(
        spec.get("selection_metric") != "pr_auc" for spec in contract.values()
    ):
        return False
    return all(stored[name].dtype == np.dtype("float64") for name in score_names)


def _valid_complete(stage: dict[str, Any]) -> bool:
    output = Path(stage["out"])
    complete = output / "COMPLETE.json"
    cell = output / "cell.json"
    predictions = output / "predictions.npz"
    if not complete.is_file() or not cell.is_file() or not predictions.is_file():
        return False
    try:
        marker = json.loads(complete.read_text(encoding="utf-8"))
        metadata = json.loads(cell.read_text(encoding="utf-8"))
        if Path(metadata["predictions"]).name != predictions.name:
            return False
        with np.load(predictions) as stored:
            required = {
                "validation_raw_index",
                "anomaly",
                "building_id",
                "site_id",
                "meter",
                *metadata["score_names"],
            }
            if not required.issubset(stored.files):
                return False
            size = len(stored["validation_raw_index"])
            if size != 10_137_155:
                return False
            if stage["name"].startswith("tree_") and not _valid_tree_contract(
                metadata, stored
            ):
                return False
        return bool(marker.get("cell"))
    except Exception as error:
        _event("completion_validation_failed", stage=stage["name"], error=repr(error))
        return False


def _prepare_publication(stage_name: str) -> None:
    _command(
        [
            sys.executable,
            "scripts/update_m5_building_curve_report.py",
            "--manifest",
            str(MANIFEST),
        ]
    )
    _command(["git", "add", str(REPORT.relative_to(ROOT))])
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=ROOT,
    ).returncode
    if changed:
        _command(["git", "commit", "-m", f"Record M5 building stage {stage_name}"])


def _push_publication(stage_name: str, push_retries: int, push_timeout: int) -> None:
    for attempt in range(1, push_retries + 1):
        result = _command(
            ["git", "push", "github", "HEAD:main"],
            check=False,
            timeout_seconds=push_timeout,
        )
        if result.returncode == 0:
            _event("publish_complete", stage=stage_name, attempt=attempt)
            return
        _event(
            "push_retry",
            stage=stage_name,
            attempt=attempt,
            returncode=result.returncode,
        )
        time.sleep(min(60 * attempt, 300))
    raise GitPushFailed(f"failed to push publication gate {stage_name}")


def main() -> int:
    args = parse_args()
    if (
        args.retry_delay < 1
        or args.stage_retries < 0
        or args.push_retries < 1
        or args.git_push_timeout < 1
    ):
        raise ValueError("retry delay/count and git push timeout are invalid")
    if FAILED_MARKER.is_file():
        _event("queue_blocked_by_failed_marker", path=str(FAILED_MARKER))
        return 2
    _clean_gate()
    _ensure_protocols()
    stages = _stages()
    state_path = SUPERVISOR_ROOT / "status.json"
    max_attempts = 1 + args.stage_retries
    failed_stages: list[str] = []
    for index, stage in enumerate(stages):
        stage_failed = _stage_failed_marker(stage["name"])
        if stage_failed.is_file():
            failed_stages.append(stage["name"])
            _event(
                "stage_skipped_failed_marker",
                stage=stage["name"],
                marker=str(stage_failed),
                requires_review=True,
            )
            continue
        attempts = _prior_stage_attempts(state_path, stage["name"])
        while not _valid_complete(stage):
            if attempts >= max_attempts:
                failure = {
                    "stage_index": index,
                    "stage_count": len(stages),
                    "stage": stage["name"],
                    "attempts": attempts,
                    "max_attempts": max_attempts,
                    "reason": "stage_retry_limit_exhausted",
                }
                _mark_stage_failed(state_path, **failure)
                _event("stage_failed", **failure)
                failed_stages.append(stage["name"])
                break
            attempts += 1
            if stage["name"].startswith("tabpfn_"):
                _wait_for_idle_gpu(stage["name"], state_path, args.retry_delay)
            _atomic_json(
                state_path,
                {
                    "status": "running",
                    "stage_index": index,
                    "stage_count": len(stages),
                    "stage": stage["name"],
                    "attempt": attempts,
                    "timestamp": time.time(),
                },
            )
            result = _command(stage["command"], check=False)
            if result.returncode == 0 and _valid_complete(stage):
                break
            _event(
                "stage_retry",
                stage=stage["name"],
                attempt=attempts,
                returncode=result.returncode,
                delay_seconds=(args.retry_delay if attempts < max_attempts else 0),
            )
            if attempts < max_attempts:
                time.sleep(args.retry_delay)
        if stage["name"] in failed_stages:
            continue
        _event("stage_complete", stage=stage["name"], attempt=attempts)
        if stage["publish"]:
            published = _published_marker(stage["name"])
            if published.is_file():
                _event("publication_reused", stage=stage["name"])
            else:
                # Report generation/add/commit are deterministic local gates:
                # failures there are fatal. Only git push is retried forever.
                _prepare_publication(stage["name"])
                while True:
                    try:
                        _push_publication(
                            stage["name"],
                            args.push_retries,
                            args.git_push_timeout,
                        )
                        break
                    except GitPushFailed as error:
                        _event(
                            "publication_retry",
                            stage=stage["name"],
                            error=repr(error),
                            delay_seconds=args.retry_delay,
                        )
                        time.sleep(args.retry_delay)
                _atomic_json(
                    published,
                    {"stage": stage["name"], "timestamp": time.time()},
                )
        _atomic_json(
            state_path,
            {
                "status": "running",
                "completed_stage": stage["name"],
                "completed_count": index + 1,
                "stage_count": len(stages),
                "timestamp": time.time(),
            },
        )

    queue_status = "completed_with_failures" if failed_stages else "completed"
    _atomic_json(
        state_path,
        {
            "status": queue_status,
            "completed_count": len(stages),
            "stage_count": len(stages),
            "failed_stages": failed_stages,
            "timestamp": time.time(),
        },
    )
    _atomic_json(
        SUPERVISOR_ROOT / "COMPLETE.json",
        {
            "status": queue_status,
            "report": str(REPORT),
            "failed_stages": failed_stages,
        },
    )
    _event("queue_complete", stages=len(stages), failed_stages=failed_stages)
    return 0


def _entrypoint() -> int:
    """Persist a fatal marker for non-retryable supervisor failures."""
    try:
        return main()
    except Exception as error:
        state_path = SUPERVISOR_ROOT / "status.json"
        if not FAILED_MARKER.is_file():
            _mark_failed(
                state_path,
                phase="supervisor",
                reason="non_retryable_supervisor_failure",
                error=repr(error),
            )
        _event("queue_failed", error=repr(error))
        raise


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
