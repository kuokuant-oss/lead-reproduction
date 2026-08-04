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
FULL_MANIFEST = FULL_PROTOCOL_ROOT / "representative" / "seed42" / "building_ladder.json"
FORMAL_ROOT = PROC / "m5_building_curve" / "formal"
SUPERVISOR_ROOT = PROC / "m5_building_curve" / "supervisor"
REPORT = ROOT / "docs" / "reports" / "m5-building-count-experiment.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-delay", type=int, default=120)
    parser.add_argument("--push-retries", type=int, default=5)
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


def _command(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    _event("command_start", command=command)
    result = subprocess.run(command, cwd=ROOT, text=True)
    _event("command_end", command=command, returncode=result.returncode)
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result


def _clean_gate() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty = {
        line[3:]
        for line in status.splitlines()
        if line.strip()
    }
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
    for _path, command in requests:
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

    def tree(name: str, manifest: Path, budget: int, features: int, publish: bool) -> dict[str, Any]:
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
        return bool(marker.get("cell"))
    except Exception as error:
        _event("completion_validation_failed", stage=stage["name"], error=repr(error))
        return False


def _publish(stage_name: str, push_retries: int) -> None:
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
    for attempt in range(1, push_retries + 1):
        result = _command(["git", "push", "github", "HEAD:main"], check=False)
        if result.returncode == 0:
            _event("publish_complete", stage=stage_name, attempt=attempt)
            return
        _event("push_retry", stage=stage_name, attempt=attempt)
        time.sleep(min(60 * attempt, 300))
    raise RuntimeError(f"failed to push publication gate {stage_name}")


def main() -> int:
    args = parse_args()
    if args.retry_delay < 1 or args.push_retries < 1:
        raise ValueError("retry settings must be positive")
    _clean_gate()
    _ensure_protocols()
    stages = _stages()
    state_path = SUPERVISOR_ROOT / "status.json"
    for index, stage in enumerate(stages):
        attempts = 0
        while not _valid_complete(stage):
            attempts += 1
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
                delay_seconds=args.retry_delay,
            )
            time.sleep(args.retry_delay)
        _event("stage_complete", stage=stage["name"], attempt=attempts)
        if stage["publish"]:
            while True:
                try:
                    _publish(stage["name"], args.push_retries)
                    break
                except Exception as error:
                    _event(
                        "publication_retry",
                        stage=stage["name"],
                        error=repr(error),
                        delay_seconds=args.retry_delay,
                    )
                    time.sleep(args.retry_delay)
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

    _atomic_json(
        state_path,
        {
            "status": "completed",
            "completed_count": len(stages),
            "stage_count": len(stages),
            "timestamp": time.time(),
        },
    )
    _atomic_json(SUPERVISOR_ROOT / "COMPLETE.json", {"report": str(REPORT)})
    _event("queue_complete", stages=len(stages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
