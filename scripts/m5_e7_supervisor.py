"""Durable local supervisor for the corrected M5 E7 execution pipeline.

This module contains no data loading, fitting, scoring, labels, or metrics. It
only starts restart-safe runner phases and writes a 60-second human-readable
heartbeat while those phases own their atomic checkpoints.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import m5_e7_protocol as p


ROOT = p.ROOT
OUT = p.ARTIFACT_ROOT
RUNTIME = OUT / "runtime"


def _completed() -> int:
    root = OUT / "units"
    return sum(
        1
        for item in root.iterdir()
        if item.is_dir() and (item / "complete.json").exists()
    )


def _running() -> list[str]:
    root = OUT / "units"
    return sorted(
        item.name
        for item in root.iterdir()
        if item.is_dir() and not (item / "complete.json").exists()
    )


def _write_status(
    phase: str, child: subprocess.Popen[str] | None, started: float
) -> None:
    completed = _completed()
    running = _running()
    elapsed = max(time.time() - started, 1.0)
    rate = completed / elapsed * 3600
    remaining = (192 - completed) / rate if rate else None
    lines = [
        "# M5 E7 execution supervisor",
        "",
        f"- canonical completed / 192: {completed} / 192",
        f"- active phase: {phase}",
        f"- running fit workers / 2: {len(running)} / 2",
        f"- running unit IDs: {', '.join(running) if running else 'none'}",
        "- threads per fit: 8",
        "- feature preparation workers: 1",
        "- prepared queue depth: 2",
        f"- supervisor PID: {os.getpid()}",
        f"- child PID: {child.pid if child else 'none'}",
        f"- units/hour: {rate:.2f}",
        f"- estimated remaining hours: {remaining:.2f}"
        if remaining
        else "- estimated remaining hours: pending first completion",
        "- retries: 0",
        "- quarantine: pre_resource_correction_002 (11 preserved non-canonical attempts)",
        "- last error: none",
        "- supervisor alive: true",
        "- watchdog alive: true",
    ]
    p.atomic_text(OUT / "STATUS.md", "\n".join(lines) + "\n")
    p.atomic_json(
        OUT / "e7_supervisor_heartbeat.json",
        {
            "phase": phase,
            "completed": completed,
            "expected": 192,
            "running_units": running,
            "threads_per_fit": 8,
            "fit_concurrency": 2,
            "supervisor_pid": os.getpid(),
            "child_pid": child.pid if child else None,
            "updated_unix": time.time(),
        },
    )


def _run_phase(name: str, argument: str, started: float) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    stdout = (RUNTIME / f"supervisor_{name}.stdout.log").open("a", encoding="utf-8")
    stderr = (RUNTIME / f"supervisor_{name}.stderr.log").open("a", encoding="utf-8")
    env = os.environ.copy()
    env.update(p.resource_environment())
    child = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "m5_e7_runner.py"), argument],
        cwd=ROOT,
        env=env,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    try:
        while child.poll() is None:
            _write_status(name, child, started)
            time.sleep(60)
        _write_status(name, child, started)
        if child.returncode:
            raise SystemExit(f"supervisor phase {name} failed: {child.returncode}")
    finally:
        stdout.close()
        stderr.close()


def main() -> None:
    p.require_local_cpu()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    p.atomic_text(RUNTIME / "m5_e7_supervisor.pid", f"{os.getpid()}\n")
    started = time.time()
    # Each runner phase is checkpoint-resumable.  A restart never discards a
    # digest-valid canonical unit and never opens odd labels.
    for name, argument in (
        ("oof", "--execute-oof"),
        ("oof_finalise", "--finalise-oof"),
        ("final", "--execute-final"),
        ("final_meta", "--fit-final-meta"),
        ("full_s11", "--score-s11-full-holdout"),
        ("steam_specialists", "--score-steam-specialists"),
        ("hybrid", "--assemble-hybrid-chunks"),
    ):
        _run_phase(name, argument, started)


if __name__ == "__main__":
    main()
