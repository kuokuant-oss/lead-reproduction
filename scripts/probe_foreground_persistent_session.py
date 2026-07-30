"""Foreground-only, non-scientific capability probe for a persistent terminal."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--interval-seconds", type=int, default=2)
    parser.add_argument(
        "--status-root", type=Path, default=Path(".scratch") / "m5-e0-foreground-probe"
    )
    args = parser.parse_args()
    if args.seconds < 20 or args.interval_seconds < 1:
        raise ValueError("probe requires at least 20 seconds and a positive interval")
    args.status_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.seconds
    tick = 0
    while True:
        elapsed = time.monotonic() - started
        atomic_json(
            args.status_root / "heartbeat.json",
            {
                "pid": os.getpid(),
                "tick": tick,
                "elapsed_seconds": elapsed,
                "state": "running",
            },
        )
        print(f"probe tick={tick} elapsed={elapsed:.1f}s", flush=True)
        if time.monotonic() >= deadline:
            break
        tick += 1
        time.sleep(args.interval_seconds)
    atomic_json(
        args.status_root / "heartbeat.json",
        {
            "pid": os.getpid(),
            "tick": tick,
            "elapsed_seconds": time.monotonic() - started,
            "state": "complete",
        },
    )
    print("probe complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
