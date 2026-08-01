"""E3 fresh-process reload diagnostic.

Runs exactly one reload-and-score pass per invocation, in its own process, from
the persisted state of the protocol's designated cell. Results are written to a
separate `fresh/` directory and are never mixed into the same-process repeat
distribution, never used as a scientific estimate, and never allowed to control
how many repeats a cell runs.

Only a load failure, a version or digest mismatch, a row-identity mismatch, a
non-finite output, or a schema error is a lifecycle hard failure. A purely
numerical difference from the same-process distribution is recorded, not judged
against any invented threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from m5_e3_runner import CELLS, atomic_json, endpoints, load_cell, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--run-index", type=int, required=True)
    args = ap.parse_args()

    proto = json.loads(args.protocol.read_text(encoding="utf-8"))["protocol"]
    spec = proto["fresh_process_diagnostic"]
    if args.cell != spec["cell"]:
        raise SystemExit(
            f"protocol fixes the fresh-process diagnostic to cell {spec['cell']}"
        )
    if not (0 <= args.run_index < spec["runs"]):
        raise SystemExit(f"protocol allows {spec['runs']} fresh runs")

    cell_root = args.run_root / f"cell_{args.cell}"
    fresh_dir = cell_root / "fresh"
    fresh_dir.mkdir(parents=True, exist_ok=True)
    out = fresh_dir / f"fresh_{args.run_index:02d}.json"
    if out.exists():
        print(f"fresh run {args.run_index} already recorded")
        return 0

    fit = json.loads((cell_root / "fit_complete.json").read_text(encoding="utf-8"))
    state_path = cell_root / "model.tabpfn_fit"

    import tabpfn
    import torch
    from tabpfn.model_loading import load_fitted_tabpfn_model

    required = proto["inherited"]["scientific_tabpfn_version"]
    if tabpfn.__version__ != required:
        raise SystemExit(
            f"HARD FAILURE version mismatch: {tabpfn.__version__} != {required}"
        )
    actual_state = sha256_file(state_path)
    if actual_state != fit["state_sha256"]:
        raise SystemExit(
            f"HARD FAILURE state digest drifted: {actual_state} != {fit['state_sha256']}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("HARD FAILURE CUDA unavailable; CPU fallback is prohibited")

    data = load_cell(args.cell, args.model_path)
    if (
        data["query_sha256"]
        != json.loads((cell_root / "fit_start.json").read_text(encoding="utf-8"))[
            "query_sha256"
        ]
    ):
        raise SystemExit("HARD FAILURE query row identity differs from the fit")

    t0 = time.perf_counter()
    model = load_fitted_tabpfn_model(state_path, device="cuda:0")
    load_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    score = np.asarray(model.predict_proba(data["q"])[:, 1], dtype="float64")
    score_seconds = time.perf_counter() - t1
    if not np.all(np.isfinite(score)):
        raise SystemExit("HARD FAILURE non-finite scores from the reloaded state")

    ep = endpoints(score, data["q_meter"], data["q_anom"])
    atomic_json(
        out,
        {
            "cell": args.cell,
            "run_index": args.run_index,
            "mode": "fresh_process_reload",
            "excluded_from_same_process_statistics": True,
            "scientific_estimate": False,
            "state_sha256": actual_state,
            "load_seconds": load_seconds,
            "score_seconds": score_seconds,
            "endpoints": ep,
            "score_sha256": hashlib.sha256(score.tobytes()).hexdigest(),
            "environment": {
                "tabpfn": tabpfn.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "python": platform.python_version(),
            },
            "timestamp": time.time(),
        },
    )
    print(
        f"fresh {args.cell} run {args.run_index}: load={load_seconds:,.1f}s "
        f"score={score_seconds:,.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
