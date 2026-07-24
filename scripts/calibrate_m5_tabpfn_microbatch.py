"""Measure how large a query microbatch the target GPU actually sustains.

The official run fixed 1024 because that was safe on a 22 GB L4. On a 40 GB
A100 that is likely to leave throughput on the table, and the worker only ever
*lowers* the microbatch, so the only way to go faster is to hand it a calibrated
starting value. This script is self-contained so it can be uploaded to a Colab
VM and executed there against the real shard inputs and the real fitted state.

For each candidate it predicts real rows from the shard, records rows/second and
the peak GPU fraction, and marks the candidate unusable if it OOMs or exceeds
the soft limit the worker itself would react to. The winner is the largest
usable candidate. Scores produced here are throughput probes and are deliberately
not written as checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--fit-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
    )
    parser.add_argument("--context-rows", type=int, default=100_000)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--warmup-rows", type=int, default=256)
    parser.add_argument("--min-rows-per-candidate", type=int, default=2048)
    parser.add_argument("--max-seconds-per-candidate", type=float, default=150.0)
    parser.add_argument("--gpu-soft-limit-fraction", type=float, default=0.86)
    parser.add_argument("--checkpoint-rows", type=int, default=20_000)
    args = parser.parse_args(argv)
    for value in args.candidates:
        if value < 64:
            raise ValueError("candidates cannot go below the contract minimum 64")
        if value > args.checkpoint_rows:
            raise ValueError(
                f"candidate {value} exceeds --checkpoint-rows {args.checkpoint_rows}"
            )
    return args


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "cuda error" in text


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    os.environ["TABPFN_NO_BROWSER"] = "1"

    import numpy as np
    import torch
    from tabpfn.model_loading import load_fitted_tabpfn_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback forbidden")

    properties = torch.cuda.get_device_properties(0)
    features = np.load(args.features, mmap_mode="r")
    model = load_fitted_tabpfn_model(args.fit_state, device="cuda")
    observed_context = getattr(model, "n_train_samples_", None)
    observed_estimators = getattr(model, "n_estimators_", None)
    if observed_context != args.context_rows:
        raise AssertionError(f"fitted context is {observed_context} rows")
    if observed_estimators != args.n_estimators:
        raise AssertionError(f"fitted state carries {observed_estimators} estimators")

    torch.cuda.reset_peak_memory_stats()
    model.predict_proba(np.ascontiguousarray(features[: args.warmup_rows]))
    torch.cuda.empty_cache()

    results: list[dict[str, Any]] = []

    def snapshot(status: str) -> dict[str, Any]:
        usable = [r for r in results if r["status"] == "ok"]
        fastest = max(usable, key=lambda r: r["rows_per_second"]) if usable else None
        return {
            "status": status,
            "gpu_name": properties.name,
            "gpu_total_mib": properties.total_memory / 1024**2,
            "context_rows": args.context_rows,
            "n_estimators": args.n_estimators,
            "n_features": int(features.shape[1]),
            "candidates": results,
            "recommended_microbatch": fastest["microbatch"] if fastest else None,
            "recommended_rows_per_second": (
                fastest["rows_per_second"] if fastest else None
            ),
        }

    # Written after every candidate: a large candidate needs a full microbatch of
    # real rows, which can run for minutes, and a run that is only observable at
    # the end is indistinguishable from a hang.
    atomic_write_json(args.out, snapshot("running"))
    for candidate in sorted(args.candidates):
        target_rows = max(candidate, args.min_rows_per_candidate)
        target_rows = min(target_rows, len(features))
        torch.cuda.reset_peak_memory_stats()
        position = 0
        rows_done = 0
        started = time.perf_counter()
        entry: dict[str, Any] = {"microbatch": candidate}
        try:
            while rows_done < target_rows:
                end = min(len(features), position + candidate)
                if end <= position:
                    position = 0
                    continue
                block = np.ascontiguousarray(features[position:end])
                model.predict_proba(block)
                rows_done += end - position
                position = end
                # One full microbatch is the minimum honest measurement; past
                # that, stop as soon as the time budget is spent.
                if time.perf_counter() - started > args.max_seconds_per_candidate:
                    break
            seconds = time.perf_counter() - started
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            peak_fraction = float((total_bytes - free_bytes) / total_bytes)
            entry.update(
                {
                    "status": "ok",
                    "rows": rows_done,
                    "seconds": seconds,
                    "rows_per_second": rows_done / seconds if seconds else None,
                    "peak_gpu_fraction": peak_fraction,
                    "torch_peak_reserved_mib": (
                        torch.cuda.max_memory_reserved() / 1024**2
                    ),
                }
            )
            if peak_fraction > args.gpu_soft_limit_fraction:
                entry["status"] = "over_soft_limit"
        except Exception as error:  # noqa: BLE001 - the OOM itself is the result
            entry.update(
                {
                    "status": "oom" if is_oom(error) else "failed",
                    "error": str(error)[:400],
                    "rows": rows_done,
                }
            )
        results.append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)
        atomic_write_json(args.out, snapshot("running"))
        torch.cuda.empty_cache()
        if entry["status"] in {"oom", "failed"}:
            break

    report = snapshot("completed")
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
