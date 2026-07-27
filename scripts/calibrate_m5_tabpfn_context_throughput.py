"""Measure TabPFN inference throughput against context size. Step 0 of the curve.

The only throughput number the project has is A100 @ 100k context: 330 rows/s at
137 features, 430 at 17. Every cost estimate for the 5k-50k contexts is an
extrapolation from that single point, and the shape of the extrapolation is not
obvious:

    t_per_row = a + b * context_rows

``b * context`` is the query-to-context attention; ``a`` is the part that does not
care how long the context is. At 100k, ``b * context`` dominates and throughput
looks inversely proportional to context. At 5k it does not -- ``a`` takes over and
throughput plateaus. Assuming "20x smaller context, 20x faster" would understate
the rented hours by a large factor, so the two coefficients get measured before
any GPU time is paid for.

Self-contained on purpose: torch and tabpfn are imported inside the worker so the
file can be scp'd to a bare rented box and run there. Run it on every machine you
intend to use, then compare -- that comparison is also how the local 4070 gets
judged fit or unfit for the 5k pass.

    uv run python scripts/calibrate_m5_tabpfn_context_throughput.py \
        --features data/processed/m5_tabpfn_137_distributed_context100000/head/features.float32.npy \
        --metadata data/processed/m5_tabpfn_137_distributed_context100000/head/metadata.npz \
        --out data/processed/m5_tabpfn_context_throughput_local4070.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CONTEXTS = (5_000, 10_000, 20_000, 50_000, 100_000)
HOLDOUT_ROWS = 10_137_155


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(".tabpfn-cache/tabpfn-v3-classifier-v3_default.ckpt"),
    )
    parser.add_argument(
        "--contexts", type=int, nargs="*", default=list(DEFAULT_CONTEXTS)
    )
    parser.add_argument(
        "--feature-widths",
        type=int,
        nargs="*",
        default=[17, 137],
        help="17 uses the first 17 columns, which are the baseline features",
    )
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--query-rows", type=int, default=20_000)
    parser.add_argument("--microbatch", type=int, default=20_000)
    parser.add_argument("--warmup-rows", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def build_context(
    matrix: np.ndarray, labels: np.ndarray, rows: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """A balanced context of the requested size, for timing only.

    This deliberately does *not* reproduce the canonical context: throughput
    depends on the shape of the problem, not on which rows were chosen, and
    requiring the real context would mean shipping every fit state to every
    machine being benchmarked. The scores produced here are thrown away.
    """
    per_class = rows // 2
    rng = np.random.RandomState(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) < per_class or len(negative) < per_class:
        raise SystemExit(
            f"need {per_class:,} rows per class for a {rows:,} context, have "
            f"positive={len(positive):,} negative={len(negative):,}. "
            "Point --features at a larger shard."
        )
    take = np.empty(rows, dtype="int64")
    take[0::2] = rng.choice(positive, per_class, replace=False)
    take[1::2] = rng.choice(negative, per_class, replace=False)
    return np.asarray(matrix[take], dtype="float32"), labels[take]


def solve_affine(contexts: list[int], seconds_per_row: list[float]):
    """Least-squares fit of t = a + b*C, plus the R^2 of that fit."""
    if len(contexts) < 2:
        return None, None, None
    x = np.asarray(contexts, dtype="float64")
    y = np.asarray(seconds_per_row, dtype="float64")
    b, a = np.polyfit(x, y, 1)
    predicted = a + b * x
    residual = float(((y - predicted) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    return float(a), float(b), r2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from tabpfn import TabPFNClassifier

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; this benchmark is meaningless on CPU")
    device = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device: {device} ({total_vram:.1f} GiB)")

    matrix = np.load(args.features, mmap_mode="r")
    with np.load(args.metadata) as payload:
        labels = np.asarray(payload["anomaly"], dtype="int8")
    if len(matrix) != len(labels):
        raise SystemExit("feature and metadata row counts differ")
    widths = sorted(set(args.feature_widths))
    if max(widths) > matrix.shape[1]:
        raise SystemExit(
            f"--feature-widths asks for {max(widths)} columns, matrix has "
            f"{matrix.shape[1]}"
        )

    rng = np.random.RandomState(args.seed + 1)
    query_rows = min(args.query_rows, len(matrix))
    query_take = rng.choice(len(matrix), query_rows, replace=False)
    query_all = np.asarray(matrix[np.sort(query_take)], dtype="float32")
    print(f"query block: {query_rows:,} rows, microbatch {args.microbatch:,}")

    results: list[dict[str, Any]] = []
    for width in widths:
        for context in sorted(set(args.contexts)):
            context_x, context_y = build_context(
                matrix[:, :width], labels, context, args.seed
            )
            model = TabPFNClassifier(
                device="cuda",
                model_path=str(args.model_path),
                random_state=args.seed,
                n_estimators=args.n_estimators,
                auto_scale_n_estimators=False,
                fit_mode="fit_preprocessors",
                memory_saving_mode="auto",
            )
            fit_started = time.perf_counter()
            model.fit(context_x, context_y)
            fit_seconds = time.perf_counter() - fit_started

            query = query_all[:, :width]
            # Warm up outside the timed region: the first call pays for kernel
            # autotuning and allocator growth, which would otherwise be charged
            # to the smallest context and bend the fitted line.
            model.predict_proba(query[: min(args.warmup_rows, len(query))])
            torch.cuda.synchronize()

            started = time.perf_counter()
            done = 0
            oom = False
            batch = args.microbatch
            try:
                while done < len(query):
                    end = min(len(query), done + batch)
                    model.predict_proba(query[done:end])
                    done = end
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                oom = True
            elapsed = time.perf_counter() - started

            peak = torch.cuda.max_memory_allocated() / 1024**3
            row = {
                "feature_width": width,
                "context_rows": context,
                "n_estimators": args.n_estimators,
                "fit_seconds": round(fit_seconds, 2),
                "query_rows": int(done),
                "elapsed_seconds": round(elapsed, 3),
                "rows_per_second": round(done / elapsed, 1) if elapsed else None,
                "seconds_per_row": elapsed / done if done else None,
                "peak_gpu_gib": round(peak, 2),
                "microbatch": batch,
                "oom": oom,
            }
            results.append(row)
            status = "OOM" if oom else f"{row['rows_per_second']:>8,.1f} rows/s"
            print(
                f"  f{width:<4} ctx {context:>7,}  {status}  "
                f"peak {peak:5.2f} GiB  fit {fit_seconds:5.1f}s"
            )
            del model, context_x, context_y
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    print("\n=== projected full-holdout cost (10,137,155 rows) ===")
    projection: dict[str, Any] = {}
    for width in widths:
        usable = [r for r in results if r["feature_width"] == width and not r["oom"]]
        a, b, r2 = solve_affine(
            [r["context_rows"] for r in usable],
            [r["seconds_per_row"] for r in usable],
        )
        entry: dict[str, Any] = {"fixed_seconds_per_row": a, "per_context_row": b}
        if r2 is not None:
            entry["r_squared"] = round(r2, 5)
            share = (b * 100_000) / (a + b * 100_000) if a is not None else None
            entry["context_bound_share_at_100k"] = (
                round(share, 3) if share is not None else None
            )
            print(
                f"f{width}: t_per_row = {a:.3e} + {b:.3e} * C   "
                f"R^2={r2:.5f}   context-bound at 100k: {share:.0%}"
            )
        hours = {}
        for r in usable:
            hours[str(r["context_rows"])] = round(
                HOLDOUT_ROWS * r["seconds_per_row"] / 3600, 2
            )
            print(
                f"    ctx {r['context_rows']:>7,}: "
                f"{hours[str(r['context_rows'])]:>6.2f} GPU-hours"
            )
        entry["gpu_hours_by_context"] = hours
        projection[f"f{width}"] = entry

    payload = {
        "schema_version": 1,
        "experiment": "m5_tabpfn_context_throughput_calibration",
        "device": device,
        "total_vram_gib": round(total_vram, 2),
        "host": platform.node(),
        "torch": torch.__version__,
        "holdout_rows": HOLDOUT_ROWS,
        "query_rows": query_rows,
        "microbatch": args.microbatch,
        "n_estimators": args.n_estimators,
        "features_source": str(args.features),
        "measurements": results,
        "projection": projection,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, args.out)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
