"""Pick one frozen TabPFN query microbatch for the whole seed 47--51 sweep.

The bundle's default query microbatch of 4096 was chosen for a 22 GB L4. This
probe measures what a smaller GPU actually sustains, using the real protocol
inputs: a tracked ladder manifest at the campaign's largest K (context is
``K x 500`` rows capped at 50,000, so that K is the worst case), a real fitted
TabPFN with the frozen 8 estimators, and a contiguous slice of the real
canonical holdout.

It answers two questions the bounded runbook validation cannot, because that
validation swaps in ``FakeTabPFNClassifier`` and never touches the GPU:

1. which microbatch sizes complete without CUDA OOM, and how fast each is;
2. whether lowering the microbatch perturbs the scores, by diffing every
   candidate's predictions against the 4096 baseline on identical rows.

Nothing here is scientific output. Scores are written only so the tolerance
comparison can be audited, under a NON_SCIENTIFIC path, and no checkpoint or
COMPLETE marker is produced.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from lead import PROC, load_m3_frame
from m5_building_curve_protocol import resolve_cell_indices
from run_m5_building_curve_tabpfn_cell import (
    _bounded_rows,
    default_model_path,
)
from run_m5_tabpfn_canonical_full_test import (
    DEFAULT_SITE_PREDICTIONS,
    create_real_model,
)
from run_m5_tabpfn_single_context_scaling import verify_fitted_context
from run_m5_tree_ensemble_matched_context import (
    build_features_keeping_index,
    feature_columns,
)

DEFAULT_AUDIT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "m5_building_count_v2_seed47_51"
    / "audit"
)
DEFAULT_OUT = PROC / "m5_building_curve" / "NON_SCIENTIFIC_MICROBATCH_CALIBRATION"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--building-seed", type=int, default=47)
    parser.add_argument("--building-budget", type=int, default=100)
    parser.add_argument("--features", type=int, choices=(17, 137), default=137)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=[4096, 2048, 1024, 512, 256],
        help="Probed in the given order; 4096 is the bundle default baseline.",
    )
    parser.add_argument(
        "--probe-holdout-rows",
        type=int,
        default=20_000,
        help="Contiguous real holdout rows per candidate (one checkpoint span).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.probe_holdout_rows <= 0 or min(args.candidates) <= 0:
        raise ValueError("probe rows and candidates must be positive")
    return args


def _predict(model: Any, matrix: np.ndarray, batch_size: int) -> np.ndarray:
    """Byte-for-byte the cell's own inference loop, so timings transfer."""
    output = np.empty(len(matrix), dtype="float32")
    for start in range(0, len(matrix), batch_size):
        end = min(len(matrix), start + batch_size)
        output[start:end] = np.asarray(
            model.predict_proba(matrix[start:end])[:, 1], dtype="float32"
        )
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("microbatch calibration requires CUDA")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.audit_root / f"building_ladder_seed{args.building_seed}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    frame = load_m3_frame(verbose=True)
    train_mask = frame["building_id"].mod(2).eq(0).to_numpy()
    resolved = resolve_cell_indices(
        frame.loc[train_mask],
        manifest,
        args.building_budget,
        require_role_class_coverage=False,
    )
    context_index = _bounded_rows(
        frame, resolved["available_rows"], None, seed=args.model_seed + 1
    )
    print(f"Context rows: {len(context_index)}", flush=True)

    print("Building context features", flush=True)
    selected_mask = frame["building_id"].isin(resolved["available_buildings"])
    train_features = build_features_keeping_index(frame.loc[selected_mask].copy())
    columns = feature_columns(args.features, list(train_features.columns))
    x_context = train_features.loc[context_index, columns].to_numpy(dtype="float32")
    y_context = frame.loc[context_index, "anomaly"].to_numpy(dtype="int64")
    del train_features
    gc.collect()

    scaler = StandardScaler()
    x_context = scaler.fit_transform(x_context).astype("float32", copy=False)

    print(
        f"Fitting real TabPFN: {len(context_index)} context rows, "
        f"{args.n_estimators} estimators",
        flush=True,
    )
    model = create_real_model(args.model_path, args.model_seed, args.n_estimators)
    fit_started = time.perf_counter()
    model.fit(x_context, y_context)
    fit_seconds = time.perf_counter() - fit_started
    verification = verify_fitted_context(
        model, len(context_index), requested_estimators=args.n_estimators
    )
    print(
        f"Fit finished in {fit_seconds:.1f}s; context verification="
        f"{verification['status']}",
        flush=True,
    )
    del x_context, y_context
    gc.collect()
    torch.cuda.empty_cache()
    fit_reserved_gib = torch.cuda.memory_reserved() / 1024**3

    with np.load(DEFAULT_SITE_PREDICTIONS) as canonical:
        holdout_index = np.asarray(canonical["validation_raw_index"], dtype="int64")
    total_holdout_rows = int(len(holdout_index))
    probe_index = holdout_index[: args.probe_holdout_rows]

    print("Building holdout features for the probe slice", flush=True)
    holdout_features = build_features_keeping_index(frame.loc[~train_mask].copy())
    block = scaler.transform(
        holdout_features.loc[probe_index, columns].to_numpy(dtype="float32")
    )
    del holdout_features, frame
    gc.collect()

    results: list[dict[str, Any]] = []
    scores: dict[int, np.ndarray] = {}
    for candidate in args.candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            score = _predict(model, block, candidate)
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            print(f"microbatch {candidate}: CUDA OOM", flush=True)
            results.append(
                {"microbatch": candidate, "status": "oom", "error": str(error)[:200]}
            )
            continue
        elapsed = time.perf_counter() - started
        peak_gib = torch.cuda.max_memory_reserved() / 1024**3
        scores[candidate] = score
        rows_per_second = len(block) / elapsed
        record = {
            "microbatch": candidate,
            "status": "ok",
            "seconds": round(elapsed, 2),
            "rows_per_second": round(rows_per_second, 1),
            "peak_reserved_gib": round(peak_gib, 2),
            "projected_hours_per_unit": round(
                total_holdout_rows / rows_per_second / 3600, 2
            ),
        }
        results.append(record)
        print(json.dumps(record), flush=True)

    baseline = next(
        (
            r["microbatch"]
            for r in results
            if r["status"] == "ok" and r["microbatch"] == 4096
        ),
        None,
    )
    if baseline is None:
        usable = [r["microbatch"] for r in results if r["status"] == "ok"]
        baseline = max(usable) if usable else None

    tolerance: list[dict[str, Any]] = []
    if baseline is not None:
        reference = scores[baseline]
        for candidate, score in sorted(scores.items(), reverse=True):
            diff = np.abs(score.astype("float64") - reference.astype("float64"))
            tolerance.append(
                {
                    "microbatch": candidate,
                    "vs_baseline": baseline,
                    "max_abs_diff": float(diff.max()),
                    "mean_abs_diff": float(diff.mean()),
                    "bitwise_identical": bool(np.array_equal(score, reference)),
                }
            )

    payload = {
        "_comment": (
            "Non-scientific throughput and tolerance probe. Selects one frozen "
            "query microbatch for all 20 TabPFN units; does not publish results."
        ),
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_gib": round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        ),
        "building_seed": args.building_seed,
        "building_budget": args.building_budget,
        "context_rows": int(len(context_index)),
        "n_estimators": args.n_estimators,
        "features": args.features,
        "fit_seconds": round(fit_seconds, 1),
        "fit_reserved_gib": round(fit_reserved_gib, 2),
        "probe_holdout_rows": int(len(block)),
        "total_holdout_rows": total_holdout_rows,
        "candidates": results,
        "tolerance_vs_baseline": tolerance,
    }
    out_path = args.out / "microbatch_calibration.json"
    with out_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
