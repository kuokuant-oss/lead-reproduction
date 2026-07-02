"""INV-3: inspect unique underlying rows in Phase D label-scarcity support sets."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from lead import (
    DOWNSAMPLE_SEEDS,
    MODEL_SEEDS,
    PROC,
    ROOT,
    load_m3_frame,
    write_json_with_provenance,
)
from run_m5_phaseD_foundation_vs_gbdt import (
    IN_DOMAIN_SPLIT,
    VALUE_CHANGE_REGIME,
    balanced_subsample_indices,
    build_split_table,
)


NON_NEGLIGIBLE_DUPLICATE_RATE = 0.01


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "inv3_scarcity_unique_support.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--scarcity-sizes",
        type=int,
        nargs="+",
        default=[200, 500, 1_000, 2_000, 5_000, 10_000],
        help="Requested balanced support-set sizes to inspect.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(MODEL_SEEDS),
        help="Phase D support sampling seeds.",
    )
    parser.add_argument(
        "--duplicate-rate-threshold",
        type=float,
        default=NON_NEGLIGIBLE_DUPLICATE_RATE,
        help=(
            "Duplicate-rate threshold above which support-size semantics are "
            "treated as materially non-literal and an additive unique-row path "
            "should be used in a follow-up run."
        ),
    )
    return parser.parse_args()


def class_counts(y, indices: np.ndarray) -> dict[str, int]:
    values = y.loc[indices].to_numpy()
    return {
        "negative": int((values == 0).sum()),
        "positive": int((values == 1).sum()),
    }


def support_stats(
    *,
    ds_idx_full: np.ndarray,
    y_train_full,
    support_size: int,
    seed: int,
) -> dict[str, Any]:
    fit_idx = balanced_subsample_indices(
        ds_idx_full,
        y_train_full,
        support_size,
        seed,
    )
    unique_idx = np.unique(fit_idx)
    requested = int(support_size)
    sampled = int(len(fit_idx))
    unique_rows = int(len(unique_idx))
    duplicate_rows = int(sampled - unique_rows)
    return {
        "support_size_requested": requested,
        "seed": int(seed),
        "sampled_rows": sampled,
        "unique_underlying_rows": unique_rows,
        "duplicate_row_uses": duplicate_rows,
        "duplicate_rate": float(duplicate_rows / sampled) if sampled else 0.0,
        "requested_minus_unique_rows": int(requested - unique_rows),
        "class_counts_sampled": class_counts(y_train_full, fit_idx),
        "class_counts_unique": class_counts(y_train_full, unique_idx),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    duplicate_rates = [r["duplicate_rate"] for r in records]
    unique_rows = [r["unique_underlying_rows"] for r in records]
    sampled_rows = [r["sampled_rows"] for r in records]
    return {
        "n_runs": int(len(records)),
        "sampled_rows_min": int(min(sampled_rows)),
        "sampled_rows_max": int(max(sampled_rows)),
        "unique_rows_min": int(min(unique_rows)),
        "unique_rows_max": int(max(unique_rows)),
        "duplicate_rate_mean": float(mean(duplicate_rates)),
        "duplicate_rate_std": float(pstdev(duplicate_rates))
        if len(duplicate_rates) > 1
        else 0.0,
        "duplicate_rate_max": float(max(duplicate_rates)),
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    df = load_m3_frame(verbose=True)
    mask_8020 = (df["building_id"] % 5 == 4).to_numpy()
    table = build_split_table(df, mask_8020, split_label=IN_DOMAIN_SPLIT)
    ds_idx_full = table["ds_idx_full"]
    y_train_full = table["y_train_full"]

    by_size = []
    material_cells: list[dict[str, Any]] = []
    for support_size in args.scarcity_sizes:
        records = [
            support_stats(
                ds_idx_full=ds_idx_full,
                y_train_full=y_train_full,
                support_size=support_size,
                seed=seed,
            )
            for seed in args.seeds
        ]
        size_summary = summarize(records)
        log(
            f"support={support_size:>6} "
            f"unique={size_summary['unique_rows_min']}-"
            f"{size_summary['unique_rows_max']} "
            f"max_dup_rate={size_summary['duplicate_rate_max']:.6f}"
        )
        for record in records:
            if record["duplicate_rate"] > args.duplicate_rate_threshold:
                material_cells.append(record)
        by_size.append(
            {
                "support_size": int(support_size),
                "summary": size_summary,
                "runs": records,
            }
        )

    verdict = "material" if material_cells else "immaterial"
    results = {
        "experiment": "inv3_scarcity_unique_support",
        "issue": 51,
        "scope": (
            "Instrument Phase D label_scarcity support sampling to compare "
            "requested support size with unique underlying train rows."
        ),
        "value_change_regime": VALUE_CHANGE_REGIME,
        "split": table["split"],
        "budgets": {
            "scarcity_sizes": list(args.scarcity_sizes),
            "seeds": list(args.seeds),
            "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
            "ds_idx_full_rows": int(len(ds_idx_full)),
            "ds_idx_full_unique_rows": int(len(np.unique(ds_idx_full))),
            "ds_idx_full_duplicate_rate": float(
                1 - (len(np.unique(ds_idx_full)) / len(ds_idx_full))
            ),
        },
        "materiality": {
            "duplicate_rate_threshold": float(args.duplicate_rate_threshold),
            "verdict": verdict,
            "material_cells": material_cells,
            "unique_row_rerun_required": bool(material_cells),
            "decision_rule": (
                "If any sampled support set has duplicate_rate greater than the "
                "threshold, support-size semantics are treated as materially "
                "non-literal and an additive unique-row scarcity path should be "
                "run. Otherwise, record the measured duplicate rates and keep "
                "the existing Phase D path unchanged."
            ),
        },
        "support_sets": by_size,
        "interpretation": {
            "support_size_semantics": "literal_enough"
            if verdict == "immaterial"
            else "non_literal_requires_additive_unique_row_path",
            "m3_default_change": "none",
            "phaseD_default_change": "none",
        },
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }
    write_json_with_provenance(
        args.out,
        results,
        root=ROOT,
        provenance={
            "command": (
                "uv run python scripts/run_inv3_scarcity_unique_support.py "
                f"--out {args.out}"
            ),
        },
    )
    log(f"Saved {args.out}")
    log(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
