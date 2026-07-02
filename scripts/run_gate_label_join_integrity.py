"""GATE: check M3 positional label integrity via anomaly run lengths."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import pandas as pd

from lead import PROC, ROOT, load_m3_frame, write_json_with_provenance


EXPECTED_ANOMALY_RATE = 0.065
ANOMALY_RATE_TOLERANCE = 0.002
DEFAULT_NULL_SEEDS = (42, 123, 999, 2025, 7, 31415, 2718, 8080)
LONG_RUN_THRESHOLDS = (2, 4, 8, 24, 72, 168)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "gate_label_join_integrity.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--null-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_NULL_SEEDS),
        help="Seeds for within-building anomaly shuffle nulls.",
    )
    return parser.parse_args()


def positive_run_lengths(values: np.ndarray) -> list[int]:
    if not len(values):
        return []
    arr = values.astype(bool, copy=False)
    padded = np.concatenate(([False], arr, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return (ends - starts).astype(int).tolist()


def summarize_run_lengths(lengths: list[int], *, n_positive: int) -> dict[str, Any]:
    arr = np.asarray(lengths, dtype=float)
    summary: dict[str, Any] = {
        "n_positive_rows": int(n_positive),
        "n_positive_runs": int(len(lengths)),
        "singleton_run_rate": None,
        "positive_rows_in_singleton_runs_rate": None,
        "mean_run_length": None,
        "median_run_length": None,
        "p90_run_length": None,
        "p95_run_length": None,
        "p99_run_length": None,
        "max_run_length": None,
        "positive_rows_in_long_runs_rate": {},
    }
    if not len(arr):
        return summary
    summary.update(
        {
            "singleton_run_rate": float(np.mean(arr == 1)),
            "positive_rows_in_singleton_runs_rate": float(
                np.sum(arr[arr == 1]) / n_positive
            )
            if n_positive
            else None,
            "mean_run_length": float(np.mean(arr)),
            "median_run_length": float(np.median(arr)),
            "p90_run_length": float(np.percentile(arr, 90)),
            "p95_run_length": float(np.percentile(arr, 95)),
            "p99_run_length": float(np.percentile(arr, 99)),
            "max_run_length": int(np.max(arr)),
            "positive_rows_in_long_runs_rate": {
                f"ge_{threshold}": float(np.sum(arr[arr >= threshold]) / n_positive)
                if n_positive
                else None
                for threshold in LONG_RUN_THRESHOLDS
            },
        }
    )
    return summary


def run_lengths_by_group(frame: pd.DataFrame, label_col: str) -> dict[str, Any]:
    lengths: list[int] = []
    per_group: list[dict[str, Any]] = []
    n_positive = int(frame[label_col].sum())
    for (building_id, meter), group in frame.groupby(
        ["building_id", "meter"], sort=False
    ):
        group_lengths = positive_run_lengths(group[label_col].to_numpy())
        lengths.extend(group_lengths)
        if group_lengths:
            per_group.append(
                {
                    "building_id": int(building_id),
                    "meter": int(meter),
                    "n_rows": int(len(group)),
                    "n_positive_rows": int(group[label_col].sum()),
                    "n_positive_runs": int(len(group_lengths)),
                    "max_run_length": int(max(group_lengths)),
                    "p95_run_length": float(np.percentile(group_lengths, 95)),
                }
            )
    top_groups = sorted(
        per_group,
        key=lambda item: (item["max_run_length"], item["n_positive_rows"]),
        reverse=True,
    )[:20]
    return {
        "overall": summarize_run_lengths(lengths, n_positive=n_positive),
        "top_groups_by_max_run": top_groups,
    }


def shuffled_within_building(frame: pd.DataFrame, *, seed: int) -> pd.Series:
    rng = np.random.RandomState(seed)
    shuffled = np.empty(len(frame), dtype=np.int8)
    for _, positions in frame.groupby("building_id", sort=False).indices.items():
        values = frame["anomaly"].to_numpy()[positions].copy()
        rng.shuffle(values)
        shuffled[positions] = values
    return pd.Series(shuffled, index=frame.index, name=f"anomaly_null_{seed}")


def aggregate_nulls(null_runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "singleton_run_rate",
        "positive_rows_in_singleton_runs_rate",
        "mean_run_length",
        "median_run_length",
        "p90_run_length",
        "p95_run_length",
        "p99_run_length",
        "max_run_length",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        values = [
            run["overall"][key] for run in null_runs if run["overall"][key] is not None
        ]
        out[key] = {
            "mean": float(mean(values)) if values else None,
            "std": float(pstdev(values))
            if len(values) > 1
            else 0.0
            if values
            else None,
            "min": float(min(values)) if values else None,
            "max": float(max(values)) if values else None,
        }
    out["positive_rows_in_long_runs_rate"] = {}
    for threshold in LONG_RUN_THRESHOLDS:
        name = f"ge_{threshold}"
        values = [
            run["overall"]["positive_rows_in_long_runs_rate"][name]
            for run in null_runs
            if run["overall"]["positive_rows_in_long_runs_rate"][name] is not None
        ]
        out["positive_rows_in_long_runs_rate"][name] = {
            "mean": float(mean(values)) if values else None,
            "std": float(pstdev(values))
            if len(values) > 1
            else 0.0
            if values
            else None,
            "min": float(min(values)) if values else None,
            "max": float(max(values)) if values else None,
        }
    return out


def gate_verdict(
    *,
    anomaly_rate: float,
    real: dict[str, Any],
    null_summary: dict[str, Any],
) -> dict[str, Any]:
    rate_ok = abs(anomaly_rate - EXPECTED_ANOMALY_RATE) <= ANOMALY_RATE_TOLERANCE
    real_ge_24 = real["overall"]["positive_rows_in_long_runs_rate"]["ge_24"]
    null_ge_24_max = null_summary["positive_rows_in_long_runs_rate"]["ge_24"]["max"]
    real_p95 = real["overall"]["p95_run_length"]
    null_p95_max = null_summary["p95_run_length"]["max"]
    long_runs_clear_null = (
        real_ge_24 is not None
        and null_ge_24_max is not None
        and real_ge_24 > null_ge_24_max
    )
    p95_clear_null = (
        real_p95 is not None and null_p95_max is not None and real_p95 > null_p95_max
    )
    status = (
        "pass" if rate_ok and long_runs_clear_null and p95_clear_null else "red_flag"
    )
    return {
        "status": status,
        "rate_ok": rate_ok,
        "expected_anomaly_rate": EXPECTED_ANOMALY_RATE,
        "anomaly_rate_tolerance": ANOMALY_RATE_TOLERANCE,
        "real_ge_24_rate": real_ge_24,
        "null_ge_24_rate_max": null_ge_24_max,
        "real_p95_run_length": real_p95,
        "null_p95_run_length_max": null_p95_max,
        "decision_rule": (
            "Pass requires overall anomaly rate near 6.50%, real positive rows "
            "in runs >=24 to exceed every within-building shuffle null, and real "
            "p95 run length to exceed every null p95. Otherwise stop as red_flag."
        ),
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    df = load_m3_frame(verbose=True)
    frame = df[["building_id", "meter", "timestamp", "anomaly"]].copy()
    frame = frame.sort_values(
        ["building_id", "meter", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    anomaly_rate = float(frame["anomaly"].mean())
    log(f"Anomaly rate: {anomaly_rate:.6f}")

    log("Computing real label run lengths")
    real = run_lengths_by_group(frame, "anomaly")

    null_runs = []
    for seed in args.null_seeds:
        log(f"Computing within-building shuffle null seed={seed}")
        null_col = f"anomaly_null_{seed}"
        frame[null_col] = shuffled_within_building(frame, seed=seed)
        null = run_lengths_by_group(frame, null_col)
        null["seed"] = int(seed)
        null_runs.append(null)
        frame = frame.drop(columns=[null_col])

    null_summary = aggregate_nulls(null_runs)
    verdict = gate_verdict(
        anomaly_rate=anomaly_rate,
        real=real,
        null_summary=null_summary,
    )
    results = {
        "experiment": "gate_label_join_integrity",
        "issue": 53,
        "scope": (
            "Check whether positional M3 labels form contiguous anomaly runs "
            "rather than random-looking scatter under a within-building shuffle null."
        ),
        "split_or_modeling": "none",
        "anomaly_rate": anomaly_rate,
        "n_rows": int(len(frame)),
        "n_buildings": int(frame["building_id"].nunique()),
        "n_building_meter_groups": int(
            frame[["building_id", "meter"]].drop_duplicates().shape[0]
        ),
        "real": real,
        "null": {
            "shuffle_unit": "within building_id",
            "seeds": list(args.null_seeds),
            "summary": null_summary,
            "runs": null_runs,
        },
        "verdict": verdict,
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }
    write_json_with_provenance(
        args.out,
        results,
        root=ROOT,
        provenance={
            "command": (
                f"uv run python scripts/run_gate_label_join_integrity.py --out {args.out}"
            ),
        },
    )
    log(f"Saved {args.out}")
    log(f"Gate verdict: {verdict['status']}")
    if verdict["status"] == "red_flag":
        raise RuntimeError("Label positional-join integrity gate failed")


if __name__ == "__main__":
    main()
