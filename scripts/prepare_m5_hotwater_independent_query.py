"""Freeze an independent, cluster-limited query for the hotwater factorial.

This is a sampling-only command.  It deliberately does not load a model or
score a query, so it can be run before any decision to permit a new fit.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from lead import ROOT
from lead.m5_context import stable_row_priority


FACTORIAL = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"
SOURCE_QUERY = (
    ROOT
    / "data"
    / "processed"
    / "m5_context_stories"
    / "queries"
    / "screening"
    / "queries.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root", type=Path, default=FACTORIAL / "independent_query"
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--rows-per-stratum", type=int, default=64)
    parser.add_argument("--max-rows-per-building", type=int, default=2)
    parser.add_argument("--max-rows-per-segment", type=int, default=2)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def segment_ids(frame: pd.DataFrame) -> pd.Series:
    anomaly = (
        frame.loc[frame["anomaly"] == 1]
        .sort_values(["building_id", "meter", "timestamp"], kind="stable")
        .copy()
    )
    gap = (
        anomaly.groupby(["building_id", "meter"], observed=True)["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )
    anomaly["segment_id"] = (gap.isna() | (gap > 1)).cumsum().astype("int64")
    result = pd.Series(index=frame.index, dtype="object")
    result.loc[anomaly.index] = anomaly["segment_id"].astype(str)
    result.loc[result.isna()] = "normal_" + frame.loc[
        result.isna(), "raw_index"
    ].astype(str)
    return result


def capped_sample(
    candidates: pd.DataFrame,
    *,
    rows: int,
    building_limit: int,
    segment_limit: int,
    seed: int,
) -> pd.DataFrame:
    ordered = candidates.assign(
        priority=stable_row_priority(candidates["raw_index"].to_numpy(), seed=seed)
    ).sort_values(["priority", "raw_index"], kind="stable")
    building_counts: Counter[int] = Counter()
    segment_counts: Counter[str] = Counter()
    chosen: list[int] = []
    for row in ordered.itertuples():
        if (
            building_counts[int(row.building_id)] >= building_limit
            or segment_counts[str(row.segment_id)] >= segment_limit
        ):
            continue
        chosen.append(int(row.Index))
        building_counts[int(row.building_id)] += 1
        segment_counts[str(row.segment_id)] += 1
        if len(chosen) == rows:
            break
    if len(chosen) != rows:
        raise RuntimeError(
            f"only selected {len(chosen)} of requested {rows} rows after caps"
        )
    return candidates.loc[chosen].copy()


def main() -> int:
    args = parse_args()
    if args.rows_per_stratum <= 0:
        raise SystemExit("--rows-per-stratum must be positive")
    with np.load(SOURCE_QUERY) as payload:
        old_raw = np.asarray(payload["raw_index"], dtype="int64")
        old_buildings = set(map(int, np.asarray(payload["building_id"], dtype="int16")))
    frame = pd.read_csv(
        ROOT / "data" / "raw" / "m3" / "train.csv",
        usecols=["building_id", "meter", "timestamp", "meter_reading"],
        parse_dates=["timestamp"],
    )
    labels = (
        pd.read_csv(ROOT / "data" / "raw" / "m3" / "bad_meter_readings.csv")
        .iloc[:, 0]
        .to_numpy(dtype="int8")
    )
    frame["anomaly"] = labels
    frame["raw_index"] = np.arange(len(frame), dtype="int64")
    frame = frame.loc[
        (frame["building_id"] % 2 == 1) & ~frame["building_id"].isin(old_buildings)
    ].copy()
    frame["segment_id"] = segment_ids(frame)
    strata = {
        "hw01_positive": (frame["meter"] == 3)
        & (frame["meter_reading"] <= 1.0)
        & (frame["anomaly"] == 1),
        "hw01_negative": (frame["meter"] == 3)
        & (frame["meter_reading"] <= 1.0)
        & (frame["anomaly"] == 0),
        "steam_positive": (frame["meter"] == 2) & (frame["anomaly"] == 1),
    }
    selections: list[pd.DataFrame] = []
    for offset, (name, mask) in enumerate(strata.items()):
        selected = capped_sample(
            frame.loc[mask],
            rows=args.rows_per_stratum,
            building_limit=args.max_rows_per_building,
            segment_limit=args.max_rows_per_segment,
            seed=args.seed + offset,
        )
        selected["stratum"] = name
        selections.append(selected)
    query = (
        pd.concat(selections, ignore_index=True)
        .sort_values(["stratum", "raw_index"], kind="stable")
        .reset_index(drop=True)
    )
    raw = query["raw_index"].to_numpy(dtype="int64")
    if (
        set(map(int, raw)) & set(map(int, old_raw))
        or set(map(int, query["building_id"])) & old_buildings
    ):
        raise AssertionError("independent query overlaps the screening query")
    args.out_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_root / "queries.npz",
        raw_index=raw,
        anomaly=query["anomaly"].to_numpy(dtype="int8"),
        meter=query["meter"].to_numpy(dtype="int8"),
        building_id=query["building_id"].to_numpy(dtype="int16"),
        meter_reading=query["meter_reading"].to_numpy(dtype="float32"),
        segment_id=query["segment_id"].astype(str).to_numpy(),
        stratum=query["stratum"].astype(str).to_numpy(),
    )
    counts = (
        query.groupby("stratum", observed=True)
        .agg(
            rows=("raw_index", "size"),
            buildings=("building_id", "nunique"),
            segments=("segment_id", "nunique"),
        )
        .reset_index()
        .to_dict("records")
    )
    atomic_json(
        args.out_root / "manifest.json",
        {
            "artifact_type": "m5_hotwater_factorial_independent_query",
            "sampling_declared_before_prediction_read": True,
            "source_query_rows_excluded": int(len(old_raw)),
            "source_query_buildings_excluded": len(old_buildings),
            "fit_rule": "no model fit or context change",
            "seed": args.seed,
            "rows_per_stratum": args.rows_per_stratum,
            "max_rows_per_building": args.max_rows_per_building,
            "max_rows_per_segment": args.max_rows_per_segment,
            "rows": int(len(query)),
            "raw_index": raw.tolist(),
            "strata": counts,
        },
    )
    query.to_csv(args.out_root / "query_audit.csv", index=False)
    print(
        f"wrote independent query: {len(query)} rows, {query['building_id'].nunique()} buildings"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
