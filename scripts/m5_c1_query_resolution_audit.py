"""352-row screening query resolution audit.

Reports, for each required stratum, the row / building / segment counts, the
number of valid positive-negative pairs available for each planned comparison,
the AUC resolution those pairs imply, and building / segment concentration.

No model is scored. Nothing is fit. The frozen 192-row query is never touched.
A continuous margin is a feasibility readout only and cannot substitute for
inadequate pair resolution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

METER_NAMES = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}
# Pre-declared: adequate resolution needs enough pairs that one pair cannot move
# AUC by more than 1%, i.e. at least 100 valid pairs, and no single building may
# carry more than half of a stratum.
MIN_VALID_PAIRS = 100
MAX_BUILDING_SHARE = 0.50


def load_query(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as data:
        return pd.DataFrame(
            {
                "raw_index": data["raw_index"].astype("int64"),
                "anomaly": data["anomaly"].astype("int8"),
                "meter": data["meter"].astype("int8"),
                "site_id": data["site_id"].astype("int16"),
                "building_id": data["building_id"].astype("int32"),
            }
        )


def attach_segments(
    query: pd.DataFrame, movement: Path, segments: Path
) -> pd.DataFrame:
    """Map each query row to its frozen anomaly segment, if it lies inside one."""
    cols = ["raw_index", "building_id", "meter_name", "timestamp"]
    mv = pq.read_table(movement, columns=cols).to_pandas()
    mv = mv[mv["raw_index"].isin(query["raw_index"])]
    query = query.merge(mv, on="raw_index", how="left", suffixes=("", "_mv"))

    seg = pq.read_table(
        segments,
        columns=[
            "segment_id",
            "building_id",
            "meter_name",
            "start_timestamp",
            "end_timestamp",
        ],
    ).to_pandas()
    seg_ids: list[object] = []
    for _, row in query.iterrows():
        if pd.isna(row.get("timestamp")):
            seg_ids.append(None)
            continue
        hit = seg[
            (seg["building_id"] == row["building_id"])
            & (seg["meter_name"] == row["meter_name"])
            & (seg["start_timestamp"] <= row["timestamp"])
            & (seg["end_timestamp"] >= row["timestamp"])
        ]
        seg_ids.append(int(hit["segment_id"].iloc[0]) if len(hit) else None)
    query["segment_id"] = seg_ids
    return query


def stratum_stats(frame: pd.DataFrame) -> dict:
    n = len(frame)
    by_building = frame["building_id"].value_counts()
    seg = frame["segment_id"].dropna()
    by_segment = seg.value_counts()
    return {
        "rows": int(n),
        "positive": int(frame["anomaly"].sum()),
        "negative": int(n - frame["anomaly"].sum()),
        "buildings": int(frame["building_id"].nunique()),
        "segments": int(seg.nunique()),
        "rows_inside_a_segment": int(len(seg)),
        "top_building_share": float(by_building.iloc[0] / n) if n else float("nan"),
        "top_segment_share": float(by_segment.iloc[0] / n) if len(by_segment) else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    proc = args.data_root / "processed"

    query = load_query(
        proc / "m5_context_stories" / "queries" / "screening" / "queries.npz"
    )
    mech = proc / "m5_context_mechanism_137"
    query = attach_segments(
        query,
        mech / "m5_137_row_score_rank_movement_tabpfn.parquet",
        mech / "m5_137_anomaly_segments.parquet",
    )
    query["stratum"] = [
        f"{METER_NAMES[int(m)]}_{'positive' if a else 'negative'}"
        for m, a in zip(query["meter"], query["anomaly"], strict=True)
    ]

    required = [
        "chilledwater_positive",
        "chilledwater_negative",
        "electricity_negative",
        "steam_negative",
        "hotwater_negative",
    ]
    strata = {s: stratum_stats(g) for s, g in query.groupby("stratum")}

    comparisons = []
    pos = strata.get("chilledwater_positive", {})
    for neg in [
        "hotwater_negative",
        "electricity_negative",
        "chilledwater_negative",
        "steam_negative",
    ]:
        n_pos = pos.get("rows", 0)
        n_neg = strata.get(neg, {}).get("rows", 0)
        pairs = n_pos * n_neg
        comparisons.append(
            {
                "comparison": f"chilledwater_positive_vs_{neg}",
                "positive_rows": n_pos,
                "negative_rows": n_neg,
                "valid_pairs": pairs,
                "auc_resolution": (1.0 / pairs) if pairs else None,
                "adequate_pairs": pairs >= MIN_VALID_PAIRS,
                "positive_top_building_share": pos.get("top_building_share"),
                "negative_top_building_share": strata.get(neg, {}).get(
                    "top_building_share"
                ),
                "building_concentration_ok": (
                    (pos.get("top_building_share", 1) <= MAX_BUILDING_SHARE)
                    and (
                        strata.get(neg, {}).get("top_building_share", 1)
                        <= MAX_BUILDING_SHARE
                    )
                ),
            }
        )

    missing = [s for s in required if s not in strata]
    for c in comparisons:
        c["adequate_resolution"] = bool(
            c["adequate_pairs"] and c["building_concentration_ok"]
        )

    payload = {
        "schema": "m5_c1_query_resolution_audit_v1",
        "query": "original 352-row screening query",
        "query_rows": int(len(query)),
        "no_model_scored": True,
        "frozen_192_row_query_touched": False,
        "thresholds": {
            "min_valid_pairs": MIN_VALID_PAIRS,
            "max_building_share": MAX_BUILDING_SHARE,
        },
        "missing_required_strata": missing,
        "strata": strata,
        "comparisons": comparisons,
        "note": (
            "valid_pairs is the count of positive-negative row pairs available to a "
            "pairwise AUC; auc_resolution is the smallest AUC increment those pairs "
            "can express. A continuous score margin is a feasibility readout only."
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root / "c1_query_resolution_audit.json"
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {out}\n")
    print(
        f"{'stratum':26s} {'rows':>5s} {'bldg':>5s} {'seg':>4s} {'in_seg':>7s} {'top_bldg':>9s}"
    )
    for s in sorted(strata):
        v = strata[s]
        print(
            f"{s:26s} {v['rows']:5d} {v['buildings']:5d} {v['segments']:4d} "
            f"{v['rows_inside_a_segment']:7d} {v['top_building_share']:9.3f}"
        )
    print()
    for c in comparisons:
        print(
            f"{c['comparison']:48s} pairs={c['valid_pairs']:6d} "
            f"res={c['auc_resolution']:.2e} adequate={c['adequate_resolution']}"
        )
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
