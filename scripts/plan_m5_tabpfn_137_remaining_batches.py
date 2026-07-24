"""Plan the remaining 137-feature batches that complete the M3 figure holdout.

Sites 1, 2 and 3 are already scored at 137 features / n_estimators=8. To redraw
the four M3 figures with a 137-feature TabPFN line, every remaining row of the
same 50/50 building holdout must be scored too.

Batching is by building_id so a batch is always a whole number of buildings and
never splits one across shards, which keeps each batch independently mergeable
and keeps the per-site figures reconstructable. Within a batch the split into
head and tail lands on a 20,000-row checkpoint boundary, matching the runbook.

Emits a plan JSON; it does not export or launch anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lead import PROC

DEFAULT_CANONICAL = PROC / "m5_tabpfn_distributed_context100000_predictions.npz"
DONE_SITES = (1, 2, 3)
CHECKPOINT_ROWS = 20_000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--done-sites", type=int, nargs="*", default=list(DONE_SITES))
    parser.add_argument(
        "--target-batch-rows",
        type=int,
        default=1_200_000,
        help="approximate rows per batch; batches end on a building boundary",
    )
    parser.add_argument(
        "--out", type=Path, default=PROC / "m5_tabpfn_137_remaining_batch_plan.json"
    )
    parser.add_argument("--rows-per-second", type=float, default=330.0)
    args = parser.parse_args(argv)

    with np.load(args.canonical) as data:
        site_id = np.asarray(data["site_id"])
        building_id = np.asarray(data["building_id"])
        anomaly = np.asarray(data["anomaly"])
        raw_index = np.asarray(data["raw_index"])

    total_rows = len(site_id)
    remaining = ~np.isin(site_id, args.done_sites)
    positions = np.flatnonzero(remaining)
    if positions.size == 0:
        raise SystemExit("no remaining rows")

    # Canonical order is by building_id, so a contiguous run of positions is a
    # contiguous run of buildings. Cut only where the building changes.
    buildings = building_id[positions]
    change = np.flatnonzero(np.diff(buildings)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(positions)]))

    batches: list[dict[str, Any]] = []
    current_start = 0
    current_rows = 0
    for start, end in zip(starts, ends, strict=True):
        current_rows += end - start
        # Close the batch at this building boundary once it is big enough.
        if current_rows >= args.target_batch_rows:
            batches.append({"start": current_start, "end": end})
            current_start = end
            current_rows = 0
    if current_start < len(positions):
        trailing = len(positions) - current_start
        # A tiny trailing batch would pay a full deploy (uploads, reassemble,
        # installs) for a few minutes of inference, so fold it into the previous
        # batch instead. Building boundaries are preserved either way.
        if batches and trailing < args.target_batch_rows // 2:
            batches[-1]["end"] = len(positions)
        else:
            batches.append({"start": current_start, "end": len(positions)})

    plan: list[dict[str, Any]] = []
    for index, span in enumerate(batches):
        sel = positions[span["start"] : span["end"]]
        rows = len(sel)
        # Head/tail split on a checkpoint boundary; keep the halves balanced.
        boundary = int(round(rows / 2 / CHECKPOINT_ROWS)) * CHECKPOINT_ROWS
        boundary = max(CHECKPOINT_ROWS, min(boundary, rows - CHECKPOINT_ROWS))
        sites = sorted({int(v) for v in site_id[sel]})
        plan.append(
            {
                "batch": index,
                # Index range into the remaining-row array, recomputed the same
                # way by the exporter so a batch is reproducible exactly. The
                # canonical positions themselves are not contiguous, because the
                # already-scored sites are interleaved by building_id.
                "remaining_index_start": int(span["start"]),
                "remaining_index_end": int(span["end"]),
                "rows": rows,
                "anomalies": int(anomaly[sel].sum()),
                "prevalence": float(anomaly[sel].mean()),
                "sites": sites,
                "min_building_id": int(building_id[sel].min()),
                "max_building_id": int(building_id[sel].max()),
                "buildings": int(len(np.unique(building_id[sel]))),
                "canonical_position_start": int(sel[0]),
                "canonical_position_end": int(sel[-1]) + 1,
                "raw_index_first": int(raw_index[sel][0]),
                "raw_index_last": int(raw_index[sel][-1]),
                "head_rows": boundary,
                "tail_rows": rows - boundary,
                "head_chunks": -(-boundary // CHECKPOINT_ROWS),
                "tail_chunks": -(-(rows - boundary) // CHECKPOINT_ROWS),
            }
        )

    remaining_rows = int(remaining.sum())
    gpu_hours = remaining_rows / args.rows_per_second / 3600
    summary = {
        "canonical": str(args.canonical.resolve()),
        "total_holdout_rows": int(total_rows),
        "done_sites": args.done_sites,
        "done_rows": int(total_rows - remaining_rows),
        "remaining_rows": remaining_rows,
        "remaining_anomalies": int(anomaly[remaining].sum()),
        "remaining_sites": sorted({int(v) for v in np.unique(site_id[remaining])}),
        "checkpoint_rows": CHECKPOINT_ROWS,
        "batches": plan,
        "throughput_rows_per_second_per_gpu": args.rows_per_second,
        "estimate_single_gpu_hours": gpu_hours,
        "estimate_two_gpu_wall_hours": gpu_hours / 2,
    }
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"remaining {remaining_rows:,} rows "
        f"({summary['remaining_anomalies']:,} anomalies) "
        f"across sites {summary['remaining_sites']}"
    )
    print(f"{len(plan)} batches, target {args.target_batch_rows:,} rows each")
    for entry in plan:
        print(
            f"  batch {entry['batch']}: {entry['rows']:>9,} rows "
            f"| buildings {entry['min_building_id']}-{entry['max_building_id']} "
            f"({entry['buildings']}) | sites {entry['sites']} "
            f"| head {entry['head_rows']:,}/{entry['head_chunks']}ck "
            f"tail {entry['tail_rows']:,}/{entry['tail_chunks']}ck"
        )
    print(
        f"\nestimate: {gpu_hours:.1f} GPU-hours, "
        f"{gpu_hours / 2:.1f} h wall on 2 A100 at {args.rows_per_second:.0f} rows/s"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
