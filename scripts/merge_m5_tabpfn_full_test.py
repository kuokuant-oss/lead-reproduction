"""Merge one TabPFN line's shards into a full-holdout prediction artifact.

Both n_estimators=8 lines were produced in two waves: sites 1/2/3 as per-site
shards, then the remaining thirteen sites as building_id batches. Both waves
scored rows drawn from the same canonical 50/50 building holdout, so the merge is
a pure regrouping -- but only if the union is exactly the holdout, with no row
scored twice and none missing.

That is what this checks before writing anything: every canonical row appears
exactly once, labels/sites/buildings agree row-for-row with the canonical
artifact, and no score is non-finite. Output is aligned to canonical order so it
drops straight into the M3 figure renderers.

``--line`` picks which set of shard roots to gather. The two lines share the
batch geometry exactly, so merging them the same way is what keeps the two
plotted curves row-for-row comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lead import PROC

DEFAULT_CANONICAL = PROC / "m5_tabpfn_distributed_context100000_predictions.npz"

# The 17-feature sweep named its per-site roots before a second feature width
# existed, so only the 137-feature line carries an f137 marker in the site roots.
LINE_ROOTS = {
    "17": [
        *[PROC / f"m5_tabpfn_site{s}_context100000_n8" for s in (1, 2, 3)],
        *[PROC / f"m5_tabpfn_f17_batch{i}_context100000_n8" for i in range(6)],
    ],
    "137": [
        *[PROC / f"m5_tabpfn_site{s}_f137_context100000_n8" for s in (1, 2, 3)],
        *[PROC / f"m5_tabpfn_f137_batch{i}_context100000_n8" for i in range(6)],
    ],
}
LINE_OUT = {
    "17": PROC / "m5_tabpfn_17_full_test_n8_predictions.npz",
    "137": PROC / "m5_tabpfn_137_full_test_n8_predictions.npz",
}


def collect(roots: list[Path]) -> dict[str, np.ndarray]:
    raw_index: list[np.ndarray] = []
    score: list[np.ndarray] = []
    anomaly: list[np.ndarray] = []
    site_id: list[np.ndarray] = []
    building_id: list[np.ndarray] = []
    for root in roots:
        for shard in ("head", "tail"):
            chunks = sorted((root / f"{shard}-results" / "chunks").glob("rows_*.npz"))
            if not chunks:
                raise FileNotFoundError(f"no checkpoints under {root}/{shard}-results")
            for path in chunks:
                with np.load(path) as payload:
                    raw_index.append(np.asarray(payload["raw_index"], dtype="int64"))
                    score.append(np.asarray(payload["score"], dtype="float32"))
                    anomaly.append(np.asarray(payload["anomaly"], dtype="int8"))
                    site_id.append(np.asarray(payload["site_id"], dtype="int8"))
                    building_id.append(
                        np.asarray(payload["building_id"], dtype="int16")
                    )
    return {
        "raw_index": np.concatenate(raw_index),
        "score": np.concatenate(score),
        "anomaly": np.concatenate(anomaly),
        "site_id": np.concatenate(site_id),
        "building_id": np.concatenate(building_id),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line", choices=sorted(LINE_ROOTS), default="137")
    parser.add_argument("--roots", type=Path, nargs="*", default=None)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    roots = args.roots if args.roots else LINE_ROOTS[args.line]
    out = args.out or LINE_OUT[args.line]

    observed = collect(list(roots))
    with np.load(args.canonical) as data:
        canonical = {
            "raw_index": np.asarray(data["raw_index"], dtype="int64"),
            "anomaly": np.asarray(data["anomaly"], dtype="int8"),
            "site_id": np.asarray(data["site_id"], dtype="int8"),
            "building_id": np.asarray(data["building_id"], dtype="int16"),
        }

    rows = len(canonical["raw_index"])
    if len(observed["raw_index"]) != rows:
        raise AssertionError(
            f"merged {len(observed['raw_index'])} rows, holdout has {rows}"
        )
    if len(np.unique(observed["raw_index"])) != rows:
        raise AssertionError("merged rows contain duplicates")

    order = np.argsort(observed["raw_index"])
    canonical_order = np.argsort(canonical["raw_index"])
    inverse = np.empty(rows, dtype="int64")
    inverse[canonical_order] = np.arange(rows)
    # Reindex the merged rows into canonical order.
    aligned = {key: value[order][inverse] for key, value in observed.items()}

    if not np.array_equal(aligned["raw_index"], canonical["raw_index"]):
        raise AssertionError("merged row set differs from the canonical holdout")
    for column in ("anomaly", "site_id", "building_id"):
        if not np.array_equal(aligned[column], canonical[column]):
            raise AssertionError(f"merged {column} disagrees with the canonical")
    if not np.isfinite(aligned["score"]).all():
        raise AssertionError("merged scores contain non-finite values")

    np.savez_compressed(
        out,
        raw_index=aligned["raw_index"],
        anomaly=aligned["anomaly"],
        tabpfn=aligned["score"].astype("float32"),
        site_id=aligned["site_id"],
        building_id=aligned["building_id"],
    )
    summary = {
        "line": args.line,
        "rows": int(rows),
        "anomalies": int(aligned["anomaly"].sum()),
        "distinct_scores": int(len(np.unique(aligned["score"]))),
        "score_min": float(aligned["score"].min()),
        "score_max": float(aligned["score"].max()),
        "roots": [str(r) for r in roots],
        "out": str(out.resolve()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
