"""C1 artifact census.

Enumerates every input C1 is allowed to use and reports path, size, SHA256,
rows, columns, and context / meter / label / building / segment coverage.
Read-only. If a required artifact is missing the census reports it and exits
non-zero; a missing artifact must never be used to justify a fit or inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

METER_NAMES = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}
CONTEXTS = (5000, 10000, 20000, 50000, 100000)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def npz_entry(path: Path, role: str, context: int | None) -> dict:
    entry: dict[str, object] = {
        "role": role,
        "path": str(path),
        "context_rows": context,
        "present": path.exists(),
    }
    if not path.exists():
        return entry
    entry["bytes"] = path.stat().st_size
    entry["sha256"] = sha256_file(path)
    with np.load(path, allow_pickle=False) as data:
        entry["arrays"] = sorted(data.files)
        for key in ("score", "scores", "raw_index", "anomaly", "meter"):
            if key in data.files:
                entry.setdefault("shapes", {})[key] = list(data[key].shape)
    return entry


def parquet_entry(path: Path, role: str, coverage_cols: list[str]) -> dict:
    entry: dict[str, object] = {
        "role": role,
        "path": str(path),
        "present": path.exists(),
    }
    if not path.exists():
        return entry
    entry["bytes"] = path.stat().st_size
    entry["sha256"] = sha256_file(path)
    pf = pq.ParquetFile(path)
    entry["rows"] = int(pf.metadata.num_rows)
    entry["columns"] = list(pf.schema_arrow.names)
    entry["column_count"] = len(entry["columns"])
    cols = [c for c in coverage_cols if c in entry["columns"]]
    if cols:
        frame = pq.read_table(path, columns=cols).to_pandas()
        cov: dict[str, object] = {}
        if "meter" in frame:
            cov["meter_rows"] = {
                METER_NAMES.get(int(k), str(k)): int(v)
                for k, v in frame["meter"].value_counts().items()
            }
        if "meter_name" in frame:
            cov["meter_rows"] = {
                str(k): int(v) for k, v in frame["meter_name"].value_counts().items()
            }
        if "anomaly" in frame:
            cov["positive_rows"] = int(frame["anomaly"].sum())
            cov["negative_rows"] = int(len(frame) - frame["anomaly"].sum())
        if "building_id" in frame:
            cov["buildings"] = int(frame["building_id"].nunique())
        if "segment_id" in frame:
            cov["segments"] = int(frame["segment_id"].nunique())
        entry["coverage"] = cov
    entry["context_columns_present"] = [
        c for c in entry["columns"] if any(str(x) in c for x in CONTEXTS)
    ]
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    proc = args.data_root / "processed"
    entries: list[dict] = []

    for ctx in CONTEXTS:
        name = (
            "m5_tabpfn_137_full_test_n8_predictions.npz"
            if ctx == 100000
            else f"m5_tabpfn_137_full_test_context{ctx}_n8_predictions.npz"
        )
        entries.append(npz_entry(proc / name, "tabpfn_full_holdout_predictions", ctx))
    for ctx in CONTEXTS:
        entries.append(
            npz_entry(
                proc / f"m5_tree_ensemble_f137_context{ctx}_predictions.npz",
                "matched_row_tree_predictions",
                ctx,
            )
        )

    mech = proc / "m5_context_mechanism_137"
    entries.append(
        parquet_entry(
            mech / "m5_137_row_score_rank_movement_tabpfn.parquet",
            "row_level_score_and_rank_movement_tabpfn",
            ["meter", "anomaly", "building_id"],
        )
    )
    entries.append(
        parquet_entry(
            mech / "m5_137_row_score_rank_movement_trees.parquet",
            "row_level_score_and_rank_movement_trees",
            ["meter", "anomaly", "building_id"],
        )
    )
    entries.append(
        parquet_entry(
            mech / "m5_137_anomaly_segments.parquet",
            "frozen_segment_definitions",
            ["meter_name", "building_id", "segment_id"],
        )
    )
    entries.append(
        parquet_entry(
            mech / "m5_137_anomaly_segment_phases.parquet",
            "frozen_segment_phases",
            ["segment_id"],
        )
    )

    train = args.data_root / "raw" / "m3" / "train.csv"
    entries.append(
        {
            "role": "row_level_labels_and_metadata",
            "path": str(train),
            "present": train.exists(),
            "bytes": train.stat().st_size if train.exists() else None,
            "note": "indirect metadata reader; building_id and meter only",
        }
    )

    query = proc / "m5_context_stories" / "queries" / "screening" / "queries.npz"
    q: dict[str, object] = {
        "role": "original_352_row_screening_query",
        "path": str(query),
        "present": query.exists(),
    }
    if query.exists():
        q["bytes"] = query.stat().st_size
        q["sha256"] = sha256_file(query)
        with np.load(query, allow_pickle=False) as data:
            q["arrays"] = sorted(data.files)
            q["rows"] = int(data["raw_index"].shape[0])
            meter = data["meter"]
            anomaly = data["anomaly"]
            q["coverage"] = {
                "buildings": int(np.unique(data["building_id"]).size),
                "sites": int(np.unique(data["site_id"]).size),
                "strata": {
                    f"{METER_NAMES[int(m)]}_{'positive' if a else 'negative'}": int(
                        ((meter == m) & (anomaly == a)).sum()
                    )
                    for m in sorted(np.unique(meter))
                    for a in (1, 0)
                },
            }
    entries.append(q)

    frozen = proc / "m5_hotwater_label_factorial" / "independent_query" / "queries.npz"
    entries.append(
        {
            "role": "frozen_192_row_independent_query",
            "path": str(frozen),
            "present": frozen.exists(),
            "status": "FROZEN — not read, not scored, not used by C1",
        }
    )

    missing = [e for e in entries if not e.get("present")]
    payload = {
        "schema": "m5_c1_artifact_census_v1",
        "execution_mode": "C1_LOCALIZATION",
        "artifact_count": len(entries),
        "missing_count": len(missing),
        "missing": [e["role"] for e in missing],
        "artifacts": entries,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root / "c1_artifact_census.json"
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    for e in entries:
        mark = "OK " if e.get("present") else "MISS"
        extra = ""
        if e.get("rows"):
            extra = f" rows={e['rows']:,}"
        print(f"  [{mark}] {e['role']}{extra}")
    if missing:
        print(f"\nMISSING {len(missing)} required artifacts; C1 stops here.")
        return 1
    print("\nall required artifacts present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
