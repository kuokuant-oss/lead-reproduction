"""Build the exact microbatch manifest and derive every call census from it.

The design audit stated call counts computed by hand from a row count. Those
numbers were wrong: a shard whose row count is not a multiple of the microbatch
size ends in a short batch, so the true count is a sum of ceilings, not a
division. Every census below is derived from this manifest programmatically, and
`tests/test_m5_e6_design.py` refuses the hand-written figures.

Identity only: `raw_index`, `building_id`, `site_id`, `anomaly` and the `meter`
feature column. No score column is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\tonykuo\projects\lead-reproduction")
DIST = ROOT / "data" / "processed" / "m5_tabpfn_137_distributed_context100000"

ROWS = 10_137_155
SHARDS = 12
MICROBATCH_MAX = 20_000
STATES = 24
SENTINEL_ROWS = 352
SENTINEL_REPEATS = 8


def atomic_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load_identity() -> dict[str, np.ndarray]:
    """Canonical-order identity columns, head then tail."""
    raw, anom, bid, sid, meter = [], [], [], [], []
    for half in ("head", "tail"):
        md = np.load(DIST / half / "metadata.npz")
        raw.append(np.asarray(md["raw_index"], dtype="int64"))
        anom.append(np.asarray(md["anomaly"], dtype="int8"))
        bid.append(np.asarray(md["building_id"], dtype="int64"))
        sid.append(np.asarray(md["site_id"], dtype="int64"))
        f = np.load(DIST / half / "features.float32.npy", mmap_mode="r")
        meter.append(np.asarray(f[:, 0]))
        del f
    m = np.concatenate(meter)
    levels = sorted(np.unique(m).tolist())
    lut = {v: i for i, v in enumerate(levels)}
    meter_code = np.zeros(m.size, dtype="int8")
    for v, i in lut.items():
        meter_code[m == v] = i
    return {
        "raw_index": np.concatenate(raw),
        "anomaly": np.concatenate(anom),
        "building_id": np.concatenate(bid),
        "site_id": np.concatenate(sid),
        "meter": meter_code,
    }


def shard_bounds(shards: int = SHARDS) -> list[tuple[int, int]]:
    base, extra = divmod(ROWS, shards)
    out, start = [], 0
    for i in range(shards):
        n = base + (1 if i < extra else 0)
        out.append((start, start + n))
        start += n
    assert start == ROWS
    return out


def build(idt: dict[str, np.ndarray]) -> dict:
    raw = idt["raw_index"]
    entries, covered = [], 0
    for shard_id, (s0, s1) in enumerate(shard_bounds()):
        pos = s0
        mb = 0
        while pos < s1:
            stop = min(pos + MICROBATCH_MAX, s1)
            sl = slice(pos, stop)
            b, si, me, an = (
                idt["building_id"][sl],
                idt["site_id"][sl],
                idt["meter"][sl],
                idt["anomaly"][sl],
            )
            entries.append(
                {
                    "shard_id": shard_id,
                    "microbatch_id": mb,
                    "canonical_start": pos,
                    "canonical_stop": stop,
                    "row_count": stop - pos,
                    "raw_index_sha256": hashlib.sha256(raw[sl].tobytes()).hexdigest(),
                    "buildings": int(np.unique(b).size),
                    "sites": int(np.unique(si).size),
                    "anomaly_rows": int(an.sum()),
                    "meter_counts": {
                        str(k): int(v)
                        for k, v in zip(*np.unique(me, return_counts=True))
                    },
                    "expected_output_start": pos,
                    "expected_output_stop": stop,
                }
            )
            covered += stop - pos
            pos, mb = stop, mb + 1
    assert covered == ROWS, covered

    calls_per_state = len(entries)
    census = {
        "microbatches_per_state": calls_per_state,
        "full_holdout_predict_proba_calls_per_state": calls_per_state,
        "full_holdout_predict_proba_calls_all_states": calls_per_state * STATES,
        "sentinel_predict_proba_calls_per_state": SENTINEL_REPEATS,
        "sentinel_predict_proba_calls_all_states": SENTINEL_REPEATS * STATES,
        "r1_plus_sentinel_total_calls": calls_per_state * STATES
        + SENTINEL_REPEATS * STATES,
        "r8_full_holdout_calls_all_states": calls_per_state * STATES * 8,
        "full_holdout_row_scores": ROWS * STATES,
        "sentinel_row_scores": SENTINEL_ROWS * SENTINEL_REPEATS * STATES,
        "r8_row_scores": ROWS * STATES * 8,
        "rows_covered": covered,
        "derivation": "counted from the microbatch manifest, not divided",
    }
    return {
        "schema": "m5_e6_microbatch_manifest_v1",
        "generated": time.time(),
        "rows": ROWS,
        "shards": SHARDS,
        "microbatch_max_rows": MICROBATCH_MAX,
        "order": "canonical stored order; shards are never reordered and no "
        "shard is sorted by raw_index",
        "every_row_exactly_once": True,
        "census": census,
        "microbatches": entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    idt = load_identity()
    if idt["raw_index"].size != ROWS:
        raise SystemExit(f"identity has {idt['raw_index'].size} rows")
    payload = build(idt)
    digest = atomic_json(args.out / "e6_microbatch_manifest.json", payload)

    c = payload["census"]
    print(f"microbatch manifest sha256 = {digest}")
    print(f"  microbatches per state              : {c['microbatches_per_state']:,}")
    print(
        f"  full-holdout calls per state        : "
        f"{c['full_holdout_predict_proba_calls_per_state']:,}"
    )
    print(
        f"  full-holdout calls, 24 states       : "
        f"{c['full_holdout_predict_proba_calls_all_states']:,}"
    )
    print(
        f"  sentinel calls, 24 states           : "
        f"{c['sentinel_predict_proba_calls_all_states']:,}"
    )
    print(
        f"  R1_PLUS_SENTINEL total calls        : {c['r1_plus_sentinel_total_calls']:,}"
    )
    print(
        f"  R8 full-holdout calls, 24 states    : "
        f"{c['r8_full_holdout_calls_all_states']:,}"
    )
    print(f"  full-holdout row scores             : {c['full_holdout_row_scores']:,}")
    print(f"  sentinel row scores                 : {c['sentinel_row_scores']:,}")
    print(f"  rows covered                        : {c['rows_covered']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
