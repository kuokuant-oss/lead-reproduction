"""Build the 192-row query feature matrix once, with E4's exact code.

The 192 rows live in the holdout half, so this uses the same build_feature_matrix
call E4 used for its 352-row query, against the same holdout frame. The result is
cached with a digest so all 24 units share one verified build.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from lead import ROOT, load_m3_frame
from run_m5_story_ae_probe import build_feature_matrix, validate_feature_matrix

QUERY_DIR = (
    ROOT / "data" / "processed" / "m5_hotwater_label_factorial" / "independent_query"
)
QUERY_NPZ_SHA = "d780f0f8a96c47f49ffe061a72906728f1301056555350cabd979348aa41a2a0"
QUERY_RAW_SHA = "2fc4a638a2a0880f2b4d7feac87875c941d155f5fe5172b75b13d041b654fa16"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, required=True)
    args = ap.parse_args()
    args.cache_root.mkdir(parents=True, exist_ok=True)
    npz = args.cache_root / "query192.npz"
    meta = npz.with_suffix(".json")

    if sha256_file(QUERY_DIR / "queries.npz") != QUERY_NPZ_SHA:
        raise SystemExit("192-row queries.npz digest mismatch")
    with np.load(QUERY_DIR / "queries.npz", allow_pickle=True) as z:
        raw = np.asarray(z["raw_index"], dtype="int64")
        meter = np.asarray(z["meter"], dtype="int8")
        anom = np.asarray(z["anomaly"], dtype="int8")
        building = np.asarray(z["building_id"], dtype="int64")
    if hashlib.sha256(raw.tobytes()).hexdigest() != QUERY_RAW_SHA:
        raise SystemExit("192-row raw_index digest mismatch")

    t0 = time.perf_counter()
    frame = load_m3_frame(verbose=False)
    holdout = frame.loc[frame["building_id"] % 2 == 1]
    q = build_feature_matrix(holdout, raw, "F4", full_frame=holdout)
    validate_feature_matrix(q, matrix_name="e5 192-row query")
    if q.shape != (192, 137):
        raise SystemExit(f"unexpected query matrix shape {q.shape}")
    print(f"built in {time.perf_counter() - t0:,.0f}s shape={q.shape}", flush=True)

    tmp = npz.with_name(f".{npz.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        np.savez(
            fh, q=q, raw_index=raw, meter=meter, anomaly=anom, building_id=building
        )
    os.replace(tmp, npz)
    payload = {
        "npz_sha256": sha256_file(npz),
        "raw_index_sha256": QUERY_RAW_SHA,
        "queries_npz_sha256": QUERY_NPZ_SHA,
        "shape": list(q.shape),
        "dtype": str(q.dtype),
        "built": time.time(),
    }
    tmpm = meta.with_name(f".{meta.name}.{os.getpid()}.tmp")
    with tmpm.open("w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmpm, meta)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
