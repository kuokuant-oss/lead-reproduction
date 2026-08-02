"""證明 GPUtw benchmark 的輸入完全不含 holdout 列,並把證明壓縮成一個 digest。

交集證明在筆電上做,因為只有筆電同時持有 probe 與 holdout 的 raw_index。
GPUtw 端只需驗證 probe 檔的 digest 與這裡記錄的相符,就繼承了整個證明 ——
holdout 的 raw_index 清單因此完全不必上傳,這比「上傳後再檢查」更強:
不存在的東西不會被誤用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

PROBE_ROWS = 200_000
HOLDOUT_ROWS = 10_137_155
HOLDOUT_SORTED_DIGEST = (
    "f0867d3e86ae2b017ea6fee2d1b9f6dead2ee241948346a467ea06305e220e76"
)
PROBE_ARTIFACT_DIGEST = (
    "afe80b114ed375298641ad57ff93752afb3214c8d8e34bfb241184a3e7b46279"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def disjoint_proof(probe_raw: np.ndarray, holdout_raw: np.ndarray) -> dict:
    """probe 與 holdout 的 raw_index 交集必須是空集合。"""
    if probe_raw.size != PROBE_ROWS:
        raise SystemExit(f"probe 有 {probe_raw.size} 列,預期 {PROBE_ROWS}")
    if holdout_raw.size != HOLDOUT_ROWS:
        raise SystemExit(f"holdout 有 {holdout_raw.size} 列,預期 {HOLDOUT_ROWS}")
    sorted_digest = hashlib.sha256(np.sort(holdout_raw).tobytes()).hexdigest()
    if sorted_digest != HOLDOUT_SORTED_DIGEST:
        raise SystemExit("holdout raw_index 與凍結的 holdout digest 不符")
    overlap = np.intersect1d(probe_raw, holdout_raw, assume_unique=False)
    if overlap.size != 0:
        raise SystemExit(
            f"HARD FAILURE probe 與 holdout 有 {overlap.size} 列重疊;"
            "GPUtw benchmark 絕不能碰到 holdout 列"
        )
    return {
        "probe_rows": int(probe_raw.size),
        "holdout_rows": int(holdout_raw.size),
        "intersection_size": 0,
        "probe_raw_index_sha256": hashlib.sha256(probe_raw.tobytes()).hexdigest(),
        "holdout_sorted_raw_index_sha256": sorted_digest,
        "probe_raw_index_unique": int(np.unique(probe_raw).size) == int(probe_raw.size),
        "method": "np.intersect1d over the full raw_index sets, not a sample",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-npz", type=Path, required=True)
    ap.add_argument("--holdout-raw-index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with np.load(args.probe_npz) as z:
        probe_raw = np.asarray(z["raw_index"], dtype="int64")
    holdout_raw = np.asarray(np.load(args.holdout_raw_index), dtype="int64")

    proof = disjoint_proof(probe_raw, holdout_raw)
    payload = {
        "schema": "m5_e6_gputw_probe_guard_v1",
        "generated": time.time(),
        "purpose": (
            "證明 GPUtw benchmark 只會讀到 even-building 的 non-holdout 列。"
            "GPUtw 端只驗 probe_npz_sha256,不需要也不應該持有 holdout 清單。"
        ),
        "probe_npz": args.probe_npz.name,
        "probe_npz_sha256": sha256_file(args.probe_npz),
        "probe_artifact_digest_expected": PROBE_ARTIFACT_DIGEST,
        "disjoint_proof": proof,
        "holdout_list_uploaded_to_gputw": False,
        "gputw_may_score_holdout": False,
    }
    digest = atomic_json(args.out / "probe_guard.json", payload)
    print(f"probe/holdout 交集 = {proof['intersection_size']}  (必須為 0)")
    print(f"probe npz sha256   = {payload['probe_npz_sha256']}")
    print(f"probe_guard.json   = {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
