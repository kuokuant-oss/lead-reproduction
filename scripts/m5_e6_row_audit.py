"""Audit the full-holdout row set. Identity only -- no score column is opened.

E6 will score these rows with new factorial states. Before designing anything
around them, their identity has to be established from the artifacts rather than
assumed from a row count: unique, odd-building only, disjoint from the training
half, and in a canonical order that every existing full-test artifact agrees on.

The `tabpfn` score column present in those artifacts is never read.
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
PROCESSED = ROOT / "data" / "processed"

EXPECTED = {
    "rows": 10_137_155,
    "buildings": 724,
    "sites": 16,
    "anomaly_rows": 637_397,
    "sorted_raw_index_sha256": (
        "f0867d3e86ae2b017ea6fee2d1b9f6dead2ee241948346a467ea06305e220e76"
    ),
}
# Columns we are permitted to read. `tabpfn` is deliberately absent.
IDENTITY_COLUMNS = ("raw_index", "building_id", "site_id", "anomaly")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def full_test_artifacts() -> list[Path]:
    return sorted(PROCESSED.glob("m5_tabpfn_*full_test*predictions.npz"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--reference",
        type=Path,
        default=PROCESSED / "m5_tabpfn_137_full_test_context20000_n8_predictions.npz",
    )
    args = ap.parse_args()

    findings: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> bool:
        print(
            f"  {'OK  ' if ok else 'FAIL'} {label}{(' -- ' + detail) if detail else ''}"
        )
        if not ok:
            findings.append(label)
        return ok

    print("== reference row set ==")
    with np.load(args.reference) as z:
        available = list(z.files)
        raw = np.asarray(z["raw_index"], dtype="int64")
        bid = np.asarray(z["building_id"], dtype="int64")
        sid = np.asarray(z["site_id"], dtype="int64")
        anom = np.asarray(z["anomaly"], dtype="int8")
    print(f"       columns present: {available}")
    print(
        f"       columns read   : {list(IDENTITY_COLUMNS)}  (score column not opened)"
    )

    check(raw.size == EXPECTED["rows"], "row count", f"{raw.size:,}")
    order = np.argsort(raw, kind="stable")
    srt = raw[order]
    check(
        np.unique(raw).size == raw.size,
        "every raw_index unique",
        f"{np.unique(raw).size:,} distinct",
    )
    check(
        hashlib.sha256(srt.tobytes()).hexdigest()
        == EXPECTED["sorted_raw_index_sha256"],
        "sorted raw_index digest",
    )
    parity = np.unique(bid % 2)
    check(
        parity.size == 1 and parity[0] == 1,
        "every row is an odd building",
        f"parities {parity.tolist()}",
    )
    check(
        np.unique(bid).size == EXPECTED["buildings"],
        "building count",
        str(np.unique(bid).size),
    )
    check(
        np.unique(sid).size == EXPECTED["sites"], "site count", str(np.unique(sid).size)
    )
    check(
        int(anom.sum()) == EXPECTED["anomaly_rows"],
        "anomaly row count",
        f"{int(anom.sum()):,}",
    )
    prevalence = float(anom.mean())
    check(
        abs(prevalence - 0.0629) < 5e-4,
        "natural prevalence",
        f"{prevalence:.6f}",
    )

    print("== disjoint from the training half ==")
    # The split rule is building_id % 2; an odd-only set cannot intersect the
    # even half, and the parity check above already proves it for every row.
    check(
        not bool((bid % 2 == 0).any()),
        "no even-building row present, so disjoint from the fit half",
    )

    print("== agreement across every existing full-test artifact ==")
    arts, mismatch = [], []
    for p in full_test_artifacts():
        with np.load(p) as z:
            if "raw_index" not in z.files:
                continue
            r = np.asarray(z["raw_index"], dtype="int64")
        same_order = np.array_equal(r, raw)
        same_set = (
            hashlib.sha256(np.sort(r).tobytes()).hexdigest()
            == EXPECTED["sorted_raw_index_sha256"]
        )
        arts.append(
            {
                "artifact": p.name,
                "rows": int(r.size),
                "identical_order": bool(same_order),
                "identical_row_set": bool(same_set),
            }
        )
        if not (same_order and same_set):
            mismatch.append(p.name)
    check(
        bool(arts) and not mismatch,
        f"{len(arts)} full-test artifacts share one row set and one order",
        ",".join(mismatch),
    )

    print("== canonical order ==")
    is_sorted = bool(np.all(np.diff(raw) > 0))
    check(True, "canonical order captured", "ascending" if is_sorted else "not sorted")

    print("== per-meter and per-site strata (identity only) ==")
    meter_path = PROCESSED / "m5_tabpfn_137_shard_verification.json"
    per_site = {}
    if meter_path.exists():
        sv = json.loads(meter_path.read_text(encoding="utf-8"))
        for _key, block in sv.items():
            if isinstance(block, dict) and "identity" in block:
                per_site = block["identity"].get("per_site", {})
                break
    check(bool(per_site), "per-site identity recovered from shard verification")

    site_rows = {str(s): int((sid == s).sum()) for s in np.unique(sid)}
    site_anom = {str(s): int(anom[sid == s].sum()) for s in np.unique(sid)}

    payload = {
        "schema": "m5_e6_row_manifest_v1",
        "generated": time.time(),
        "scope": "identity audit only; no score column was opened",
        "columns_read": list(IDENTITY_COLUMNS),
        "columns_present_but_not_read": [
            c for c in available if c not in IDENTITY_COLUMNS
        ],
        "row_set": {
            "rows": int(raw.size),
            "unique_raw_index": int(np.unique(raw).size),
            "sorted_raw_index_sha256": hashlib.sha256(srt.tobytes()).hexdigest(),
            "as_stored_raw_index_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            "stored_order_is_ascending": is_sorted,
            "buildings": int(np.unique(bid).size),
            "sites": int(np.unique(sid).size),
            "anomaly_rows": int(anom.sum()),
            "natural_prevalence": prevalence,
            "building_id_parity": [int(x) for x in parity],
            "split_rule": "holdout is building_id % 2 == 1",
            "disjoint_from_fit_half": True,
        },
        "per_site_rows": site_rows,
        "per_site_anomaly": site_anom,
        "shard_verification_per_site": per_site,
        "agreeing_artifacts": arts,
        "required_wording": (
            "natural-prevalence factorial confirmation using new factorial "
            "states on previously characterised holdout rows"
        ),
        "forbidden_wording": [
            "untouched holdout",
            "first contact",
            "previously unseen row set",
        ],
        "findings": findings,
        "passed": not findings,
    }
    digest = atomic_json(args.out / "e6_row_manifest.json", payload)
    print(f"\nrow manifest sha256 = {digest}")
    print(f"AUDIT {'PASSED' if not findings else 'FAILED: ' + ', '.join(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
