"""Build the canonical raw F4/137 full-holdout matrix for M5 E6.

The existing 5.3 GB distributed artifact cannot be reused: its features are
already scaled, and E6 must apply each unit's own scaler to raw features. So the
raw matrix is built once here, digest-frozen, and shared byte-identically by both
hosts.

`build_feature_matrix` recomputes `add_value_change_features` over the entire
`full_frame` on every call and then selects the requested rows. Calling it once
per chunk would repeat a 10.1M-row computation 21 times. This module hoists that
computation out of the loop and then proves the hoist is faithful by rebuilding
sampled blocks through the real function and requiring bit-exact agreement --
the hoist is a performance change only if it is checkable, so it is checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROWS = 10_137_155
FEATURES = 137
CHUNK = 500_000
VERIFY_BLOCKS = 4
VERIFY_BLOCK_ROWS = 500
VERIFY_SEED = 20260802
HOLDOUT_DIGEST = "f0867d3e86ae2b017ea6fee2d1b9f6dead2ee241948346a467ea06305e220e76"


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


def canonical_raw_index(dist_root: Path) -> np.ndarray:
    """The canonical stored order: head then tail, never re-sorted."""
    parts = []
    for half in ("head", "tail"):
        md = np.load(dist_root / half / "metadata.npz")
        parts.append(np.asarray(md["raw_index"], dtype="int64"))
    raw = np.concatenate(parts)
    if raw.size != ROWS:
        raise SystemExit(f"canonical order has {raw.size} rows, expected {ROWS}")
    sorted_digest = hashlib.sha256(np.sort(raw).tobytes()).hexdigest()
    if sorted_digest != HOLDOUT_DIGEST:
        raise SystemExit("canonical raw_index does not match the frozen holdout")
    if np.unique(raw).size != ROWS:
        raise SystemExit("canonical raw_index contains duplicates")
    return raw


def hoisted_frame(holdout):
    """`build_feature_matrix`'s full-frame stage, computed once."""
    from run_m5_story_ae_probe import SHIFTS, add_value_change_features

    tagged = holdout.copy()
    tagged["__raw_index_carrier"] = tagged.index.to_numpy(dtype="int64")
    built = add_value_change_features(
        tagged, list(SHIFTS), value_change_regime="timestamp_merge"
    )
    built.index = built["__raw_index_carrier"].to_numpy(dtype="int64")
    return built


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dist-root", type=Path, required=True)
    ap.add_argument("--screening", type=Path, required=True)
    args = ap.parse_args()

    from lead import load_m3_frame
    from run_m5_story_ae_probe import (
        build_feature_matrix,
        feature_names,
        validate_feature_matrix,
    )

    columns = feature_names("F4")
    if len(columns) != FEATURES:
        raise SystemExit(f"F4 has {len(columns)} columns, expected {FEATURES}")

    raw = canonical_raw_index(args.dist_root)
    print(f"canonical order: {raw.size:,} rows, holdout digest verified")

    frame = load_m3_frame(verbose=False)
    holdout = frame.loc[frame["building_id"] % 2 == 1]
    if len(holdout) != ROWS:
        raise SystemExit(f"odd-building holdout has {len(holdout)} rows")

    args.out.mkdir(parents=True, exist_ok=True)
    npy = args.out / "e6_holdout_raw_f4_137.float32.npy"

    # The matrix is written atomically, so a file of the right shape and dtype
    # is a complete one -- but it is not yet a verified one, because the
    # manifest is written last and only after every check below passes. Resume
    # therefore skips the rebuild and never skips a check.
    resumed = False
    if npy.exists():
        try:
            existing = np.load(npy, mmap_mode="r")
            resumed = (
                existing.shape == (ROWS, FEATURES) and existing.dtype == np.float32
            )
            del existing
        except (OSError, ValueError):
            resumed = False
        if not resumed:
            print(f"discarding an unusable {npy.name} and rebuilding")
            npy.unlink()

    hoist_seconds = 0.0
    write_seconds = 0.0
    nonfinite = -1
    if resumed:
        print(f"resuming: {npy.name} is complete; re-running every verification")
    else:
        t0 = time.perf_counter()
        built = hoisted_frame(holdout)
        hoist_seconds = time.perf_counter() - t0
        print(f"full-frame stage computed once in {hoist_seconds:,.0f}s")

        tmp = npy.with_name(f".{npy.name}.{os.getpid()}.tmp")
        mm = np.lib.format.open_memmap(
            tmp, mode="w+", dtype="float32", shape=(ROWS, FEATURES)
        )
        t0 = time.perf_counter()
        nonfinite = 0
        for start in range(0, ROWS, CHUNK):
            stop = min(start + CHUNK, ROWS)
            block = built.loc[raw[start:stop], columns].to_numpy(
                dtype="float32", copy=True
            )
            if block.shape != (stop - start, FEATURES):
                raise SystemExit(f"chunk {start}: shape {block.shape}")
            nonfinite += int((~np.isfinite(block)).sum())
            mm[start:stop] = block
            del block
            print(
                f"  rows {stop:>10,} / {ROWS:,}  ({time.perf_counter() - t0:,.0f}s)",
                flush=True,
            )
        mm.flush()
        write_seconds = time.perf_counter() - t0
        del mm, built
        os.replace(tmp, npy)
        print(
            f"matrix written in {write_seconds:,.0f}s; non-finite cells: {nonfinite:,}"
        )

    # The hoist is only legitimate if it reproduces the real function exactly.
    check = np.load(npy, mmap_mode="r")
    if nonfinite < 0:
        nonfinite = 0
        for start in range(0, ROWS, CHUNK):
            nonfinite += int(
                (~np.isfinite(np.asarray(check[start : start + CHUNK]))).sum()
            )
        print(f"non-finite cells recounted from the resumed matrix: {nonfinite:,}")
    rng = np.random.default_rng(VERIFY_SEED)
    verifications = []
    for b in range(VERIFY_BLOCKS):
        start = int(rng.integers(0, ROWS - VERIFY_BLOCK_ROWS))
        stop = start + VERIFY_BLOCK_ROWS
        ref = build_feature_matrix(holdout, raw[start:stop], "F4", full_frame=holdout)
        got = np.asarray(check[start:stop])
        if ref.dtype != np.float32 or got.dtype != np.float32:
            raise SystemExit("dtype drift in the hoist verification")
        same_nan = np.array_equal(np.isnan(ref), np.isnan(got))
        finite = np.isfinite(ref) & np.isfinite(got)
        max_diff = (
            float(np.abs(ref[finite] - got[finite]).max()) if finite.any() else 0.0
        )
        exact = bool(same_nan and max_diff == 0.0)
        verifications.append(
            {
                "block": b,
                "canonical_start": start,
                "canonical_stop": stop,
                "max_abs_difference": max_diff,
                "nan_pattern_identical": same_nan,
                "bit_exact": exact,
            }
        )
        print(f"  hoist check block {b}: rows {start:,}-{stop:,}  exact={exact}")
        if not exact:
            raise SystemExit(
                "HARD FAILURE the hoisted build does not reproduce "
                "build_feature_matrix bit for bit"
            )

    validate_feature_matrix(
        np.asarray(check[:VERIFY_BLOCK_ROWS]), matrix_name="e6 holdout"
    )

    # The 352 sentinel rows, built through the real function so the runner can
    # prove its slice of the big matrix reproduces E4's construction.
    with np.load(args.screening / "queries.npz") as z:
        sent_raw = np.asarray(z["raw_index"], dtype="int64")
    sent_x = build_feature_matrix(holdout, sent_raw, "F4", full_frame=holdout)
    if sent_x.shape != (sent_raw.size, FEATURES) or sent_x.dtype != np.float32:
        raise SystemExit(f"sentinel matrix {sent_x.shape} {sent_x.dtype}")
    sent_path = args.out / "e6_sentinel_query.npz"
    stmp = sent_path.with_name(f".{sent_path.name}.{os.getpid()}.tmp")
    with stmp.open("wb") as fh:
        np.savez(fh, x=sent_x, raw_index=sent_raw)
    os.replace(stmp, sent_path)
    print(f"sentinel query built: {sent_raw.size} rows")

    digest = hashlib.sha256()
    with npy.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    sha = digest.hexdigest()
    size = npy.stat().st_size

    raw_path = args.out / "e6_holdout_raw_index.npy"
    np.save(raw_path, raw)

    manifest = {
        "schema": "m5_e6_feature_manifest_v1",
        "generated": time.time(),
        "path": npy.name,
        "shape": [ROWS, FEATURES],
        "dtype": "float32",
        "size_bytes": size,
        "size_gb": size / 1e9,
        "sha256": sha,
        "raw_index_path": raw_path.name,
        "sentinel_query_path": sent_path.name,
        "sentinel_query_sha256": sha256_file(sent_path),
        "sentinel_rows": int(sent_raw.size),
        "raw_index_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "sorted_raw_index_sha256": HOLDOUT_DIGEST,
        "feature_tag": "F4",
        "scaled": False,
        "scaling_note": "raw features; each unit applies its own scaler at "
        "scoring time, which is why the existing scaled distributed artifact "
        "could not be reused",
        "order": "canonical stored order (head then tail); never sorted",
        "source_frame": "load_m3_frame() restricted to building_id % 2 == 1",
        "full_frame": "the same odd-building holdout frame",
        "construction": "add_value_change_features over the whole holdout once, "
        "then row selection per chunk",
        "hoist_verification": verifications,
        "hoist_seconds": hoist_seconds,
        "write_seconds": write_seconds,
        "non_finite_cells": nonfinite,
        "chunk_rows": CHUNK,
        "resumed_from_an_existing_matrix": resumed,
        "verification_is_never_skipped_on_resume": True,
    }
    mdigest = atomic_json(args.out / "e6_feature_manifest.json", manifest)

    print(f"\nfeature matrix sha256 = {sha}")
    print(f"  size                = {size / 1e9:.2f} GB")
    print(f"  manifest sha256     = {mdigest}")
    print(f"  hoist verified      = {len(verifications)}/{VERIFY_BLOCKS} bit-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
