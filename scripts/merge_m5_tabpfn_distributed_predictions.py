"""Merge local-head and Colab-tail TabPFN checkpoints with exact row checks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

from lead import PROC


DEFAULT_CANONICAL = PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"
DEFAULT_HEAD = PROC / "m5_tabpfn_canonical_full_test_context100000.work" / "chunks"
DEFAULT_TAIL = PROC / "m5_tabpfn_distributed_context100000" / "tail-results"
DEFAULT_OUT = PROC / "m5_tabpfn_distributed_context100000_predictions.npz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--head-chunks", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--tail-chunks", type=Path, default=DEFAULT_TAIL)
    parser.add_argument("--boundary", type=int, default=5_060_000)
    parser.add_argument("--checkpoint-rows", type=int, default=20_000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.boundary <= 0 or args.boundary % args.checkpoint_rows:
        raise ValueError("boundary must be a positive checkpoint boundary")
    return args


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def head_checkpoint_path(directory: Path, start: int, checkpoint_rows: int) -> Path:
    return directory / f"chunk_{start // checkpoint_rows:06d}.npz"


def tail_checkpoint_path(directory: Path, start: int, end: int) -> Path:
    return directory / f"rows_{start:08d}_{end:08d}.npz"


def load_score_checkpoint(
    path: Path,
    *,
    expected_raw_index: np.ndarray,
    expected_y: np.ndarray,
    expected_site: np.ndarray,
    expected_building: np.ndarray,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        y_key = "anomaly" if "anomaly" in payload.files else "y"
        required = {"raw_index", y_key, "score", "site_id", "building_id"}
        if missing := required - set(payload.files):
            raise ValueError(f"checkpoint missing arrays: {sorted(missing)}")
        observed = {
            "raw_index": np.asarray(payload["raw_index"]),
            "y": np.asarray(payload[y_key]),
            "score": np.asarray(payload["score"]),
            "site_id": np.asarray(payload["site_id"]),
            "building_id": np.asarray(payload["building_id"]),
        }
    comparisons = {
        "raw_index": expected_raw_index,
        "y": expected_y,
        "site_id": expected_site,
        "building_id": expected_building,
    }
    for name, expected in comparisons.items():
        if not np.array_equal(observed[name], expected):
            raise AssertionError(f"checkpoint {name} drifted: {path}")
    if not np.isfinite(observed["score"]).all():
        raise AssertionError(f"checkpoint scores are non-finite: {path}")
    return observed["score"].astype("float32", copy=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with np.load(args.canonical) as canonical:
        required = {"validation_raw_index", "anomaly", "site_id", "building_id"}
        if missing := required - set(canonical.files):
            raise ValueError(f"canonical artifact missing: {sorted(missing)}")
        raw_index = np.asarray(canonical["validation_raw_index"], dtype="int64")
        y = np.asarray(canonical["anomaly"], dtype="int8")
        site_id = np.asarray(canonical["site_id"], dtype="int8")
        building_id = np.asarray(canonical["building_id"], dtype="int16")
    if not 0 < args.boundary < len(raw_index):
        raise ValueError("boundary lies outside canonical rows")

    score = np.empty(len(raw_index), dtype="float32")
    for start in range(0, len(raw_index), args.checkpoint_rows):
        end = min(len(raw_index), start + args.checkpoint_rows)
        if start < args.boundary:
            if end > args.boundary:
                raise AssertionError("head checkpoint crosses distributed boundary")
            path = head_checkpoint_path(args.head_chunks, start, args.checkpoint_rows)
        else:
            path = tail_checkpoint_path(args.tail_chunks, start, end)
        score[start:end] = load_score_checkpoint(
            path,
            expected_raw_index=raw_index[start:end],
            expected_y=y[start:end],
            expected_site=site_id[start:end],
            expected_building=building_id[start:end],
        )

    atomic_write_npz(
        args.out,
        raw_index=raw_index,
        anomaly=y,
        tabpfn=score,
        site_id=site_id,
        building_id=building_id,
    )
    print(f"Saved {args.out} with {len(score):,} exactly aligned rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
