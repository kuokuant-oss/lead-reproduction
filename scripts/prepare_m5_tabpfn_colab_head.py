"""Convert verified local-head checkpoints into portable Colab checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from lead import PROC


DEFAULT_SOURCE = PROC / "m5_tabpfn_canonical_full_test_context100000.work" / "chunks"
DEFAULT_METADATA = (
    PROC / "m5_tabpfn_distributed_context100000" / "head" / "metadata.npz"
)
DEFAULT_OUT = PROC / "m5_tabpfn_distributed_context100000" / "head-results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-chunks", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint-rows", type=int, default=20_000)
    return parser.parse_args(argv)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def convert_checkpoint(
    source: Path,
    destination: Path,
    expected: dict[str, np.ndarray],
) -> None:
    with np.load(source) as payload:
        required = {"raw_index", "y", "score", "site_id", "building_id"}
        if missing := required - set(payload.files):
            raise ValueError(f"checkpoint missing arrays: {sorted(missing)}")
        observed = {name: np.asarray(payload[name]) for name in required}
    comparisons = {
        "raw_index": expected["raw_index"],
        "y": expected["anomaly"],
        "site_id": expected["site_id"],
        "building_id": expected["building_id"],
    }
    for name, wanted in comparisons.items():
        if not np.array_equal(observed[name], wanted):
            raise AssertionError(f"{source} {name} does not match head metadata")
    if not np.isfinite(observed["score"]).all():
        raise AssertionError(f"{source} contains non-finite scores")
    atomic_write_npz(
        destination,
        raw_index=observed["raw_index"].astype("int64", copy=False),
        anomaly=observed["y"].astype("int8", copy=False),
        score=observed["score"].astype("float32", copy=False),
        site_id=observed["site_id"].astype("int8", copy=False),
        building_id=observed["building_id"].astype("int16", copy=False),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with np.load(args.metadata) as payload:
        required = {
            "raw_index",
            "anomaly",
            "site_id",
            "building_id",
            "global_position",
        }
        if missing := required - set(payload.files):
            raise ValueError(f"metadata missing arrays: {sorted(missing)}")
        metadata = {name: np.asarray(payload[name]) for name in required}
    chunks = args.out_dir / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    converted = 0
    for source in sorted(args.source_chunks.glob("chunk_*.npz")):
        index = int(source.stem.rsplit("_", 1)[1])
        start = index * args.checkpoint_rows
        end = min(len(metadata["raw_index"]), start + args.checkpoint_rows)
        if start >= len(metadata["raw_index"]):
            raise ValueError(f"{source} lies outside head metadata")
        expected = {
            name: values[start:end]
            for name, values in metadata.items()
            if name != "global_position"
        }
        destination = chunks / f"rows_{start:08d}_{end:08d}.npz"
        convert_checkpoint(source, destination, expected)
        converted += end - start
    progress = {
        "status": "prepared",
        "direction": "forward",
        "completed_rows": converted,
        "completed_fraction": converted / len(metadata["raw_index"]),
        "checkpoint_count": len(list(chunks.glob("rows_*.npz"))),
        "effective_microbatch_size": 1024,
    }
    atomic_write_json(args.out_dir / "progress.json", progress)
    print(json.dumps(progress, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
