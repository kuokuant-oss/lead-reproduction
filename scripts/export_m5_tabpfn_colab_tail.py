"""Export a CPU-only, portable shard for parallel Colab inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import numpy as np

from lead import BASELINE_FEATURE_COLS, PROC, load_m3_frame


DEFAULT_SOURCE_WORK = PROC / "m5_tabpfn_canonical_full_test_context100000.work"
DEFAULT_SITE_PREDICTIONS = (
    PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"
)
DEFAULT_OUT = PROC / "m5_tabpfn_distributed_context100000" / "tail"
DEFAULT_REMOTE_ROOT = PurePosixPath("/content/lead_tabpfn_tail")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-start", type=int, default=5_060_000)
    parser.add_argument("--global-end", type=int)
    parser.add_argument("--shard", choices=("head", "tail"), default="tail")
    parser.add_argument(
        "--direction", choices=("forward", "reverse"), default="reverse"
    )
    parser.add_argument("--source-work-dir", type=Path, default=DEFAULT_SOURCE_WORK)
    parser.add_argument(
        "--site-predictions", type=Path, default=DEFAULT_SITE_PREDICTIONS
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--remote-root", type=PurePosixPath, default=DEFAULT_REMOTE_ROOT
    )
    parser.add_argument("--export-block-rows", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.global_start < 0:
        raise ValueError("global start must be nonnegative")
    if args.global_end is not None and args.global_end <= args.global_start:
        raise ValueError("global end must be greater than global start")
    if args.export_block_rows <= 0:
        raise ValueError("export block rows must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def relocate_fitted_archive(
    source: Path, destination: Path, remote_model_path: PurePosixPath
) -> None:
    """Copy an official fitted archive with only its model path relocated."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with zipfile.ZipFile(source, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    if "init_params.json" not in members:
        raise ValueError("fitted archive lacks init_params.json")
    parameters = json.loads(members["init_params.json"])
    parameters["model_path"] = str(remote_model_path)
    members["init_params.json"] = json.dumps(parameters).encode("utf-8")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.out_dir / "features.float32.npy"
    metadata_path = args.out_dir / "metadata.npz"
    portable_fit_path = args.out_dir / "model.portable.tabpfn_fit"
    manifest_path = args.out_dir / "manifest.json"
    if not args.force and any(
        path.exists()
        for path in (features_path, metadata_path, portable_fit_path, manifest_path)
    ):
        raise FileExistsError("shard export already exists; pass --force to replace")

    source_fit = args.source_work_dir / "model.tabpfn_fit"
    scaler_path = args.source_work_dir / "scaler.joblib"
    fit_manifest_path = args.source_work_dir / "fit_manifest.json"
    for required in (source_fit, scaler_path, fit_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    frame = load_m3_frame(verbose=True)
    with np.load(args.site_predictions) as site:
        required = {
            "validation_raw_index",
            "anomaly",
            "site_id",
            "building_id",
        }
        if missing := required - set(site.files):
            raise ValueError(f"site artifact missing arrays: {sorted(missing)}")
        full_raw_index = np.asarray(site["validation_raw_index"], dtype="int64")
        full_y = np.asarray(site["anomaly"], dtype="int8")
        full_site = np.asarray(site["site_id"], dtype="int8")
        full_building = np.asarray(site["building_id"], dtype="int16")
    if not np.all(full_building[1:] >= full_building[:-1]):
        raise AssertionError("canonical target is not ordered by building ID")
    shard_start = args.global_start
    if shard_start >= len(full_building):
        raise ValueError("global start selects no rows")
    shard_end = len(full_building) if args.global_end is None else args.global_end
    if shard_end > len(full_building):
        raise ValueError("global end exceeds canonical rows")
    raw_index = full_raw_index[shard_start:shard_end]
    y = full_y[shard_start:shard_end]
    site_id = full_site[shard_start:shard_end]
    building_id = full_building[shard_start:shard_end]
    global_position = np.arange(shard_start, shard_end, dtype="int64")
    if not np.array_equal(frame.loc[raw_index, "anomaly"].to_numpy(dtype="int8"), y):
        raise AssertionError("raw row IDs do not map to canonical labels")
    if not np.array_equal(
        frame.loc[raw_index, "site_id"].to_numpy(dtype="int8"), site_id
    ):
        raise AssertionError("raw row IDs do not map to canonical sites")
    if not np.array_equal(
        frame.loc[raw_index, "building_id"].to_numpy(dtype="int16"), building_id
    ):
        raise AssertionError("raw row IDs do not map to canonical buildings")

    scaler = joblib.load(scaler_path)
    temporary_features = features_path.with_name(features_path.name + ".tmp")
    matrix = np.lib.format.open_memmap(
        temporary_features,
        mode="w+",
        dtype="float32",
        shape=(len(raw_index), len(BASELINE_FEATURE_COLS)),
    )
    for start in range(0, len(raw_index), args.export_block_rows):
        end = min(len(raw_index), start + args.export_block_rows)
        block = frame.loc[raw_index[start:end], BASELINE_FEATURE_COLS].to_numpy(
            dtype="float32", copy=True
        )
        matrix[start:end] = scaler.transform(block).astype("float32", copy=False)
        matrix.flush()
    del matrix, frame, scaler
    with temporary_features.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary_features, features_path)
    atomic_write_npz(
        metadata_path,
        raw_index=raw_index,
        anomaly=y,
        site_id=site_id,
        building_id=building_id,
        global_position=global_position,
    )

    remote_checkpoint = args.remote_root / "tabpfn-v3-classifier-v3_default.ckpt"
    relocate_fitted_archive(source_fit, portable_fit_path, remote_checkpoint)
    fit_manifest = json.loads(fit_manifest_path.read_text(encoding="utf-8"))
    foundation_path = Path(fit_manifest["model_path"])
    if not foundation_path.is_file():
        raise FileNotFoundError(foundation_path)
    if sha256_file(foundation_path) != fit_manifest["model_sha256"]:
        raise AssertionError("foundation checkpoint SHA-256 drifted")
    manifest = {
        "status": "ready",
        "shard": args.shard,
        "direction": args.direction,
        "min_building_id": int(building_id[0]),
        "max_building_id": int(building_id[-1]),
        "global_start": shard_start,
        "global_end": shard_end,
        "rows": len(raw_index),
        "features": {
            "path": str(features_path.resolve()),
            "shape": [len(raw_index), len(BASELINE_FEATURE_COLS)],
            "dtype": "float32",
            "sha256": sha256_file(features_path),
        },
        "metadata": {
            "path": str(metadata_path.resolve()),
            "sha256": sha256_file(metadata_path),
            "raw_index_sha256": array_sha256(raw_index.astype("<i8")),
            "label_sha256": array_sha256(y),
            "site_sha256": array_sha256(site_id),
            "building_sha256": array_sha256(building_id.astype("<i2")),
        },
        "fit_state": {
            "path": str(portable_fit_path.resolve()),
            "sha256": sha256_file(portable_fit_path),
            "context_rows": fit_manifest["context_rows"],
            "context_sha256": fit_manifest["context_sha256"],
            "remote_model_path": str(remote_checkpoint),
        },
        "foundation_checkpoint": {
            "local_path": str(foundation_path.resolve()),
            "remote_path": str(remote_checkpoint),
            "sha256": fit_manifest["model_sha256"],
            "size_bytes": foundation_path.stat().st_size,
        },
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
