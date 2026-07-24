"""Export CPU-portable 137-feature shards for parallel Colab inference.

Mirrors export_m5_tabpfn_colab_tail.py but with the 137-feature offline matrix
(17 baseline + 120 value-change columns) instead of the 17 baseline columns. The
value-change features are built once on the test split (building_id % 2 == 1),
with the original frame index preserved so shard rows map correctly, then both
the head and tail shards are written from that single matrix.

Feature order is taken verbatim from the 137 fit_manifest so the exported matrix
matches the fitted state column-for-column.
"""

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

from lead import PROC, SHIFTS, add_value_change_features, load_m3_frame

DEFAULT_SOURCE_WORK = PROC / "m5_tabpfn_137_full_test_context100000.work"
DEFAULT_SITE_PREDICTIONS = (
    PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"
)
DEFAULT_OUT_ROOT = PROC / "m5_tabpfn_137_distributed_context100000"
BOUNDARY = 5_060_000
VALUE_CHANGE_REGIME = "timestamp_merge"

SHARDS = {
    "head": {
        "global_start": 0,
        "global_end": BOUNDARY,
        "direction": "forward",
        "remote_root": PurePosixPath("/content/lead_tabpfn_137_head"),
    },
    "tail": {
        "global_start": BOUNDARY,
        "global_end": None,
        "direction": "reverse",
        "remote_root": PurePosixPath("/content/lead_tabpfn_137_tail"),
    },
}


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


def write_shard(
    name: str,
    spec: dict[str, Any],
    vframe: Any,
    feature_names: list[str],
    scaler: Any,
    source_work: Path,
    fit_manifest: dict[str, Any],
    full: dict[str, np.ndarray],
    out_root: Path,
    block_rows: int,
    force: bool,
) -> dict[str, Any]:
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "features.float32.npy"
    metadata_path = out_dir / "metadata.npz"
    portable_fit_path = out_dir / "model.portable.tabpfn_fit"
    manifest_path = out_dir / "manifest.json"
    if not force and any(
        p.exists()
        for p in (features_path, metadata_path, portable_fit_path, manifest_path)
    ):
        raise FileExistsError(f"{name} export exists; pass --force to replace")

    start = spec["global_start"]
    end = len(full["building_id"]) if spec["global_end"] is None else spec["global_end"]
    raw_index = full["raw_index"][start:end]
    y = full["anomaly"][start:end]
    site_id = full["site_id"][start:end]
    building_id = full["building_id"][start:end]
    global_position = np.arange(start, end, dtype="int64")

    missing = set(raw_index.tolist()) - set(vframe.index.tolist())
    if missing:
        raise AssertionError(f"{name}: {len(missing)} shard rows outside value frame")

    temporary_features = features_path.with_name(features_path.name + ".tmp")
    matrix = np.lib.format.open_memmap(
        temporary_features,
        mode="w+",
        dtype="float32",
        shape=(len(raw_index), len(feature_names)),
    )
    for s in range(0, len(raw_index), block_rows):
        e = min(len(raw_index), s + block_rows)
        block = vframe.loc[raw_index[s:e], feature_names].to_numpy(
            dtype="float32", copy=True
        )
        matrix[s:e] = scaler.transform(block).astype("float32", copy=False)
        matrix.flush()
    del matrix
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

    remote_checkpoint = spec["remote_root"] / "tabpfn-v3-classifier-v3_default.ckpt"
    relocate_fitted_archive(
        source_work / "model.tabpfn_fit", portable_fit_path, remote_checkpoint
    )
    foundation_path = Path(fit_manifest["model_path"])
    if not foundation_path.is_file():
        raise FileNotFoundError(foundation_path)
    if sha256_file(foundation_path) != fit_manifest["model_sha256"]:
        raise AssertionError("foundation checkpoint SHA-256 drifted")

    manifest = {
        "status": "ready",
        "shard": name,
        "direction": spec["direction"],
        "n_features": len(feature_names),
        "min_building_id": int(building_id[0]),
        "max_building_id": int(building_id[-1]),
        "global_start": start,
        "global_end": end,
        "rows": len(raw_index),
        "features": {
            "path": str(features_path.resolve()),
            "shape": [len(raw_index), len(feature_names)],
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
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-work-dir", type=Path, default=DEFAULT_SOURCE_WORK)
    parser.add_argument(
        "--site-predictions", type=Path, default=DEFAULT_SITE_PREDICTIONS
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--export-block-rows", type=int, default=100_000)
    parser.add_argument("--shard", choices=("head", "tail", "both"), default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    source_work = args.source_work_dir
    fit_manifest = json.loads(
        (source_work / "fit_manifest.json").read_text(encoding="utf-8")
    )
    feature_names = list(fit_manifest["feature_names"])
    if len(feature_names) != 137:
        raise AssertionError(f"expected 137 feature names, got {len(feature_names)}")
    scaler = joblib.load(source_work / "scaler.joblib")

    with np.load(args.site_predictions) as site:
        full = {
            "raw_index": np.asarray(site["validation_raw_index"], dtype="int64"),
            "anomaly": np.asarray(site["anomaly"], dtype="int8"),
            "site_id": np.asarray(site["site_id"], dtype="int8"),
            "building_id": np.asarray(site["building_id"], dtype="int16"),
        }
    if not np.all(full["building_id"][1:] >= full["building_id"][:-1]):
        raise AssertionError("canonical target is not ordered by building ID")

    frame = load_m3_frame(verbose=True)
    # Shard rows are the test split; build its value-change features once, keeping
    # the original frame index so shard raw_index lookups land on the right rows.
    test_mask = (frame["building_id"] % 2 == 1).to_numpy()
    test_df = frame.loc[test_mask].copy()
    test_df["_orig_index"] = test_df.index.to_numpy()
    print("building 137-feature value-change matrix on the test split", flush=True)
    vframe = add_value_change_features(
        test_df, list(SHIFTS), value_change_regime=VALUE_CHANGE_REGIME
    ).set_index("_orig_index")
    missing_cols = set(feature_names) - set(vframe.columns)
    if missing_cols:
        raise AssertionError(
            f"value frame missing feature columns: {sorted(missing_cols)}"
        )

    # Label/identity alignment proof against the canonical target.
    if not np.array_equal(
        frame.loc[full["raw_index"], "anomaly"].to_numpy(dtype="int8"), full["anomaly"]
    ):
        raise AssertionError("raw row IDs do not map to canonical labels")
    if not np.array_equal(
        frame.loc[full["raw_index"], "site_id"].to_numpy(dtype="int8"), full["site_id"]
    ):
        raise AssertionError("raw row IDs do not map to canonical sites")
    del frame

    names = ("head", "tail") if args.shard == "both" else (args.shard,)
    for name in names:
        print(f"=== exporting {name} ===", flush=True)
        manifest = write_shard(
            name,
            SHARDS[name],
            vframe,
            feature_names,
            scaler,
            source_work,
            fit_manifest,
            full,
            args.out_root,
            args.export_block_rows,
            args.force,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
