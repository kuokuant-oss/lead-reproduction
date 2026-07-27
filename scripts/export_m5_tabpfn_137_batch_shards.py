"""Export one planned 137-feature batch as head/tail portable shards.

The batch's rows are selected out of the already-built full-test 137-feature
matrix rather than recomputed: that matrix was scaled with this fit's context
scaler, so slicing preserves the exact values the model expects and skips the
quarter-hour value-change feature build.

A batch's canonical positions are deliberately *not* contiguous -- sites 1, 2
and 3 are already scored and are interleaved by building_id -- so the row set is
recomputed from the same rule the planner used and cross-checked against the
plan's recorded row count, anomaly count and building range before anything is
written.
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

from lead import PROC

DEFAULT_PLAN = PROC / "m5_tabpfn_137_remaining_batch_plan.json"
DEFAULT_CANONICAL = PROC / "m5_tabpfn_distributed_context100000_predictions.npz"


def default_source_work(context_rows: int) -> Path:
    return PROC / f"m5_tabpfn_137_full_test_context{context_rows}_n8.work"


def default_slice_root(context_rows: int) -> Path:
    # The 100k slice root predates the context suffix, so it keeps its name.
    suffix = "" if context_rows == 100_000 else "_n8"
    return PROC / f"m5_tabpfn_137_distributed_context{context_rows}{suffix}"


def default_out_root(batch: int, context_rows: int) -> Path:
    return PROC / f"m5_tabpfn_f137_batch{batch}_context{context_rows}_n8"


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
    parameters = json.loads(members["init_params.json"])
    parameters["model_path"] = str(remote_model_path)
    members["init_params.json"] = json.dumps(parameters).encode("utf-8")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def open_slice_sources(slice_root: Path, n_features: int):
    sources = []
    offset = 0
    for shard in ("head", "tail"):
        manifest = json.loads(
            (slice_root / shard / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest["n_features"] != n_features:
            raise AssertionError("slice source feature count differs from the fit")
        matrix = np.load(slice_root / shard / "features.float32.npy", mmap_mode="r")
        sources.append((offset, matrix))
        offset += len(matrix)
    return sources, offset


def read_rows(sources, wanted: np.ndarray, n_features: int) -> np.ndarray:
    out = np.empty((len(wanted), n_features), dtype="float32")
    for offset, matrix in sources:
        local = wanted - offset
        inside = (local >= 0) & (local < len(matrix))
        if inside.any():
            out[inside] = matrix[local[inside]]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--context-rows", type=int, default=100_000)
    parser.add_argument("--source-work-dir", type=Path, default=None)
    parser.add_argument("--slice-from", type=Path, default=None)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--block-rows", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    source_work = args.source_work_dir or default_source_work(args.context_rows)
    slice_from = args.slice_from or default_slice_root(args.context_rows)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    entry = next(b for b in plan["batches"] if b["batch"] == args.batch)
    out_root = args.out_root or default_out_root(args.batch, args.context_rows)

    fit_manifest = json.loads((source_work / "fit_manifest.json").read_text("utf-8"))
    if int(fit_manifest["n_estimators"]) != 8:
        raise AssertionError("batch shards expect the n_estimators=8 fitted state")
    # Every path above can be overridden, so the context is checked against the
    # fit rather than inferred from a directory name. A mismatch here would ship
    # a matrix scaled by the wrong context: finite scores, correct rows, wrong
    # experiment, and no downstream gate can see it.
    fitted_context = int(fit_manifest["context_rows"])
    if fitted_context != args.context_rows:
        raise AssertionError(
            f"{source_work} was fit with context_rows={fitted_context}, "
            f"but --context-rows={args.context_rows}"
        )
    feature_names = list(fit_manifest["feature_names"])
    n_features = len(feature_names)

    with np.load(args.canonical) as data:
        site_id = np.asarray(data["site_id"])
        building_id = np.asarray(data["building_id"])
        anomaly = np.asarray(data["anomaly"])
        raw_index = np.asarray(data["raw_index"])

    remaining = ~np.isin(site_id, plan["done_sites"])
    positions = np.flatnonzero(remaining)
    selected = positions[entry["remaining_index_start"] : entry["remaining_index_end"]]

    # Refuse to write unless the recomputed slice is exactly what was planned.
    if len(selected) != entry["rows"]:
        raise AssertionError(
            f"batch {args.batch}: recomputed {len(selected)} rows, plan says {entry['rows']}"
        )
    if int(anomaly[selected].sum()) != entry["anomalies"]:
        raise AssertionError(f"batch {args.batch}: anomaly count differs from the plan")
    if (
        int(building_id[selected].min()) != entry["min_building_id"]
        or int(building_id[selected].max()) != entry["max_building_id"]
    ):
        raise AssertionError(
            f"batch {args.batch}: building range differs from the plan"
        )
    if np.isin(site_id[selected], plan["done_sites"]).any():
        raise AssertionError(f"batch {args.batch}: contains an already-scored site")

    sources, total = open_slice_sources(slice_from, n_features)
    if total != len(site_id):
        raise AssertionError("slice source rows differ from the canonical holdout")

    boundary = entry["head_rows"]
    specs = {
        "head": (0, boundary, "forward"),
        "tail": (boundary, len(selected), "reverse"),
    }
    foundation = Path(fit_manifest["model_path"])
    if sha256_file(foundation) != fit_manifest["model_sha256"]:
        raise AssertionError("foundation checkpoint SHA-256 drifted")

    for shard, (start, end, direction) in specs.items():
        out_dir = out_root / shard
        out_dir.mkdir(parents=True, exist_ok=True)
        features_path = out_dir / "features.float32.npy"
        metadata_path = out_dir / "metadata.npz"
        portable_fit_path = out_dir / "model.portable.tabpfn_fit"
        manifest_path = out_dir / "manifest.json"
        if not args.force and manifest_path.exists():
            raise FileExistsError(f"{out_dir} exists; pass --force")

        wanted = selected[start:end]
        remote_root = PurePosixPath(
            f"/content/lead_tabpfn_b{args.batch}_{shard}_f137_n8"
        )

        temporary = features_path.with_name(features_path.name + ".tmp")
        matrix = np.lib.format.open_memmap(
            temporary, mode="w+", dtype="float32", shape=(len(wanted), n_features)
        )
        for s in range(0, len(wanted), args.block_rows):
            e = min(len(wanted), s + args.block_rows)
            matrix[s:e] = read_rows(sources, wanted[s:e], n_features)
            matrix.flush()
        del matrix
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, features_path)

        atomic_write_npz(
            metadata_path,
            raw_index=raw_index[wanted].astype("int64"),
            anomaly=anomaly[wanted].astype("int8"),
            site_id=site_id[wanted].astype("int8"),
            building_id=building_id[wanted].astype("int16"),
            global_position=np.arange(start, end, dtype="int64"),
        )
        relocate_fitted_archive(
            source_work / "model.tabpfn_fit",
            portable_fit_path,
            remote_root / "tabpfn-v3-classifier-v3_default.ckpt",
        )

        manifest = {
            "status": "ready",
            "batch": args.batch,
            "shard": shard,
            "direction": direction,
            "n_features": n_features,
            "rows": len(wanted),
            "anomalies": int(anomaly[wanted].sum()),
            "sites": sorted({int(v) for v in site_id[wanted]}),
            "min_building_id": int(building_id[wanted].min()),
            "max_building_id": int(building_id[wanted].max()),
            "remote_root": str(remote_root),
            "features": {
                "path": str(features_path.resolve()),
                "shape": [len(wanted), n_features],
                "dtype": "float32",
                "sha256": sha256_file(features_path),
            },
            "metadata": {
                "path": str(metadata_path.resolve()),
                "sha256": sha256_file(metadata_path),
                "raw_index_sha256": array_sha256(raw_index[wanted].astype("<i8")),
                "label_sha256": array_sha256(anomaly[wanted].astype("int8")),
            },
            "fit_state": {
                "path": str(portable_fit_path.resolve()),
                "sha256": sha256_file(portable_fit_path),
                "context_rows": fit_manifest["context_rows"],
                "context_sha256": fit_manifest["context_sha256"],
                "n_estimators": 8,
                "source_work_dir": str(source_work.resolve()),
                "remote_model_path": str(
                    remote_root / "tabpfn-v3-classifier-v3_default.ckpt"
                ),
            },
            "foundation_checkpoint": {
                "local_path": str(foundation.resolve()),
                "remote_path": str(
                    remote_root / "tabpfn-v3-classifier-v3_default.ckpt"
                ),
                "sha256": fit_manifest["model_sha256"],
                "size_bytes": foundation.stat().st_size,
            },
        }
        atomic_write_json(manifest_path, manifest)
        print(
            f"batch {args.batch} {shard}: {len(wanted):,} rows, "
            f"{manifest['anomalies']:,} anomalies, sites {manifest['sites']}, "
            f"-> {remote_root}",
            flush=True,
        )

    _ = joblib.load(source_work / "scaler.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
