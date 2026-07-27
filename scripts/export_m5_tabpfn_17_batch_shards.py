"""Export the planned 17-feature batches as head/tail portable shards.

Sites 1, 2 and 3 are already scored at 17 features / n_estimators=8 by the
estimator sweep. Replacing the orange 17-feature TabPFN line on the four M3
figures needs the remaining thirteen sites scored the same way, so this exports
the same building_id batches the 137-feature line used -- identical geometry
keeps the two TabPFN lines comparable batch for batch.

Unlike the 137-feature exporter there is no prebuilt full-test matrix to slice
from, so rows are read out of the M3 frame and scaled with this fit's context
scaler. The frame costs a couple of minutes to load, which is why every batch is
exported in one pass by default rather than one process per batch.

A batch's canonical positions are deliberately *not* contiguous -- the
already-scored sites are interleaved by building_id -- so the row set is
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

from lead import BASELINE_FEATURE_COLS, PROC, load_m3_frame

DEFAULT_PLAN = PROC / "m5_tabpfn_17_remaining_batch_plan.json"
DEFAULT_CANONICAL = PROC / "m5_tabpfn_distributed_context100000_predictions.npz"


def default_source_work(context_rows: int) -> Path:
    return PROC / f"m5_tabpfn_canonical_full_test_context{context_rows}_n8.work"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def out_root_for(batch: int, context_rows: int = 100_000) -> Path:
    return PROC / f"m5_tabpfn_f17_batch{batch}_context{context_rows}_n8"


def remote_root_for(
    batch: int,
    shard: str,
    context_rows: int = 100_000,
    prefix: str = "/content",
) -> PurePosixPath:
    """Feature count, estimator count and context size belong in the remote path.

    The 137-feature batches cover exactly these rows, and the per-site 17-feature
    shards overlap the same holdout. Without both markers a --resume would happily
    accept another line's checkpoints as this line's finished work.

    The context curve adds a third way to collide, and it is the worst one: every
    context scores the same rows with the same feature count, so two contexts
    differ only in the scores. A resumed run across that collision is finite,
    complete, and silently the wrong experiment.

    100k keeps its unmarked path so the twelve already-exported and verified
    shards stay addressable, matching how ``estimator_suffix`` treats n=1.
    ``prefix`` is ``/content`` on Colab and ``/workspace`` on gputw.ai.
    """
    marker = "" if context_rows == 100_000 else f"c{context_rows}_"
    return PurePosixPath(f"{prefix}/lead_tabpfn_{marker}b{batch}_{shard}_f17_n8")


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


def select_batch_rows(
    entry: dict[str, Any],
    plan: dict[str, Any],
    canonical: dict[str, np.ndarray],
) -> np.ndarray:
    """Recompute a batch's canonical positions and refuse to trust the plan.

    The planner's own numbers are re-derived here rather than read, so a plan
    edited or regenerated against a different holdout cannot silently produce a
    shard covering the wrong rows.
    """
    remaining = ~np.isin(canonical["site_id"], plan["done_sites"])
    positions = np.flatnonzero(remaining)
    selected = positions[entry["remaining_index_start"] : entry["remaining_index_end"]]

    batch = entry["batch"]
    if len(selected) != entry["rows"]:
        raise AssertionError(
            f"batch {batch}: recomputed {len(selected)} rows, plan says {entry['rows']}"
        )
    if int(canonical["anomaly"][selected].sum()) != entry["anomalies"]:
        raise AssertionError(f"batch {batch}: anomaly count differs from the plan")
    if (
        int(canonical["building_id"][selected].min()) != entry["min_building_id"]
        or int(canonical["building_id"][selected].max()) != entry["max_building_id"]
    ):
        raise AssertionError(f"batch {batch}: building range differs from the plan")
    if np.isin(canonical["site_id"][selected], plan["done_sites"]).any():
        raise AssertionError(f"batch {batch}: contains an already-scored site")
    return selected


def prove_row_identity(
    frame: Any, canonical: dict[str, np.ndarray], selected: np.ndarray
) -> None:
    """The canonical row IDs must still map to the labels recorded for them."""
    raw_index = canonical["raw_index"][selected]
    for column in ("anomaly", "site_id", "building_id"):
        observed = frame.loc[raw_index, column].to_numpy(dtype=canonical[column].dtype)
        if not np.array_equal(observed, canonical[column][selected]):
            raise AssertionError(f"raw row IDs do not map to canonical {column}")


def write_shard(
    batch: int,
    shard: str,
    direction: str,
    wanted: np.ndarray,
    start: int,
    end: int,
    canonical: dict[str, np.ndarray],
    frame: Any,
    feature_names: list[str],
    scaler: Any,
    source_work: Path,
    fit_manifest: dict[str, Any],
    out_root: Path,
    block_rows: int,
    force: bool,
    context_rows: int = 100_000,
    remote_prefix: str = "/content",
) -> dict[str, Any]:
    out_dir = out_root / shard
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "features.float32.npy"
    metadata_path = out_dir / "metadata.npz"
    portable_fit_path = out_dir / "model.portable.tabpfn_fit"
    manifest_path = out_dir / "manifest.json"
    if not force and manifest_path.exists():
        raise FileExistsError(f"{out_dir} exists; pass --force")

    raw_index = canonical["raw_index"][wanted]
    anomaly = canonical["anomaly"][wanted]
    site_id = canonical["site_id"][wanted]
    building_id = canonical["building_id"][wanted]

    temporary = features_path.with_name(features_path.name + ".tmp")
    matrix = np.lib.format.open_memmap(
        temporary, mode="w+", dtype="float32", shape=(len(wanted), len(feature_names))
    )
    for s in range(0, len(wanted), block_rows):
        e = min(len(wanted), s + block_rows)
        block = frame.loc[raw_index[s:e], feature_names].to_numpy(
            dtype="float32", copy=True
        )
        matrix[s:e] = scaler.transform(block).astype("float32", copy=False)
        matrix.flush()
    del matrix
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, features_path)

    atomic_write_npz(
        metadata_path,
        raw_index=raw_index.astype("int64"),
        anomaly=anomaly.astype("int8"),
        site_id=site_id.astype("int8"),
        building_id=building_id.astype("int16"),
        global_position=np.arange(start, end, dtype="int64"),
    )

    remote_root = remote_root_for(batch, shard, context_rows, remote_prefix)
    remote_checkpoint = remote_root / "tabpfn-v3-classifier-v3_default.ckpt"
    relocate_fitted_archive(
        source_work / "model.tabpfn_fit", portable_fit_path, remote_checkpoint
    )
    foundation = Path(fit_manifest["model_path"])
    if sha256_file(foundation) != fit_manifest["model_sha256"]:
        raise AssertionError("foundation checkpoint SHA-256 drifted")

    manifest = {
        "status": "ready",
        "batch": batch,
        "shard": shard,
        "direction": direction,
        "n_features": len(feature_names),
        "batch_start": start,
        "batch_end": end,
        "rows": len(wanted),
        "anomalies": int(anomaly.sum()),
        "sites": sorted({int(v) for v in site_id}),
        "min_building_id": int(building_id.min()),
        "max_building_id": int(building_id.max()),
        "remote_root": str(remote_root),
        "features": {
            "path": str(features_path.resolve()),
            "shape": [len(wanted), len(feature_names)],
            "dtype": "float32",
            "sha256": sha256_file(features_path),
        },
        "metadata": {
            "path": str(metadata_path.resolve()),
            "sha256": sha256_file(metadata_path),
            "raw_index_sha256": array_sha256(raw_index.astype("<i8")),
            "label_sha256": array_sha256(anomaly.astype("int8")),
            "site_sha256": array_sha256(site_id.astype("int8")),
            "building_sha256": array_sha256(building_id.astype("<i2")),
        },
        "fit_state": {
            "path": str(portable_fit_path.resolve()),
            "sha256": sha256_file(portable_fit_path),
            "context_rows": fit_manifest["context_rows"],
            "context_sha256": fit_manifest["context_sha256"],
            "n_estimators": 8,
            "source_work_dir": str(source_work.resolve()),
            "remote_model_path": str(remote_checkpoint),
        },
        "foundation_checkpoint": {
            "local_path": str(foundation.resolve()),
            "remote_path": str(remote_checkpoint),
            "sha256": fit_manifest["model_sha256"],
            "size_bytes": foundation.stat().st_size,
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="*", default=None)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--context-rows", type=int, default=100_000)
    parser.add_argument("--source-work-dir", type=Path, default=None)
    parser.add_argument(
        "--remote-prefix",
        default="/content",
        help="remote parent dir: /content on Colab, /workspace on gputw.ai",
    )
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--block-rows", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    source_work = args.source_work_dir or default_source_work(args.context_rows)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    wanted_batches = (
        args.batches if args.batches else [b["batch"] for b in plan["batches"]]
    )
    entries = [b for b in plan["batches"] if b["batch"] in wanted_batches]
    if len(entries) != len(wanted_batches):
        raise SystemExit("plan does not contain every requested batch")

    fit_manifest = json.loads(
        (source_work / "fit_manifest.json").read_text(encoding="utf-8")
    )
    if int(fit_manifest.get("n_estimators", 1)) != 8:
        raise AssertionError("batch shards expect the n_estimators=8 fitted state")
    # Checked against the fit, not the directory name, because --source-work-dir
    # can override the convention. A wrong-context matrix yields finite scores on
    # the correct rows, which every downstream gate accepts.
    fitted_context = int(fit_manifest["context_rows"])
    if fitted_context != args.context_rows:
        raise AssertionError(
            f"{source_work} was fit with context_rows={fitted_context}, "
            f"but --context-rows={args.context_rows}"
        )
    feature_names = list(fit_manifest["feature_names"])
    if feature_names != list(BASELINE_FEATURE_COLS):
        raise AssertionError("fitted state does not use the 17 baseline features")
    scaler = joblib.load(source_work / "scaler.joblib")

    with np.load(args.canonical) as data:
        canonical = {
            "raw_index": np.asarray(data["raw_index"], dtype="int64"),
            "anomaly": np.asarray(data["anomaly"], dtype="int8"),
            "site_id": np.asarray(data["site_id"], dtype="int8"),
            "building_id": np.asarray(data["building_id"], dtype="int16"),
        }

    # Validate every requested batch before loading the frame, so a bad plan
    # fails in seconds instead of after the multi-minute load.
    selections = {
        entry["batch"]: select_batch_rows(entry, plan, canonical) for entry in entries
    }

    frame = load_m3_frame(verbose=True)

    for entry in entries:
        batch = entry["batch"]
        selected = selections[batch]
        prove_row_identity(frame, canonical, selected)
        out_root = out_root_for(batch, args.context_rows)
        boundary = entry["head_rows"]
        specs = {
            "head": (0, boundary, "forward"),
            "tail": (boundary, len(selected), "reverse"),
        }
        for shard, (start, end, direction) in specs.items():
            manifest = write_shard(
                batch,
                shard,
                direction,
                selected[start:end],
                start,
                end,
                canonical,
                frame,
                feature_names,
                scaler,
                source_work,
                fit_manifest,
                out_root,
                args.block_rows,
                args.force,
                args.context_rows,
                args.remote_prefix,
            )
            print(
                f"batch {batch} {shard}: {manifest['rows']:,} rows, "
                f"{manifest['anomalies']:,} anomalies, sites {manifest['sites']}, "
                f"-> {manifest['remote_root']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
