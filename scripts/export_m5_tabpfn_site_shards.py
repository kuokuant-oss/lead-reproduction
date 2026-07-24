"""Export per-site 17-feature portable shards for the estimator sweep.

The sweep scores one site at a time, so each shard covers that site's rows
inside the canonical 50/50 building holdout -- the same rows the merged official
prediction artifact holds for that site, in the same canonical order. Rows are
split into a head and a tail half on a 20,000-row checkpoint boundary so two
GPUs can work the site in parallel, exactly as the official dual-shard run does.

Only the fitted state differs between the n=4 and n=8 roots of one site, so
``--reuse-features-from`` hard-links the already-scaled matrix after re-proving
its digest, its context and its scaler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import numpy as np

from lead import BASELINE_FEATURE_COLS, PROC, load_m3_frame

DEFAULT_SITE_PREDICTIONS = (
    PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"
)
# Nearest 20,000-row checkpoint boundary to the midpoint of each site.
DEFAULT_BOUNDARIES = {1: 140_000, 2: 640_000, 3: 600_000}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def default_source_work(n_estimators: int) -> Path:
    suffix = "" if n_estimators == 1 else f"_n{n_estimators}"
    return PROC / f"m5_tabpfn_canonical_full_test_context100000{suffix}.work"


def default_out_root(site: int, n_estimators: int) -> Path:
    return PROC / f"m5_tabpfn_site{site}_context100000_n{n_estimators}"


def remote_root_for(
    site: int, shard: str, n_estimators: int, n_features: int
) -> PurePosixPath:
    """Feature count joins the estimator count in the path.

    The 137-feature line scores the same rows as the 17-feature line, so without
    this both would deploy to one remote work dir and --resume would accept the
    other line's checkpoints as finished work.
    """
    suffix = "" if n_features == 17 else f"_f{n_features}"
    return PurePosixPath(
        f"/content/lead_tabpfn_s{site}_{shard}{suffix}_n{n_estimators}"
    )


def remote_root(site: int, shard: str, n_estimators: int) -> PurePosixPath:
    """The estimator count must be in the path.

    Two estimator values share the same rows, so without it both would deploy to
    one remote work dir and --resume would accept the other value's checkpoints
    as already-complete work -- returning n=4 scores under an n=8 label. The
    n=4 shards already deployed keep the unsuffixed root recorded in their own
    manifests, which stay the source of truth for their recovery.
    """
    return PurePosixPath(f"/content/lead_tabpfn_s{site}_{shard}_n{n_estimators}")


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


def link_or_copy(source: Path, destination: Path) -> str:
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
        mode = "hardlink"
    except OSError:
        shutil.copyfile(source, temporary)
        mode = "copy"
    os.replace(temporary, destination)
    return mode


def verify_reuse_compatibility(
    reuse_root: Path, fit_manifest: dict[str, Any], scaler: Any, site: int
) -> None:
    source_manifest = json.loads(
        (reuse_root / "head" / "manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest["site"] != site:
        raise AssertionError("reuse source covers a different site")
    source_fit_state = source_manifest["fit_state"]
    if source_fit_state["context_sha256"] != fit_manifest["context_sha256"]:
        raise AssertionError("reuse source was fit on a different context")
    source_work = Path(source_fit_state["source_work_dir"])
    source_fit_manifest = json.loads(
        (source_work / "fit_manifest.json").read_text(encoding="utf-8")
    )
    if list(source_fit_manifest["feature_names"]) != list(
        fit_manifest["feature_names"]
    ):
        raise AssertionError("reuse source has a different feature order")
    source_scaler = joblib.load(source_work / "scaler.joblib")
    for attribute in ("mean_", "scale_"):
        if not np.array_equal(
            getattr(source_scaler, attribute), getattr(scaler, attribute)
        ):
            raise AssertionError(f"reuse source scaler {attribute} differs")


def reuse_shard_inputs(
    shard: str,
    reuse_root: Path,
    features_path: Path,
    metadata_path: Path,
    rows: int,
) -> dict[str, Any]:
    source_dir = reuse_root / shard
    source_manifest = json.loads(
        (source_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest["rows"] != rows:
        raise AssertionError(f"{shard}: reuse source row count differs")
    source_features = source_dir / "features.float32.npy"
    source_metadata = source_dir / "metadata.npz"
    if sha256_file(source_features) != source_manifest["features"]["sha256"]:
        raise AssertionError(f"{shard}: reuse source feature digest drifted")
    if sha256_file(source_metadata) != source_manifest["metadata"]["sha256"]:
        raise AssertionError(f"{shard}: reuse source metadata digest drifted")
    mode = link_or_copy(source_features, features_path)
    link_or_copy(source_metadata, metadata_path)
    return {"root": str(source_dir.resolve()), "mode": mode}


def load_full_test_slice(
    slice_root: Path, site: int, n_features: int
) -> tuple[np.ndarray, list[Any]]:
    """Locate this site's rows inside a prebuilt full-test feature matrix.

    The 137-feature matrix over the whole holdout already exists and costs about
    a quarter hour of value-change feature building to reproduce, so the per-site
    shard selects rows out of it instead of rebuilding. Returns the canonical
    positions of this site's rows plus the open memmaps to read them from.
    """
    sources = []
    positions: list[np.ndarray] = []
    site_ids: list[np.ndarray] = []
    raw_index: list[np.ndarray] = []
    offset = 0
    for shard in ("head", "tail"):
        manifest = json.loads(
            (slice_root / shard / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest["n_features"] != n_features:
            raise AssertionError(
                f"slice source has {manifest['n_features']} features, "
                f"fitted state has {n_features}"
            )
        matrix = np.load(slice_root / shard / "features.float32.npy", mmap_mode="r")
        with np.load(slice_root / shard / "metadata.npz") as metadata:
            shard_sites = np.asarray(metadata["site_id"])
            shard_raw = np.asarray(metadata["raw_index"], dtype="int64")
        if len(matrix) != len(shard_sites):
            raise AssertionError(f"{shard}: slice source rows disagree with metadata")
        sources.append((offset, matrix))
        site_ids.append(shard_sites)
        raw_index.append(shard_raw)
        positions.append(np.arange(offset, offset + len(matrix), dtype="int64"))
        offset += len(matrix)

    all_sites = np.concatenate(site_ids)
    all_positions = np.concatenate(positions)
    mask = all_sites == site
    return all_positions[mask], sources


def read_sliced_rows(
    sources: list[Any], wanted: np.ndarray, n_features: int
) -> np.ndarray:
    """Gather the requested canonical positions out of the shard memmaps."""
    out = np.empty((len(wanted), n_features), dtype="float32")
    for offset, matrix in sources:
        local = wanted - offset
        inside = (local >= 0) & (local < len(matrix))
        if not inside.any():
            continue
        out[inside] = matrix[local[inside]]
    return out


def write_shard(
    shard: str,
    site: int,
    start: int,
    end: int,
    direction: str,
    site_rows: dict[str, np.ndarray],
    frame: Any,
    feature_names: list[str],
    scaler: Any,
    source_work: Path,
    fit_manifest: dict[str, Any],
    out_root: Path,
    block_rows: int,
    force: bool,
    reuse_root: Path | None,
    slice_sources: list[Any] | None = None,
    slice_positions: np.ndarray | None = None,
) -> dict[str, Any]:
    out_dir = out_root / shard
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "features.float32.npy"
    metadata_path = out_dir / "metadata.npz"
    portable_fit_path = out_dir / "model.portable.tabpfn_fit"
    manifest_path = out_dir / "manifest.json"
    if not force and any(
        p.exists()
        for p in (features_path, metadata_path, portable_fit_path, manifest_path)
    ):
        raise FileExistsError(f"{shard} export exists; pass --force to replace")

    raw_index = site_rows["raw_index"][start:end]
    y = site_rows["anomaly"][start:end]
    site_id = site_rows["site_id"][start:end]
    building_id = site_rows["building_id"][start:end]
    global_position = np.arange(start, end, dtype="int64")

    reused: dict[str, Any] | None = None
    if reuse_root is not None:
        reused = reuse_shard_inputs(
            shard, reuse_root, features_path, metadata_path, len(raw_index)
        )
    else:
        temporary_features = features_path.with_name(features_path.name + ".tmp")
        matrix = np.lib.format.open_memmap(
            temporary_features,
            mode="w+",
            dtype="float32",
            shape=(len(raw_index), len(feature_names)),
        )
        for s in range(0, len(raw_index), block_rows):
            e = min(len(raw_index), s + block_rows)
            if slice_sources is not None:
                # Rows come from a prebuilt full-test matrix that was already
                # scaled by this same context scaler, so no transform here.
                matrix[s:e] = read_sliced_rows(
                    slice_sources,
                    slice_positions[start + s : start + e],
                    len(feature_names),
                )
            else:
                block = frame.loc[raw_index[s:e], feature_names].to_numpy(
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

    remote_checkpoint = (
        remote_root_for(
            site, shard, int(fit_manifest["n_estimators"]), len(feature_names)
        )
        / "tabpfn-v3-classifier-v3_default.ckpt"
    )
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
        "site": site,
        "shard": shard,
        "direction": direction,
        "n_features": len(feature_names),
        "site_start": start,
        "site_end": end,
        "rows": len(raw_index),
        "anomalies": int(y.sum()),
        "min_building_id": int(building_id[0]),
        "max_building_id": int(building_id[-1]),
        "remote_root": str(
            remote_root_for(
                site, shard, int(fit_manifest["n_estimators"]), len(feature_names)
            )
        ),
        "reused_inputs": reused,
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
            "n_estimators": int(fit_manifest["n_estimators"]),
            "source_work_dir": str(source_work.resolve()),
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
    parser.add_argument("--site", type=int, required=True)
    parser.add_argument("--n-estimators", type=int, required=True)
    parser.add_argument("--boundary", type=int, default=None)
    parser.add_argument("--source-work-dir", type=Path, default=None)
    parser.add_argument(
        "--site-predictions", type=Path, default=DEFAULT_SITE_PREDICTIONS
    )
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--reuse-features-from", type=Path, default=None)
    parser.add_argument(
        "--slice-from",
        type=Path,
        default=None,
        help="full-test shard root to take this site's already-scaled rows from",
    )
    parser.add_argument("--export-block-rows", type=int, default=100_000)
    parser.add_argument("--shard", choices=("head", "tail", "both"), default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    boundary = args.boundary or DEFAULT_BOUNDARIES.get(args.site)
    if boundary is None:
        raise ValueError(f"no default boundary for site {args.site}; pass --boundary")
    if boundary % 20_000:
        raise ValueError("boundary must land on a 20,000-row checkpoint boundary")

    source_work = args.source_work_dir or default_source_work(args.n_estimators)
    out_root = args.out_root or default_out_root(args.site, args.n_estimators)
    fit_manifest = json.loads(
        (source_work / "fit_manifest.json").read_text(encoding="utf-8")
    )
    if int(fit_manifest.get("n_estimators", 1)) != args.n_estimators:
        raise AssertionError(
            f"{source_work} was fit with n_estimators="
            f"{fit_manifest.get('n_estimators', 1)}"
        )
    feature_names = list(fit_manifest["feature_names"])
    # The 17-feature line must be exactly the baseline columns. Wider feature
    # sets (the 137-feature line) are only allowed when their rows are sliced
    # out of a matrix that was already built and scaled for that same fit.
    if len(feature_names) == 17:
        if feature_names != list(BASELINE_FEATURE_COLS):
            raise AssertionError("fitted state does not use the 17 baseline features")
    elif args.slice_from is None:
        raise ValueError(
            f"fitted state has {len(feature_names)} features; pass --slice-from"
        )
    scaler = joblib.load(source_work / "scaler.joblib")

    with np.load(args.site_predictions) as site:
        canonical = {
            "raw_index": np.asarray(site["validation_raw_index"], dtype="int64"),
            "anomaly": np.asarray(site["anomaly"], dtype="int8"),
            "site_id": np.asarray(site["site_id"], dtype="int8"),
            "building_id": np.asarray(site["building_id"], dtype="int16"),
        }
    mask = canonical["site_id"] == args.site
    site_rows = {key: value[mask] for key, value in canonical.items()}
    rows = len(site_rows["raw_index"])
    if rows == 0:
        raise AssertionError(f"site {args.site} has no canonical rows")
    if boundary >= rows:
        raise ValueError(f"boundary {boundary} is not inside site {args.site}")
    print(
        f"site {args.site}: {rows} rows, {int(site_rows['anomaly'].sum())} anomalies, "
        f"head [0, {boundary}) tail [{boundary}, {rows})",
        flush=True,
    )

    frame = None
    slice_sources = None
    slice_positions = None
    if args.reuse_features_from is not None:
        verify_reuse_compatibility(
            args.reuse_features_from, fit_manifest, scaler, args.site
        )
    elif args.slice_from is not None:
        slice_positions, slice_sources = load_full_test_slice(
            args.slice_from, args.site, len(feature_names)
        )
        if len(slice_positions) != rows:
            raise AssertionError(
                f"slice source holds {len(slice_positions)} rows for site "
                f"{args.site}, canonical has {rows}"
            )
        print(
            f"slicing {len(slice_positions)} site {args.site} rows from "
            f"{args.slice_from}",
            flush=True,
        )
    else:
        frame = load_m3_frame(verbose=True)
        # Identity proof: the canonical row IDs must still map to the labels,
        # sites and buildings the canonical artifact recorded for them.
        for column in ("anomaly", "site_id", "building_id"):
            observed = frame.loc[site_rows["raw_index"], column].to_numpy(
                dtype=site_rows[column].dtype
            )
            if not np.array_equal(observed, site_rows[column]):
                raise AssertionError(f"raw row IDs do not map to canonical {column}")

    specs = {
        "head": (0, boundary, "forward"),
        "tail": (boundary, rows, "reverse"),
    }
    names = ("head", "tail") if args.shard == "both" else (args.shard,)
    for shard in names:
        start, end, direction = specs[shard]
        print(f"=== exporting site {args.site} {shard} ===", flush=True)
        manifest = write_shard(
            shard,
            args.site,
            start,
            end,
            direction,
            site_rows,
            frame,
            feature_names,
            scaler,
            source_work,
            fit_manifest,
            out_root,
            args.export_block_rows,
            args.force,
            args.reuse_features_from,
            slice_sources,
            slice_positions,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
