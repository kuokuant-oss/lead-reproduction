"""Prove the exported 137-feature shards are ready for distributed inference.

The 137-feature line reuses one feature matrix across the n_estimators=1/4/8
shard roots, so a mistake there would silently score the wrong rows on every
line at once. This checker re-proves, for each root:

* the head and tail metadata concatenate back to the canonical row order,
* the feature matrix shape, dtype and digest match the manifest,
* the portable fitted archive differs from its work-dir source only in the
  remote ``model_path``, and carries the estimator count the root claims.

With ``--smoke-rows`` it additionally runs the real portable worker over a small
prefix of the head shard on the local GPU, which is the only step that proves
the fitted state actually predicts. That smoke uses the work-dir fitted state
(local checkpoint path), not the relocated portable one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from lead import PROC, ROOT

DEFAULT_CANONICAL = PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"
WORKER = ROOT / "scripts" / "run_m5_tabpfn_portable_shard.py"
SHARDS = ("head", "tail")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_manifests(root: Path) -> dict[str, dict[str, Any]]:
    manifests = {}
    for name in SHARDS:
        path = root / name / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        manifests[name] = json.loads(path.read_text(encoding="utf-8"))
    return manifests


def check_row_identity(
    root: Path, manifests: dict[str, dict[str, Any]], canonical: Path
) -> dict[str, Any]:
    with np.load(canonical) as site:
        expected = {
            "raw_index": np.asarray(site["validation_raw_index"], dtype="int64"),
            "anomaly": np.asarray(site["anomaly"], dtype="int8"),
            "site_id": np.asarray(site["site_id"], dtype="int8"),
            "building_id": np.asarray(site["building_id"], dtype="int16"),
        }

    parts: dict[str, list[np.ndarray]] = {key: [] for key in expected}
    positions: list[np.ndarray] = []
    for name in SHARDS:
        with np.load(root / name / "metadata.npz") as metadata:
            for key in expected:
                parts[key].append(np.asarray(metadata[key]))
            positions.append(np.asarray(metadata["global_position"], dtype="int64"))
        rows = len(positions[-1])
        if rows != manifests[name]["rows"]:
            raise AssertionError(f"{name}: metadata rows disagree with the manifest")

    observed = {key: np.concatenate(values) for key, values in parts.items()}
    global_position = np.concatenate(positions)
    if not np.array_equal(global_position, np.arange(len(global_position))):
        raise AssertionError("head+tail global positions are not 0..N-1")
    for key, values in expected.items():
        if not np.array_equal(observed[key].astype(values.dtype), values):
            raise AssertionError(f"head+tail {key} does not match the canonical order")

    site_id = observed["site_id"]
    anomaly = observed["anomaly"]
    per_site = {}
    for value in sorted(set(site_id.tolist())):
        mask = site_id == value
        per_site[int(value)] = {
            "rows": int(mask.sum()),
            "anomalies": int(anomaly[mask].sum()),
        }
    return {"rows": len(site_id), "per_site": per_site}


def check_features(root: Path, manifests: dict[str, dict[str, Any]], digests: bool):
    summary = {}
    for name in SHARDS:
        manifest = manifests[name]
        path = root / name / "features.float32.npy"
        matrix = np.load(path, mmap_mode="r")
        if matrix.dtype != np.dtype("float32"):
            raise AssertionError(f"{name}: feature matrix is not float32")
        if list(matrix.shape) != list(manifest["features"]["shape"]):
            raise AssertionError(f"{name}: feature matrix shape drifted")
        if matrix.shape[1] != 137:
            raise AssertionError(f"{name}: expected 137 columns")
        entry = {"shape": list(matrix.shape), "digest_verified": False}
        del matrix
        if digests:
            if sha256_file(path) != manifest["features"]["sha256"]:
                raise AssertionError(f"{name}: feature matrix digest drifted")
            if (
                sha256_file(root / name / "metadata.npz")
                != manifest["metadata"]["sha256"]
            ):
                raise AssertionError(f"{name}: metadata digest drifted")
            entry["digest_verified"] = True
        summary[name] = entry
    return summary


def check_portable_archive(root: Path, manifests: dict[str, dict[str, Any]]):
    summary = {}
    for name in SHARDS:
        manifest = manifests[name]
        portable = root / name / "model.portable.tabpfn_fit"
        source = Path(manifest["fit_state"]["source_work_dir"]) / "model.tabpfn_fit"
        with zipfile.ZipFile(portable) as archive:
            portable_members = {n: archive.read(n) for n in archive.namelist()}
        with zipfile.ZipFile(source) as archive:
            source_members = {n: archive.read(n) for n in archive.namelist()}
        if set(portable_members) != set(source_members):
            raise AssertionError(f"{name}: portable archive member set differs")
        for member in portable_members:
            if member == "init_params.json":
                continue
            if portable_members[member] != source_members[member]:
                raise AssertionError(f"{name}: portable archive changed {member}")
        portable_params = json.loads(portable_members["init_params.json"])
        source_params = json.loads(source_members["init_params.json"])
        differing = {
            key
            for key in set(portable_params) | set(source_params)
            if portable_params.get(key) != source_params.get(key)
        }
        if differing != {"model_path"}:
            raise AssertionError(
                f"{name}: portable init_params differ in {sorted(differing)}"
            )
        declared = manifest["fit_state"]["n_estimators"]
        if int(portable_params.get("n_estimators", 1)) != declared:
            raise AssertionError(
                f"{name}: archive estimator count differs from manifest"
            )
        summary[name] = {
            "n_estimators": declared,
            "remote_model_path": portable_params["model_path"],
        }
    return summary


def run_smoke(
    root: Path,
    manifests: dict[str, dict[str, Any]],
    rows: int,
    microbatch: int,
) -> dict[str, Any]:
    """Predict a small head prefix with the real worker on the local GPU."""
    manifest = manifests["head"]
    source_work = Path(manifest["fit_state"]["source_work_dir"])
    features = np.load(root / "head" / "features.float32.npy", mmap_mode="r")
    rows = min(rows, len(features))

    with tempfile.TemporaryDirectory() as directory:
        mini = Path(directory)
        np.save(mini / "features.float32.npy", np.ascontiguousarray(features[:rows]))
        with np.load(root / "head" / "metadata.npz") as metadata:
            np.savez(
                mini / "metadata.npz",
                **{key: np.asarray(metadata[key])[:rows] for key in metadata.files},
            )
        command = [
            sys.executable,
            str(WORKER),
            "--features",
            str(mini / "features.float32.npy"),
            "--metadata",
            str(mini / "metadata.npz"),
            "--fit-state",
            str(source_work / "model.tabpfn_fit"),
            "--work-dir",
            str(mini / "work"),
            "--context-rows",
            "100000",
            "--n-features",
            "137",
            "--n-estimators",
            str(manifest["fit_state"]["n_estimators"]),
            "--query-microbatch-size",
            str(microbatch),
            "--min-query-microbatch-size",
            "64",
            "--checkpoint-rows",
            str(max(microbatch, rows // 2)),
            "--direction",
            "forward",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        result_path = mini / "work" / "result.json"
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.is_file()
            else {"status": "missing", "stderr": completed.stderr[-2000:]}
        )
        if result.get("status") != "completed":
            return {"status": "failed", "rows": rows, "detail": result}

        scores = []
        labels = []
        for chunk in sorted((mini / "work" / "chunks").glob("*.npz")):
            with np.load(chunk) as payload:
                scores.append(np.asarray(payload["score"]))
                labels.append(np.asarray(payload["anomaly"]))
        score = np.concatenate(scores)
        label = np.concatenate(labels)
        if len(score) != rows:
            raise AssertionError("smoke checkpoints do not cover the prefix")
        if not np.isfinite(score).all():
            raise AssertionError("smoke produced non-finite scores")
        summary = {
            "status": "completed",
            "rows": rows,
            "distinct_scores": int(len(np.unique(score))),
            "score_min": float(score.min()),
            "score_max": float(score.max()),
            "positives": int(label.sum()),
            "effective_microbatch_size": result["effective_microbatch_size"],
            "rows_per_second": rows / result["elapsed_seconds_this_session"],
        }
        if len(set(label.tolist())) == 2:
            from sklearn.metrics import roc_auc_score

            summary["prefix_roc_auc"] = float(roc_auc_score(label, score))
        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--skip-digests", action="store_true")
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=0,
        help="predict this many head rows locally; 0 skips the smoke",
    )
    parser.add_argument("--smoke-microbatch", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report: dict[str, Any] = {}
    for root in args.roots:
        print(f"=== {root} ===", flush=True)
        manifests = load_manifests(root)
        identity = check_row_identity(root, manifests, args.canonical)
        features = check_features(root, manifests, not args.skip_digests)
        archives = check_portable_archive(root, manifests)
        entry: dict[str, Any] = {
            "identity": identity,
            "features": features,
            "portable_archives": archives,
        }
        if args.smoke_rows > 0:
            entry["smoke"] = run_smoke(
                root, manifests, args.smoke_rows, args.smoke_microbatch
            )
        report[str(root)] = entry
        print(json.dumps(entry, indent=2, sort_keys=True), flush=True)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
