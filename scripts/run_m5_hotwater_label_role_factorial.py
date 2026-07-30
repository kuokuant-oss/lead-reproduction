"""Resumably fit fixed-query F4 hotwater label-role factorial cells.

The runner consumes only validated manifests produced by
``prepare_m5_hotwater_label_role_factorial.py``.  TabPFN is CUDA-only and runs
both scaler arms; trees use the same rows and run a scaler-invariance pilot
before avoiding redundant frozen-scaler fits.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lead import ROOT, array_sha256, load_m3_frame, validate_context_manifest
from lead.m5_context import query_paths
from run_m5_story_ae_probe import (
    build_feature_matrix,
    load_tree_runner,
    validate_feature_matrix,
)


DEFAULT_ROOT = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"
QUERY_ROOT = ROOT / "data" / "processed" / "m5_context_stories"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tabpfn", "trees"), required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--query-root", type=Path, default=QUERY_ROOT)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / ".tabpfn-cache" / "tabpfn-v3-classifier-v3_default.ckpt",
    )
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--tree-tolerance", type=float, default=1e-6)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def manifests(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    result: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((root / "manifests").glob("seed*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("story") != "hotwater_label_role_factorial":
            continue
        result.append((path, payload))
    if len(result) != 12:
        raise RuntimeError(f"expected 12 factorial manifests, found {len(result)}")
    return result


def result_dir(
    root: Path, model: str, manifest: dict[str, Any], scaler_arm: str
) -> Path:
    return (
        root
        / "predictions"
        / model
        / f"seed{manifest['context_seed']}"
        / manifest["factorial_cell_id"]
        / scaler_arm
    )


def ready(path: Path, *, force: bool) -> bool:
    if (
        not force
        and (path / "predictions.npz").is_file()
        and (path / "result.json").is_file()
    ):
        return True
    return False


def transform(
    scaler: StandardScaler, x_fit: np.ndarray, x_query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return (
        scaler.transform(x_fit).astype("float32", copy=False),
        scaler.transform(x_query).astype("float32", copy=False),
    )


def fit_tree(x_fit: np.ndarray, y_fit: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    runner = load_tree_runner()
    models, _ = runner.fit_frozen_models(x_fit, pd.Series(y_fit))
    return np.mean(
        [
            runner.predict_probability(name, models[name], x_query)
            for name in runner.MODEL_ORDER
        ],
        axis=0,
    ).astype("float32")


def fit_tabpfn(
    x_fit: np.ndarray, y_fit: np.ndarray, x_query: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from tabpfn import TabPFNClassifier

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is prohibited")
    required = {
        "device",
        "random_state",
        "fit_mode",
        "memory_saving_mode",
        "model_path",
    }
    missing = required - set(inspect.signature(TabPFNClassifier).parameters)
    if missing:
        raise RuntimeError("TabPFN API missing: " + ", ".join(sorted(missing)))
    model = TabPFNClassifier(
        n_estimators=args.n_estimators,
        auto_scale_n_estimators=False,
        model_path=str(args.model_path),
        device="cuda",
        random_state=args.model_seed,
        fit_mode="low_memory",
        memory_saving_mode=True,
        keep_cache_on_device=False,
        ignore_pretraining_limits=True,
        n_preprocessing_jobs=1,
        inference_config={"SUBSAMPLE_SAMPLES": None},
        show_progress_bar=False,
    )
    torch.cuda.reset_peak_memory_stats()
    model.fit(x_fit, y_fit)
    fitted_rows = int(getattr(model, "n_train_samples_", -1))
    if fitted_rows != len(y_fit):
        raise AssertionError(f"TabPFN fitted {fitted_rows}, expected {len(y_fit)} rows")
    scores = np.asarray(model.predict_proba(x_query)[:, 1], dtype="float32")
    return scores, {
        "gpu": torch.cuda.get_device_name(0),
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "fitted_rows": fitted_rows,
    }


def main() -> int:
    args = parse_args()
    if args.model == "tabpfn" and not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    resolved = manifests(args.root)
    frame = load_m3_frame(verbose=True)
    query_manifest_path, query_path = query_paths(args.query_root, "screening")
    query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
    with np.load(query_path) as payload:
        query_raw = np.asarray(payload["raw_index"], dtype="int64")
        query_y = np.asarray(payload["anomaly"], dtype="int8")
    if array_sha256(query_raw) != query_manifest["raw_index_sha256"]:
        raise AssertionError("fixed query digest drifted")
    fit_frame = frame.loc[frame["building_id"] % 2 == 0]
    holdout_frame = frame.loc[frame["building_id"] % 2 == 1]
    all_raw = np.unique(
        np.concatenate(
            [np.asarray(item[1]["raw_index"], dtype="int64") for item in resolved]
        )
    )
    print(
        f"building F4 once for {len(all_raw):,} context rows and {len(query_raw):,} fixed-query rows",
        flush=True,
    )
    fit_cache = build_feature_matrix(fit_frame, all_raw, "F4", full_frame=fit_frame)
    query_matrix = build_feature_matrix(
        holdout_frame, query_raw, "F4", full_frame=holdout_frame
    )
    validate_feature_matrix(query_matrix, matrix_name="factorial query")
    matrices: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for path, manifest in resolved:
        validate_context_manifest(frame, manifest)
        raw = np.asarray(manifest["raw_index"], dtype="int64")
        pos = np.searchsorted(all_raw, raw)
        if not np.array_equal(all_raw[pos], raw):
            raise AssertionError("factorial matrix cache lost a context row")
        x = fit_cache[pos]
        y = frame.iloc[raw]["anomaly"].to_numpy(dtype="int8")
        if x.shape != (20_000, 137) or int(y.sum()) != 10_000:
            raise AssertionError("factorial F4 or label-balance contract failed")
        matrices[str(path)] = (x, y, manifest)

    scalers: dict[int, StandardScaler] = {}
    for _, (_, _, manifest) in matrices.items():
        seed = int(manifest["context_seed"])
        if seed in scalers:
            continue
        reference = next(
            (
                item
                for item in matrices.values()
                if int(item[2]["context_seed"]) == seed
                and item[2]["factorial_cell_id"] == "hw_pos_present__hw_neg_present"
            ),
            None,
        )
        if reference is None:
            raise AssertionError(f"seed {seed} has no pooled-reference cell")
        scaler = StandardScaler().fit(reference[0])
        scalers[seed] = scaler
        scaler_path = args.root / "scalers" / f"seed{seed}_pooled_reference.joblib"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    tree_invariant = False
    if args.model == "trees":
        pilot_key = next(
            key
            for key, (_, _, manifest) in matrices.items()
            if int(manifest["context_seed"]) == 42
            and manifest["factorial_cell_id"] == "hw_pos_excluded__hw_neg_excluded"
        )
        x, y, manifest = matrices[pilot_key]
        cell_scaler = StandardScaler().fit(x)
        cell_fit, cell_query = transform(cell_scaler, x, query_matrix)
        cell_scores = fit_tree(cell_fit, y, cell_query)
        frozen_fit, frozen_query = transform(
            scalers[int(manifest["context_seed"])], x, query_matrix
        )
        frozen_scores = fit_tree(frozen_fit, y, frozen_query)
        max_abs = float(np.max(np.abs(cell_scores - frozen_scores)))
        pilot = {
            "seed": 42,
            "cell": manifest["factorial_cell_id"],
            "tolerance": args.tree_tolerance,
            "max_abs_prediction_difference": max_abs,
            "invariant": bool(max_abs <= args.tree_tolerance),
        }
        atomic_json(args.root / "reports" / "tree_scaler_invariance_pilot.json", pilot)
        tree_invariant = bool(pilot["invariant"])
        if not tree_invariant:
            print(
                "tree scaler-invariance pilot failed; retaining both tree scaler arms",
                flush=True,
            )

    for _, (x, y, manifest) in matrices.items():
        arms = (
            (
                ("cell_specific",)
                if tree_invariant
                else ("cell_specific", "frozen_reference")
            )
            if args.model == "trees"
            else ("cell_specific", "frozen_reference")
        )
        for arm in arms:
            destination = result_dir(args.root, args.model, manifest, arm)
            if ready(destination, force=args.force):
                print(
                    f"skip {args.model} seed{manifest['context_seed']} {manifest['factorial_cell_id']} {arm}",
                    flush=True,
                )
                continue
            scaler = (
                StandardScaler().fit(x)
                if arm == "cell_specific"
                else scalers[int(manifest["context_seed"])]
            )
            x_fit, x_query = transform(scaler, x, query_matrix)
            started = time.perf_counter()
            if args.model == "trees":
                scores, extra = (
                    fit_tree(x_fit, y, x_query),
                    {"scaler_invariance_pilot": tree_invariant},
                )
            else:
                scores, extra = fit_tabpfn(x_fit, y, x_query, args)
            if len(scores) != len(query_raw) or not np.isfinite(scores).all():
                raise AssertionError("factorial prediction is invalid")
            atomic_npz(
                destination / "predictions.npz",
                raw_index=query_raw,
                anomaly=query_y,
                score=scores,
                context_raw_index_sha256=np.asarray(manifest["raw_index_sha256"]),
                query_raw_index_sha256=np.asarray(query_manifest["raw_index_sha256"]),
            )
            atomic_json(
                destination / "result.json",
                {
                    "experiment": "m5_hotwater_label_role_factorial",
                    "model": args.model,
                    "scaler_arm": arm,
                    "context_seed": manifest["context_seed"],
                    "model_seed": args.model_seed,
                    "factorial_cell_id": manifest["factorial_cell_id"],
                    "context_raw_index_sha256": manifest["raw_index_sha256"],
                    "query_raw_index_sha256": query_manifest["raw_index_sha256"],
                    "n_estimators": args.n_estimators
                    if args.model == "tabpfn"
                    else None,
                    "elapsed_seconds": time.perf_counter() - started,
                    **extra,
                },
            )
            print(
                f"complete {args.model} seed{manifest['context_seed']} {manifest['factorial_cell_id']} {arm}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
