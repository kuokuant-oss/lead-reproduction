"""Exact-design, state-preserving recovery refit for the M5 Path-A factorial.

This runner deliberately writes below ``<root>/recovery`` and never overwrites
the first-pass predictions.  It has no wall-time timeout: completed cell states
and query scores are atomically durable and a later invocation resumes them.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
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
ARMS = ("cell_specific", "frozen_reference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tabpfn", "trees"), required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--recovery-root", type=Path, default=None)
    parser.add_argument(
        "--query", choices=("screening", "independent"), default="screening"
    )
    parser.add_argument("--query-root", type=Path, default=QUERY_ROOT)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / ".tabpfn-cache" / "tabpfn-v3-classifier-v3_default.ckpt",
    )
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--model-seed", type=int, default=42)
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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def manifests(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    items: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((root / "manifests").glob("seed*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("story") == "hotwater_label_role_factorial":
            items.append((path, payload))
    if len(items) != 12:
        raise RuntimeError(f"expected 12 factorial manifests, found {len(items)}")
    return items


def destination(recovery: Path, model: str, manifest: dict[str, Any], arm: str) -> Path:
    return (
        recovery
        / "states"
        / model
        / f"seed{manifest['context_seed']}"
        / manifest["factorial_cell_id"]
        / arm
    )


def environment(model_path: Path) -> dict[str, Any]:
    import sklearn

    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "model_checkpoint": str(model_path),
        "model_checkpoint_sha256": sha256_file(model_path)
        if model_path.is_file()
        else None,
    }
    try:
        import torch
        import tabpfn

        result.update(
            {
                "torch": torch.__version__,
                "tabpfn": getattr(tabpfn, "__version__", "unknown"),
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
            }
        )
    except ImportError:
        result["cuda_available"] = False
    return result


def query_arrays(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    if args.query == "screening":
        manifest_path, path = query_paths(args.query_root, "screening")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(path) as payload:
            raw, y = (
                np.asarray(payload["raw_index"], dtype="int64"),
                np.asarray(payload["anomaly"], dtype="int8"),
            )
        digest = manifest["raw_index_sha256"]
    else:
        path = args.root / "independent_query" / "queries.npz"
        manifest_path = args.root / "independent_query" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(path) as payload:
            raw, y = (
                np.asarray(payload["raw_index"], dtype="int64"),
                np.asarray(payload["anomaly"], dtype="int8"),
            )
        digest = array_sha256(raw)
    if array_sha256(raw) != digest:
        raise AssertionError("query raw-index digest drifted")
    return raw, y, digest, manifest


def initialise_tabpfn(args: argparse.Namespace):
    import torch
    from tabpfn import TabPFNClassifier

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; exact TabPFN recovery cannot use a CPU fallback"
        )
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
    return TabPFNClassifier(
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


def save_tabpfn_state(model: Any, path: Path) -> None:
    from tabpfn.model_loading import save_fitted_tabpfn_model

    path.parent.mkdir(parents=True, exist_ok=True)
    # TabPFN validates the suffix before writing its zip payload.
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    save_fitted_tabpfn_model(model, temporary)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.model == "tabpfn" and not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    recovery = args.recovery_root or args.root / "recovery"
    if (
        args.query == "independent"
        and not (recovery / "reproduction_gate.json").is_file()
    ):
        raise RuntimeError(
            "independent query is locked until the screening-query reproduction gate passes"
        )
    resolved = manifests(args.root)
    raw_query, y_query, query_digest, query_manifest = query_arrays(args)
    frame = load_m3_frame(verbose=True)
    fit_frame, holdout_frame = (
        frame.loc[frame["building_id"] % 2 == 0],
        frame.loc[frame["building_id"] % 2 == 1],
    )
    all_raw = np.unique(
        np.concatenate(
            [np.asarray(payload["raw_index"], dtype="int64") for _, payload in resolved]
        )
    )
    fit_cache = build_feature_matrix(fit_frame, all_raw, "F4", full_frame=fit_frame)
    query_matrix = build_feature_matrix(
        holdout_frame, raw_query, "F4", full_frame=holdout_frame
    )
    validate_feature_matrix(query_matrix, matrix_name=f"factorial {args.query} query")
    cells: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    for _, manifest in resolved:
        validate_context_manifest(frame, manifest)
        raw = np.asarray(manifest["raw_index"], dtype="int64")
        positions = np.searchsorted(all_raw, raw)
        if not np.array_equal(all_raw[positions], raw):
            raise AssertionError("context matrix cache lost ordered rows")
        x, y = fit_cache[positions], frame.iloc[raw]["anomaly"].to_numpy(dtype="int8")
        if x.shape != (20_000, 137) or y.sum() != 10_000:
            raise AssertionError("exact F4/label-balance design drift")
        cells.append((x, y, manifest))
    frozen: dict[int, StandardScaler] = {}
    for x, _, manifest in cells:
        seed = int(manifest["context_seed"])
        if (
            seed not in frozen
            and manifest["factorial_cell_id"] == "hw_pos_present__hw_neg_present"
        ):
            frozen[seed] = StandardScaler().fit(x)
    if set(frozen) != {42, 123, 999}:
        raise AssertionError("pooled-reference scalers missing")
    atomic_json(
        recovery / f"environment_provenance_{args.model}.json",
        environment(args.model_path),
    )
    atomic_json(
        recovery / f"recovery_design_{args.model}.json",
        {
            "manifests": [
                {"path": str(path), "digest": payload["raw_index_sha256"]}
                for path, payload in resolved
            ],
            "query": args.query,
            "query_raw_index_sha256": query_digest,
            "query_manifest": query_manifest,
            "model": args.model,
            "model_seed": args.model_seed,
            "n_estimators": args.n_estimators if args.model == "tabpfn" else None,
            "scaler_arms": list(ARMS),
        },
    )
    for x, y, manifest in cells:
        for arm in ARMS:
            out = destination(recovery, args.model, manifest, arm)
            state = out / (
                "model.tabpfn_fit" if args.model == "tabpfn" else "tree_ensemble.joblib"
            )
            prediction = (
                out / "screening_predictions.npz"
                if args.query == "screening"
                else out / "independent_predictions.npz"
            )
            scaler_path = out / "scaler.joblib"
            if state.is_file() and scaler_path.is_file():
                scaler = joblib.load(scaler_path)
                if args.model == "tabpfn":
                    from tabpfn.model_loading import load_fitted_tabpfn_model

                    model = load_fitted_tabpfn_model(state, device="cuda")
                    scores = np.asarray(
                        model.predict_proba(
                            scaler.transform(query_matrix).astype("float32", copy=False)
                        )[:, 1],
                        dtype="float32",
                    )
                else:
                    saved = joblib.load(state)
                    runner = load_tree_runner()
                    transformed = scaler.transform(query_matrix).astype(
                        "float32", copy=False
                    )
                    scores = np.mean(
                        [
                            runner.predict_probability(
                                name, saved["models"][name], transformed
                            )
                            for name in saved["model_order"]
                        ],
                        axis=0,
                    ).astype("float32")
                action = "loaded"
            else:
                scaler = (
                    StandardScaler().fit(x)
                    if arm == "cell_specific"
                    else frozen[int(manifest["context_seed"])]
                )
                x_fit = scaler.transform(x).astype("float32", copy=False)
                x_eval = scaler.transform(query_matrix).astype("float32", copy=False)
                started = time.perf_counter()
                if args.model == "tabpfn":
                    model = initialise_tabpfn(args)
                    model.fit(x_fit, y)
                    if int(getattr(model, "n_train_samples_", -1)) != len(y):
                        raise AssertionError("TabPFN fitted-row count drifted")
                    save_tabpfn_state(model, state)
                    scores = np.asarray(
                        model.predict_proba(x_eval)[:, 1], dtype="float32"
                    )
                else:
                    runner = load_tree_runner()
                    models, fit_seconds = runner.fit_frozen_models(x_fit, pd.Series(y))
                    atomic_joblib(
                        state,
                        {
                            "models": models,
                            "model_order": list(runner.MODEL_ORDER),
                            "fit_seconds": fit_seconds,
                        },
                    )
                    scores = np.mean(
                        [
                            runner.predict_probability(name, models[name], x_eval)
                            for name in runner.MODEL_ORDER
                        ],
                        axis=0,
                    ).astype("float32")
                atomic_joblib(scaler_path, scaler)
                atomic_json(
                    out / "fit.json",
                    {
                        "action": "fitted",
                        "elapsed_seconds": time.perf_counter() - started,
                        "feature_count": 137,
                        "context_rows": len(y),
                        "context_raw_index_sha256": manifest["raw_index_sha256"],
                        "model_seed": args.model_seed,
                        "scaler_arm": arm,
                    },
                )
                action = "fitted"
            if len(scores) != len(raw_query) or not np.isfinite(scores).all():
                raise AssertionError("invalid recovery scores")
            atomic_npz(
                prediction,
                raw_index=raw_query,
                anomaly=y_query,
                score=scores,
                query_raw_index_sha256=np.asarray(query_digest),
                context_raw_index_sha256=np.asarray(manifest["raw_index_sha256"]),
            )
            atomic_json(
                out / f"{args.query}_result.json",
                {
                    "action": action,
                    "model": args.model,
                    "seed": manifest["context_seed"],
                    "cell": manifest["factorial_cell_id"],
                    "scaler_arm": arm,
                    "state": str(state),
                    "state_sha256": sha256_file(state),
                    "scaler": str(scaler_path),
                    "scaler_sha256": sha256_file(scaler_path),
                    "query": args.query,
                    "query_raw_index_sha256": query_digest,
                },
            )
            print(
                f"{action} {args.model} seed{manifest['context_seed']} {manifest['factorial_cell_id']} {arm} -> {args.query}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
