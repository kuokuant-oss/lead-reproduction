"""Run one TabPFN cell from the same M5 building-ladder manifest as trees.

TabPFN consumes all available even-building rows in the K cell as in-context
examples.  It has no task-specific epoch loop and therefore no early stopping.
Default ``plan`` mode performs no fit.  ``validation`` uses the deterministic
fake model with explicit row caps; ``formal`` requires a clean worktree and CUDA.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from lead import PROC, ROOT, load_m3_frame, write_json_with_provenance
from m5_building_curve_protocol import (
    SAMPLING_PROFILE,
    int_array_sha256,
    manifest_building_seed,
    resolve_cell_indices,
)
from run_m5_tabpfn_canonical_full_test import (
    DEFAULT_SITE_PREDICTIONS,
    atomic_joblib_dump,
    atomic_save_fitted_model,
    create_real_model,
)
from run_m5_tabpfn_single_context_scaling import (
    FakeTabPFNClassifier,
    atomic_write_npz,
    evaluation_metrics,
    verify_fitted_context,
)
from run_m5_tree_ensemble_matched_context import (
    build_features_keeping_index,
    feature_columns,
)


def default_model_path() -> Path:
    cache = os.environ.get("TABPFN_MODEL_CACHE_DIR")
    return (Path(cache) if cache else ROOT / ".tabpfn-cache") / (
        "tabpfn-v3-classifier-v3_default.ckpt"
    )


def _building_seed_tag(path: Path) -> str:
    if path.is_file():
        try:
            seed = manifest_building_seed(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            seed = None
        if seed is not None:
            return f"building_seed{seed}"
    return "building_seed_unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building-manifest", type=Path, required=True)
    parser.add_argument("--building-budget", type=int, required=True)
    parser.add_argument("--features", type=int, choices=(17, 137), default=137)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument(
        "--model-seed", "--seed", dest="model_seed", type=int, default=42
    )
    parser.add_argument("--experiment-version", choices=("m5_building_count_v2",))
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--query-microbatch-size", type=int, default=4096)
    parser.add_argument("--checkpoint-rows", type=int, default=20_000)
    parser.add_argument(
        "--mode", choices=("plan", "validation", "formal"), default="plan"
    )
    parser.add_argument("--max-context-rows", type=int)
    parser.add_argument("--max-holdout-rows", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-root", type=Path)
    args = parser.parse_args(argv)
    if args.building_budget <= 0 or args.n_estimators <= 0:
        raise ValueError("building budget and n-estimators must be positive")
    if (
        args.query_microbatch_size <= 0
        or args.checkpoint_rows < args.query_microbatch_size
    ):
        raise ValueError("invalid query/checkpoint row sizes")
    if args.mode == "validation":
        if args.max_context_rows is None or args.max_holdout_rows is None:
            raise ValueError("validation mode requires context and holdout row caps")
    elif args.max_context_rows is not None or args.max_holdout_rows is not None:
        raise ValueError("row caps are only allowed in validation mode")
    tag = f"{_building_seed_tag(args.building_manifest)}_k{args.building_budget}_f{args.features}"
    if args.out_root is None:
        base = PROC / "m5_building_curve"
        if args.experiment_version == "m5_building_count_v2":
            base = base / "v2"
        args.out_root = (
            base / "NON_SCIENTIFIC_VALIDATION" / f"tabpfn_{tag}"
            if args.mode == "validation"
            else base / "formal" / f"tabpfn_{tag}"
        )
    return args


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _bounded_rows(
    frame: Any, rows: np.ndarray, cap: int | None, *, seed: int
) -> np.ndarray:
    values = np.asarray(rows, dtype="int64")
    if cap is None or len(values) <= cap:
        return values
    labels = frame.loc[values, "anomaly"].to_numpy(dtype="int8")
    per_class = cap // 2
    selected: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        candidates = values[labels == label]
        if per_class < 1 or len(candidates) < per_class:
            raise ValueError(f"bounded validation lacks class {label} support")
        selected.append(rng.permutation(candidates)[:per_class])
    return np.column_stack(selected).reshape(-1).astype("int64", copy=False)


def _formal_gate(model_path: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise SystemExit("formal run requires a clean committed implementation")
    if not model_path.is_file():
        raise SystemExit(f"TabPFN checkpoint is missing: {model_path}")


def _predict(model: Any, matrix: np.ndarray, batch_size: int) -> np.ndarray:
    output = np.empty(len(matrix), dtype="float32")
    for start in range(0, len(matrix), batch_size):
        end = min(len(matrix), start + batch_size)
        output[start:end] = np.asarray(
            model.predict_proba(matrix[start:end])[:, 1], dtype="float32"
        )
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(args.building_manifest.read_text(encoding="utf-8"))
    building_seed = manifest_building_seed(manifest)
    seeded_sensitivity = manifest.get("experiment") in {
        "m5_building_candidate_sensitivity_pilot",
        "m5_building_source_sampling_sensitivity",
    }
    if args.experiment_version == "m5_building_count_v2":
        if manifest.get("sampling_profile") != SAMPLING_PROFILE:
            raise SystemExit(f"V2 requires sampling_profile={SAMPLING_PROFILE!r}")
    if seeded_sensitivity or args.experiment_version == "m5_building_count_v2":
        if building_seed is None:
            raise SystemExit("seeded manifest lacks building_seed identity")
        if f"building_seed{building_seed}" not in str(args.out_root):
            raise SystemExit("seeded out-root must contain its building_seed identity")
    cell = manifest.get("cells", {}).get(str(args.building_budget))
    if cell is None:
        raise SystemExit(f"manifest has no K={args.building_budget} cell")
    if args.features != 137:
        raise SystemExit(
            "building-count TabPFN cells use the preserved 137-feature pipeline"
        )
    print(
        f"K={args.building_budget}: TabPFN context buildings="
        f"{len(cell['available_buildings'])}; early stopping=not applicable",
        flush=True,
    )
    if args.mode == "plan":
        print("Plan mode: no frame load, TabPFN import, fit, or prediction.")
        return 0
    if args.mode == "formal":
        _formal_gate(args.model_path)
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("formal TabPFN building cell requires CUDA")

    args.out_root.mkdir(parents=True, exist_ok=True)
    frame = load_m3_frame(verbose=True)
    train_mask = frame["building_id"].mod(2).eq(0).to_numpy()
    resolved = resolve_cell_indices(
        frame.loc[train_mask],
        manifest,
        args.building_budget,
        require_role_class_coverage=(
            args.experiment_version != "m5_building_count_v2"
        ),
    )
    context_index = _bounded_rows(
        frame,
        resolved["available_rows"],
        args.max_context_rows,
        seed=args.model_seed + 1,
    )
    if frame.loc[context_index, "building_id"].mod(2).any():
        raise AssertionError("TabPFN context contains an odd holdout building")

    canonical_path = DEFAULT_SITE_PREDICTIONS
    if not canonical_path.is_file():
        canonical_path = (
            ROOT.parent
            / "lead-reproduction"
            / "data"
            / "processed"
            / DEFAULT_SITE_PREDICTIONS.name
        )
    if not canonical_path.is_file():
        raise SystemExit(f"canonical holdout artifact is missing: {canonical_path}")
    with np.load(canonical_path) as canonical:
        holdout_index = np.asarray(canonical["validation_raw_index"], dtype="int64")
        holdout_y = np.asarray(canonical["anomaly"], dtype="int8")
        holdout_site = np.asarray(canonical["site_id"], dtype="int8")
        holdout_building = np.asarray(canonical["building_id"], dtype="int16")
    if args.max_holdout_rows is not None:
        holdout_index = _bounded_rows(
            frame, holdout_index, args.max_holdout_rows, seed=args.model_seed + 2
        )
        holdout_y = frame.loc[holdout_index, "anomaly"].to_numpy(dtype="int8")
        holdout_site = frame.loc[holdout_index, "site_id"].to_numpy(dtype="int8")
        holdout_building = frame.loc[holdout_index, "building_id"].to_numpy(
            dtype="int16"
        )
    if frame.loc[holdout_index, "building_id"].mod(2).eq(0).any():
        raise AssertionError("TabPFN holdout contains an even training building")
    holdout_meter = frame.loc[holdout_index, "meter"].to_numpy(dtype="int8")

    provenance = {
        "mode": "FORMAL" if args.mode == "formal" else "NON_SCIENTIFIC_VALIDATION",
        "manifest": str(args.building_manifest.resolve()),
        "manifest_sha256": _sha256_file(args.building_manifest),
        "sampling_profile": manifest["sampling_profile"],
        "row_policy": manifest.get("row_policy", "all_rows"),
        "average_rows_per_building_limit": manifest.get(
            "average_rows_per_building_limit"
        ),
        "context_limit": manifest.get("max_context_rows"),
        "building_budget": args.building_budget,
        "features": args.features,
        "building_seed": building_seed,
        "row_seed": manifest.get("row_seed", manifest.get("row_selection_seed")),
        "role_seed": manifest.get("role_seed"),
        "model_seed": args.model_seed,
        "seed": args.model_seed,
        "n_estimators": args.n_estimators,
        "query_microbatch_size": args.query_microbatch_size,
        "context_row_sha256": int_array_sha256(context_index),
        "holdout_row_sha256": int_array_sha256(holdout_index),
        "context_rows": int(len(context_index)),
        "holdout_rows": int(len(holdout_index)),
        "model_path": str(args.model_path.resolve()),
        "model_sha256": _sha256_file(args.model_path)
        if args.mode == "formal"
        else None,
        "early_stopping": "not_applicable_in_context_learning_no_weight_updates",
        "caps": {
            "context": args.max_context_rows,
            "holdout": args.max_holdout_rows,
        },
    }
    if args.experiment_version is not None:
        provenance.update(
            {
                "experiment_version": args.experiment_version,
                "training_sampling": "exact_manifest_available_rows_no_resampling",
                "class_ratio_policy": "natural_prevalence_of_manifest_available_rows",
            }
        )
    provenance_path = args.out_root / "provenance.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text(encoding="utf-8")) != provenance:
            raise AssertionError("result-affecting provenance differs from saved run")
    else:
        _atomic_json(provenance_path, provenance)

    print("Building timestamp-merge features for even context buildings", flush=True)
    selected_buildings = resolved["available_buildings"]
    selected_mask = frame["building_id"].isin(selected_buildings)
    train_features = build_features_keeping_index(frame.loc[selected_mask].copy())
    columns = feature_columns(args.features, list(train_features.columns))
    x_context = train_features.loc[context_index, columns].to_numpy(dtype="float32")
    y_context = frame.loc[context_index, "anomaly"].to_numpy(dtype="int64")
    del train_features
    gc.collect()

    scaler_path = args.out_root / "scaler.joblib"
    fake_path = args.out_root / "fake_model.joblib"
    real_path = args.out_root / "model.tabpfn_fit"
    saved_model = real_path if args.mode == "formal" else fake_path
    if args.resume and saved_model.exists() and scaler_path.exists():
        scaler = joblib.load(scaler_path)
        if args.mode == "formal":
            from tabpfn.model_loading import load_fitted_tabpfn_model

            model = load_fitted_tabpfn_model(saved_model, device="cuda")
        else:
            model = joblib.load(saved_model)
        fit_action = "loaded"
    else:
        scaler = StandardScaler()
        x_context = scaler.fit_transform(x_context).astype("float32", copy=False)
        model = (
            create_real_model(args.model_path, args.model_seed, args.n_estimators)
            if args.mode == "formal"
            else FakeTabPFNClassifier()
        )
        fit_started = time.perf_counter()
        model.fit(x_context, y_context)
        fit_seconds = time.perf_counter() - fit_started
        atomic_joblib_dump(scaler, scaler_path)
        if args.mode == "formal":
            atomic_save_fitted_model(model, real_path)
        else:
            atomic_joblib_dump(model, fake_path)
        fit_action = "fitted"
        _atomic_json(args.out_root / "fit.json", {"fit_seconds": fit_seconds})
    context_verification = verify_fitted_context(
        model,
        len(context_index),
        requested_estimators=args.n_estimators if args.mode == "formal" else 1,
    )
    if context_verification["status"] != "verified":
        raise AssertionError(
            "TabPFN fitted state failed effective-context verification"
        )
    del x_context, y_context
    gc.collect()

    print("Building timestamp-merge features for odd canonical holdout", flush=True)
    holdout_features = build_features_keeping_index(frame.loc[~train_mask].copy())
    del frame
    gc.collect()
    spans = [
        (start, min(len(holdout_index), start + args.checkpoint_rows))
        for start in range(0, len(holdout_index), args.checkpoint_rows)
    ]
    chunks = args.out_root / "prediction_chunks"
    started = time.perf_counter()
    for unit, (start, end) in enumerate(spans):
        path = chunks / f"rows_{start:09d}_{end:09d}.npz"
        if path.exists():
            with np.load(path) as stored:
                if not np.array_equal(
                    stored["validation_raw_index"], holdout_index[start:end]
                ):
                    raise AssertionError(
                        f"prediction checkpoint identity drifted: {path}"
                    )
        else:
            block = scaler.transform(
                holdout_features.loc[holdout_index[start:end], columns].to_numpy(
                    dtype="float32"
                )
            )
            score = _predict(model, block, args.query_microbatch_size)
            atomic_write_npz(
                path,
                validation_raw_index=holdout_index[start:end],
                tabpfn=score,
            )
            del block, score
            gc.collect()
        _atomic_json(
            args.out_root / "heartbeat.json",
            {
                "status": "running",
                "completed_units": unit + 1,
                "total_units": len(spans),
                "current_unit": path.name,
                "timestamp": time.time(),
            },
        )
    del holdout_features
    score = np.empty(len(holdout_index), dtype="float32")
    for start, end in spans:
        with np.load(chunks / f"rows_{start:09d}_{end:09d}.npz") as stored:
            score[start:end] = stored["tabpfn"]
    predictions_path = args.out_root / "predictions.npz"
    atomic_write_npz(
        predictions_path,
        validation_raw_index=holdout_index,
        anomaly=holdout_y,
        building_id=holdout_building,
        site_id=holdout_site,
        meter=holdout_meter,
        tabpfn=score,
    )
    result_path = args.out_root / "cell.json"
    write_json_with_provenance(
        result_path,
        {
            "schema_version": 2 if args.experiment_version else 1,
            "experiment": (
                "m5_building_count_v2_tabpfn_cell"
                if args.experiment_version == "m5_building_count_v2"
                else "m5_building_count_curve_tabpfn_cell"
            ),
            **provenance,
            "score_names": ["tabpfn"],
            "available_buildings": len(cell["available_buildings"]),
            "tree_fit_buildings_reference": len(cell["tree_fit_buildings"]),
            "tree_early_stop_buildings_reference": len(
                cell["tree_early_stop_buildings"]
            ),
            "fit_action": fit_action,
            "context_verification": context_verification,
            "metrics": {"tabpfn": evaluation_metrics(holdout_y, score)},
            "predictions": predictions_path.name,
            "elapsed_seconds": time.perf_counter() - started,
        },
        root=ROOT,
    )
    _atomic_json(
        args.out_root / "heartbeat.json",
        {
            "status": "completed",
            "completed_units": len(spans),
            "total_units": len(spans),
            "current_unit": None,
            "timestamp": time.time(),
        },
    )
    _atomic_json(args.out_root / "COMPLETE.json", {"cell": str(result_path)})
    print(f"Wrote {result_path} and {predictions_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
