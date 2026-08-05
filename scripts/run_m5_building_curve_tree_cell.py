"""Run one early-stopped tree cell from an M5 building-ladder manifest.

Default mode is ``plan`` and performs no model fit.  ``validation`` requires
deterministic row and iteration caps and writes NON_SCIENTIFIC_VALIDATION
artifacts.  ``formal`` refuses caps and requires a clean committed worktree;
the operator must invoke it explicitly after implementation review.
"""

from __future__ import annotations

import argparse
import ctypes
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

from lead import PROC, ROOT, SHIFTS, load_m3_frame, write_json_with_provenance
from m5_building_curve_protocol import int_array_sha256, resolve_cell_indices
from m5_tree_early_stopping import (
    MODEL_ORDER,
    ensemble_probabilities,
    fit_early_stopped_models,
    refit_models_at_selected_iterations,
)
from run_m3_figure_observations import evaluation_summary
from run_m5_tree_ensemble_matched_context import (
    CANONICAL_ORDER,
    atomic_write_npz,
    build_features_keeping_index,
    feature_columns,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building-manifest", type=Path, required=True)
    parser.add_argument("--building-budget", type=int, required=True)
    parser.add_argument("--features", type=int, choices=(17, 137), default=137)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict-batch-rows", type=int, default=200_000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--hist-patience", type=int, default=20)
    parser.add_argument(
        "--mode", choices=("plan", "validation", "formal"), default="plan"
    )
    parser.add_argument("--max-fit-rows", type=int)
    parser.add_argument("--max-early-stop-rows", type=int)
    parser.add_argument("--max-holdout-rows", type=int)
    parser.add_argument("--validation-iteration-ceiling", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-root", type=Path)
    args = parser.parse_args(argv)
    if args.building_budget <= 0:
        raise ValueError("building budget must be positive")
    if args.predict_batch_rows <= 0:
        raise ValueError("predict batch rows must be positive")
    caps = (args.max_fit_rows, args.max_early_stop_rows, args.max_holdout_rows)
    if args.mode == "validation":
        if any(value is None or value <= 0 for value in caps):
            raise ValueError("validation mode requires all three positive row caps")
    elif any(value is not None for value in caps):
        raise ValueError("row caps are only allowed in validation mode")
    if args.mode == "formal" and args.validation_iteration_ceiling != 8:
        raise ValueError("formal mode cannot use the validation iteration ceiling")
    tag = (
        f"{args.building_manifest.parent.name}_k{args.building_budget}_f{args.features}"
    )
    if args.out_root is None:
        base = PROC / "m5_building_curve"
        args.out_root = (
            base / "NON_SCIENTIFIC_VALIDATION" / tag
            if args.mode == "validation"
            else base / "formal" / tag
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


def _atomic_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(payload, temporary)
    with temporary.open("rb+") as stream:
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
    if per_class < 1:
        raise ValueError("bounded validation cap is too small for two classes")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in (0, 1):
        candidates = values[labels == label]
        if len(candidates) < per_class:
            raise ValueError(f"bounded rows lack class {label} support")
        selected.append(rng.permutation(candidates)[:per_class])
    output = np.column_stack(selected).reshape(-1)
    return output.astype("int64", copy=False)


def _formal_gate() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise SystemExit("formal run requires a clean committed implementation")


def _write_heartbeat(
    path: Path,
    *,
    status: str,
    completed: int,
    total: int,
    current: str | None,
    started: float,
) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = completed / elapsed
    pending = total - completed
    _atomic_json(
        path,
        {
            "status": status,
            "completed_units": completed,
            "total_units": total,
            "pending_units": pending,
            "current_unit": current,
            "elapsed_seconds": elapsed,
            "units_per_second": rate,
            "eta_seconds": pending / rate if rate else None,
            "timestamp": time.time(),
        },
    )


def _matrix_columns(features: int, frame_columns: list[str]) -> list[str]:
    """Resolve matrix columns without first materializing all 137 features."""
    available = list(frame_columns)
    if features == 137:
        for shift in SHIFTS:
            available.append(f"lag_value_diff_{shift}")
            available.append(f"lag_value_ratio_{shift}")
    return feature_columns(features, available)


def _collect_and_trim() -> None:
    # Keep Python semantics unchanged while returning free glibc arenas to WSL.
    gc.collect()
    try:
        allocator = ctypes.CDLL(None)
        malloc_trim = allocator.malloc_trim
    except (OSError, AttributeError):
        return
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)


def _scale_matrix(scaler: StandardScaler, values: np.ndarray) -> np.ndarray:
    """Scale in place when writable; copy only a read-only input view."""
    writable = values if values.flags.writeable else values.copy()
    return scaler.transform(writable, copy=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(args.building_manifest.read_text(encoding="utf-8"))
    cell = manifest.get("cells", {}).get(str(args.building_budget))
    if cell is None:
        raise SystemExit(f"manifest has no K={args.building_budget} cell")
    if args.features == 17 and args.building_budget != int(
        manifest["candidate_buildings"]
    ):
        raise SystemExit("17 features are reserved for the full-building baseline")
    print(
        f"K={args.building_budget} available buildings="
        f"{len(cell['available_buildings'])}, tree fit/ES="
        f"{len(cell['tree_fit_buildings'])}/{len(cell['tree_early_stop_buildings'])}",
        flush=True,
    )
    if args.mode == "plan":
        print("Plan mode: no frame load, feature construction, fit, or prediction.")
        return 0
    if args.mode == "formal":
        _formal_gate()

    args.out_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frame = load_m3_frame(verbose=True)
    train_mask = frame["building_id"].mod(2).eq(0).to_numpy()
    if frame.loc[train_mask, "building_id"].mod(2).any():
        raise AssertionError("training half contains an odd building")
    resolved = resolve_cell_indices(
        frame.loc[train_mask], manifest, args.building_budget
    )
    fit_index = _bounded_rows(
        frame, resolved["tree_fit_rows"], args.max_fit_rows, seed=args.seed + 1
    )
    early_stop_index = _bounded_rows(
        frame,
        resolved["tree_early_stop_rows"],
        args.max_early_stop_rows,
        seed=args.seed + 2,
    )
    available_index = (
        np.concatenate([fit_index, early_stop_index])
        if args.mode == "validation"
        else resolved["available_rows"]
    )

    canonical_path = CANONICAL_ORDER
    if not canonical_path.is_file():
        canonical_path = (
            ROOT.parent
            / "lead-reproduction"
            / "data"
            / "processed"
            / CANONICAL_ORDER.name
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
            frame, holdout_index, args.max_holdout_rows, seed=args.seed + 3
        )
        holdout_y = frame.loc[holdout_index, "anomaly"].to_numpy(dtype="int8")
        holdout_site = frame.loc[holdout_index, "site_id"].to_numpy(dtype="int8")
        holdout_building = frame.loc[holdout_index, "building_id"].to_numpy(
            dtype="int16"
        )
    if frame.loc[holdout_index, "building_id"].mod(2).eq(0).any():
        raise AssertionError("canonical holdout contains an even training building")
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
        "max_context_rows": manifest.get("max_context_rows"),
        "building_budget": args.building_budget,
        "features": args.features,
        "seed": args.seed,
        "fit_row_sha256": int_array_sha256(fit_index),
        "early_stop_row_sha256": int_array_sha256(early_stop_index),
        "available_row_sha256": int_array_sha256(available_index),
        "holdout_row_sha256": int_array_sha256(holdout_index),
        "caps": {
            "fit": args.max_fit_rows,
            "early_stop": args.max_early_stop_rows,
            "holdout": args.max_holdout_rows,
        },
    }
    provenance_path = args.out_root / "provenance.json"
    if provenance_path.exists():
        observed = json.loads(provenance_path.read_text(encoding="utf-8"))
        if observed != provenance:
            raise AssertionError("result-affecting provenance differs from saved run")
    else:
        _atomic_json(provenance_path, provenance)

    columns = _matrix_columns(args.features, list(frame.columns))
    model_path = args.out_root / "models.joblib"
    fit_path = args.out_root / "fit.json"
    if args.resume and model_path.exists() and fit_path.exists():
        cached = joblib.load(model_path)
        scaler, models = cached["scaler"], cached["models"]
        fit_record = json.loads(fit_path.read_text(encoding="utf-8"))
        if fit_record["fit_row_sha256"] != provenance["fit_row_sha256"]:
            raise AssertionError("cached model fit-row identity drifted")
        if fit_record["final_refit_row_sha256"] != provenance["available_row_sha256"]:
            raise AssertionError("cached model final-refit row identity drifted")
        print("Reused fitted models without rebuilding training features", flush=True)
    else:
        if args.features == 137:
            print(
                "Building timestamp-merge features for even training buildings",
                flush=True,
            )
            selected_buildings = resolved["available_buildings"]
            selected_mask = frame["building_id"].isin(selected_buildings)
            train_features = build_features_keeping_index(frame.loc[selected_mask])
        else:
            print("Using the 17-feature baseline matrix", flush=True)
            train_features = frame.loc[train_mask]
        x_fit = train_features.loc[fit_index, columns].to_numpy(dtype="float32")
        x_early_stop = train_features.loc[early_stop_index, columns].to_numpy(
            dtype="float32"
        )
        y_fit = frame.loc[fit_index, "anomaly"].to_numpy(dtype="int8")
        y_early_stop = frame.loc[early_stop_index, "anomaly"].to_numpy(dtype="int8")
        y_available = frame.loc[available_index, "anomaly"].to_numpy(dtype="int8")
        # Neither source representation is needed during early stopping.
        # Rebuilding features once for final refit costs time but avoids keeping
        # a 5.6 GiB DataFrame beside the 5.1 GiB fit/ES matrices and model
        # working memory.
        del train_features, frame
        _collect_and_trim()
        selection_scaler = StandardScaler()
        selection_scaler.fit(x_fit)
        x_fit = _scale_matrix(selection_scaler, x_fit)
        x_early_stop = _scale_matrix(selection_scaler, x_early_stop)
        ceilings = None
        if args.mode == "validation":
            ceilings = {name: args.validation_iteration_ceiling for name in MODEL_ORDER}
        models, records, contract = fit_early_stopped_models(
            x_fit,
            y_fit,
            x_early_stop,
            y_early_stop,
            seed=args.seed,
            patience=min(args.patience, args.validation_iteration_ceiling)
            if args.mode == "validation"
            else args.patience,
            hist_patience=min(args.hist_patience, args.validation_iteration_ceiling)
            if args.mode == "validation"
            else args.hist_patience,
            ceilings=ceilings,
            checkpoint_dir=args.out_root / "model_checkpoints",
            resume=args.resume,
        )
        del models, x_fit, x_early_stop
        gc.collect()
        frame = load_m3_frame(verbose=True)
        train_mask = frame["building_id"].mod(2).eq(0).to_numpy()
        if args.features == 137:
            print(
                "Rebuilding timestamp-merge features for final refit",
                flush=True,
            )
            selected_buildings = resolved["available_buildings"]
            selected_mask = frame["building_id"].isin(selected_buildings)
            train_features = build_features_keeping_index(frame.loc[selected_mask])
        else:
            train_features = frame.loc[train_mask]
        x_available = train_features.loc[available_index, columns].to_numpy(
            dtype="float32"
        )
        del train_features, frame
        _collect_and_trim()
        scaler = StandardScaler()
        scaler.fit(x_available)
        x_available = _scale_matrix(scaler, x_available)
        models = refit_models_at_selected_iterations(
            x_available,
            y_available,
            records,
            contract,
            checkpoint_dir=args.out_root / "final_model_checkpoints",
            resume=args.resume,
        )
        fit_record = {
            "fit_row_sha256": provenance["fit_row_sha256"],
            "early_stop_row_sha256": provenance["early_stop_row_sha256"],
            "fit_rows": int(len(fit_index)),
            "early_stop_rows": int(len(early_stop_index)),
            "final_refit_rows": int(len(available_index)),
            "final_refit_row_sha256": provenance["available_row_sha256"],
            "final_refit": "all available rows at ES-selected iteration counts",
            "records": records,
            "model_contract": contract,
        }
        _atomic_joblib(model_path, {"scaler": scaler, "models": models})
        _atomic_json(fit_path, fit_record)
        del x_available, y_available
        # Reload only after the fit peak has passed; this preserves the exact
        # source rows while preventing the source frame from coexisting with
        # the largest training matrices and model-library working memory.
        frame = load_m3_frame(verbose=True)
        train_mask = frame["building_id"].mod(2).eq(0).to_numpy()
    gc.collect()

    spans = [
        (start, min(len(holdout_index), start + args.predict_batch_rows))
        for start in range(0, len(holdout_index), args.predict_batch_rows)
    ]
    chunks = args.out_root / "prediction_chunks"
    missing_chunks = any(
        not (chunks / f"rows_{start:09d}_{end:09d}.npz").exists()
        for start, end in spans
    )
    holdout_features = None
    if missing_chunks:
        if args.features == 137:
            print(
                "Building timestamp-merge features for odd canonical holdout",
                flush=True,
            )
            holdout_features = build_features_keeping_index(frame.loc[~train_mask])
        else:
            holdout_features = frame.loc[~train_mask]
    else:
        print(
            "Reused all prediction chunks without rebuilding holdout features",
            flush=True,
        )
    del frame
    gc.collect()
    heartbeat = args.out_root / "heartbeat.json"
    _write_heartbeat(
        heartbeat,
        status="running",
        completed=0,
        total=len(spans),
        current=None,
        started=started,
    )
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
            if holdout_features is None:
                raise AssertionError(
                    "missing prediction chunk without holdout features"
                )
            block = holdout_features.loc[holdout_index[start:end], columns].to_numpy(
                dtype="float32"
            )
            block = _scale_matrix(scaler, block)
            score = ensemble_probabilities(models, block)
            atomic_write_npz(
                path,
                validation_raw_index=holdout_index[start:end],
                **{name: values.astype("float32") for name, values in score.items()},
            )
            del block, score
            gc.collect()
        _write_heartbeat(
            heartbeat,
            status="running",
            completed=unit + 1,
            total=len(spans),
            current=path.name,
            started=started,
        )
    if holdout_features is not None:
        del holdout_features
    scores = {
        name: np.empty(len(holdout_index), dtype="float32")
        for name in (*MODEL_ORDER, "ensemble")
    }
    for start, end in spans:
        path = chunks / f"rows_{start:09d}_{end:09d}.npz"
        with np.load(path) as stored:
            for name in scores:
                scores[name][start:end] = stored[name]

    predictions_path = args.out_root / "predictions.npz"
    atomic_write_npz(
        predictions_path,
        validation_raw_index=holdout_index,
        anomaly=holdout_y,
        building_id=holdout_building,
        site_id=holdout_site,
        meter=holdout_meter,
        **scores,
    )
    metrics = {
        name: evaluation_summary(holdout_y, value) for name, value in scores.items()
    }
    result_path = args.out_root / "cell.json"
    write_json_with_provenance(
        result_path,
        {
            "schema_version": 1,
            "experiment": "m5_building_count_curve_tree_cell",
            **provenance,
            "score_names": [*MODEL_ORDER, "ensemble"],
            "available_buildings": len(cell["available_buildings"]),
            "tree_fit_buildings": len(cell["tree_fit_buildings"]),
            "tree_early_stop_buildings": len(cell["tree_early_stop_buildings"]),
            "metrics": metrics,
            "predictions": predictions_path.name,
            "fit": fit_record,
        },
        root=ROOT,
    )
    _write_heartbeat(
        heartbeat,
        status="completed",
        completed=len(spans),
        total=len(spans),
        current=None,
        started=started,
    )
    _atomic_json(args.out_root / "COMPLETE.json", {"cell": str(result_path)})
    print(f"Wrote {result_path} and {predictions_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
