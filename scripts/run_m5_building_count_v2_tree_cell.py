"""Run one M5 building-count V2 frozen-tree cell without early stopping.

V2 uses every manifest-allocated available row for both trees and TabPFN.
The tree ensemble follows the frozen matched-context contract: fixed iteration
counts, no validation split, no M3 class resampling, and no early stopping.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

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
from run_m3_figure_observations import (
    MODEL_ORDER,
    evaluation_summary,
    fit_frozen_models,
    frozen_model_contract,
    predict_probability,
)
from run_m5_building_curve_tree_cell import (
    _atomic_joblib,
    _atomic_json,
    _bounded_rows,
    _collect_and_trim,
    _formal_gate,
    _matrix_columns,
    _scale_matrix,
    _sha256_file,
    _write_heartbeat,
)
from run_m5_tree_ensemble_matched_context import (
    CANONICAL_ORDER,
    atomic_write_npz,
    build_features_keeping_index,
)

EXPERIMENT_VERSION = "m5_building_count_v2"
MATRIX_DTYPE = np.dtype("float32")
PREDICTION_DTYPE = np.dtype("float32")


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
    parser.add_argument("--features", type=int, choices=(137,), default=137)
    parser.add_argument(
        "--model-seed", "--seed", dest="model_seed", type=int, default=42
    )
    parser.add_argument("--predict-batch-rows", type=int, default=200_000)
    parser.add_argument(
        "--mode", choices=("plan", "validation", "formal"), default="plan"
    )
    parser.add_argument("--max-context-rows", type=int)
    parser.add_argument("--max-holdout-rows", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-root", type=Path)
    args = parser.parse_args(argv)
    if args.building_budget <= 0:
        raise ValueError("building budget must be positive")
    if args.predict_batch_rows <= 0:
        raise ValueError("predict batch rows must be positive")
    caps = (args.max_context_rows, args.max_holdout_rows)
    if args.mode == "validation":
        if any(value is None or value <= 0 for value in caps):
            raise ValueError("validation mode requires positive context/holdout caps")
    elif any(value is not None for value in caps):
        raise ValueError("row caps are only allowed in validation mode")
    tag = (
        f"{_building_seed_tag(args.building_manifest)}_"
        f"k{args.building_budget}_f{args.features}"
    )
    if args.out_root is None:
        base = PROC / "m5_building_curve" / "v2"
        args.out_root = (
            base / "NON_SCIENTIFIC_VALIDATION" / tag
            if args.mode == "validation"
            else base / "formal" / tag
        )
    return args


def _canonical_holdout(
    frame: object, cap: int | None, seed: int
) -> dict[str, np.ndarray]:
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
        output = {
            "validation_raw_index": np.asarray(
                canonical["validation_raw_index"], dtype="int64"
            ),
            "anomaly": np.asarray(canonical["anomaly"], dtype="int8"),
            "site_id": np.asarray(canonical["site_id"], dtype="int8"),
            "building_id": np.asarray(canonical["building_id"], dtype="int16"),
        }
    if cap is not None:
        output["validation_raw_index"] = _bounded_rows(
            frame,
            output["validation_raw_index"],
            cap,
            seed=seed,
        )
        index = output["validation_raw_index"]
        output["anomaly"] = frame.loc[index, "anomaly"].to_numpy(dtype="int8")
        output["site_id"] = frame.loc[index, "site_id"].to_numpy(dtype="int8")
        output["building_id"] = frame.loc[index, "building_id"].to_numpy(dtype="int16")
    index = output["validation_raw_index"]
    if frame.loc[index, "building_id"].mod(2).eq(0).any():
        raise AssertionError("canonical holdout contains an even training building")
    output["meter"] = frame.loc[index, "meter"].to_numpy(dtype="int8")
    return output


def _ensemble_probabilities(
    models: dict[str, object], matrix: np.ndarray
) -> dict[str, np.ndarray]:
    scores = {
        name: np.asarray(
            predict_probability(name, models[name], matrix),
            dtype=PREDICTION_DTYPE,
        )
        for name in MODEL_ORDER
    }
    scores["ensemble"] = np.mean(
        [scores[name] for name in MODEL_ORDER], axis=0, dtype="float64"
    ).astype(PREDICTION_DTYPE)
    return scores


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(args.building_manifest.read_text(encoding="utf-8"))
    building_seed = manifest_building_seed(manifest)
    if manifest.get("sampling_profile") != SAMPLING_PROFILE:
        raise SystemExit(f"V2 requires sampling_profile={SAMPLING_PROFILE!r}")
    if building_seed is None:
        raise SystemExit("V2 manifest lacks building_seed identity")
    if f"building_seed{building_seed}" not in str(args.out_root):
        raise SystemExit("V2 out-root must contain its building_seed identity")
    cell = manifest.get("cells", {}).get(str(args.building_budget))
    if cell is None:
        raise SystemExit(f"manifest has no K={args.building_budget} cell")
    if not cell.get("constraint_pass"):
        raise SystemExit(
            f"manifest K={args.building_budget} failed sampling constraints"
        )

    print(
        f"V2 K={args.building_budget}: available buildings="
        f"{len(cell['available_buildings'])}; tree early stopping=disabled; "
        "all available rows are fit rows",
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
    resolved = resolve_cell_indices(
        frame.loc[train_mask],
        manifest,
        args.building_budget,
        require_role_class_coverage=False,
    )
    context_index = _bounded_rows(
        frame,
        resolved["available_rows"],
        args.max_context_rows,
        seed=args.model_seed + 1,
    )
    if frame.loc[context_index, "building_id"].mod(2).any():
        raise AssertionError("V2 tree context contains an odd holdout building")
    holdout = _canonical_holdout(
        frame,
        args.max_holdout_rows,
        args.model_seed + 2,
    )
    holdout_index = holdout["validation_raw_index"]

    provenance = {
        "mode": "FORMAL" if args.mode == "formal" else "NON_SCIENTIFIC_VALIDATION",
        "experiment_version": EXPERIMENT_VERSION,
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
        "training_sampling": "exact_manifest_available_rows_no_resampling",
        "class_ratio_policy": "natural_prevalence_of_manifest_available_rows",
        "tree_role_policy": "ignored_for_fit_all_K_buildings_used",
        "early_stopping": False,
        "early_stopping_policy": "disabled_frozen_iteration_contract",
        "context_row_sha256": int_array_sha256(context_index),
        "fit_row_sha256": int_array_sha256(context_index),
        "holdout_row_sha256": int_array_sha256(holdout_index),
        "context_rows": int(len(context_index)),
        "holdout_rows": int(len(holdout_index)),
        "matrix_dtype": MATRIX_DTYPE.name,
        "prediction_dtype": PREDICTION_DTYPE.name,
        "caps": {
            "context": args.max_context_rows,
            "holdout": args.max_holdout_rows,
        },
    }
    provenance_path = args.out_root / "provenance.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text(encoding="utf-8")) != provenance:
            raise AssertionError(
                "result-affecting V2 provenance differs from saved run"
            )
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
            raise AssertionError("cached V2 model fit-row identity drifted")
        print(
            "Reused frozen V2 models without rebuilding training features", flush=True
        )
    else:
        print(
            "Building timestamp-merge features for selected even buildings",
            flush=True,
        )
        selected_mask = frame["building_id"].isin(resolved["available_buildings"])
        train_features = build_features_keeping_index(frame.loc[selected_mask].copy())
        selected = train_features.loc[context_index]
        if not np.array_equal(selected.index.to_numpy(dtype="int64"), context_index):
            raise AssertionError(
                "V2 tree training row order differs from manifest rows"
            )
        x_fit = selected[columns].to_numpy(dtype=MATRIX_DTYPE)
        y_fit = frame.loc[context_index, "anomaly"].to_numpy(dtype="int8")
        del selected, train_features, frame
        _collect_and_trim()
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_fit).astype(MATRIX_DTYPE, copy=False)
        models, fit_seconds = fit_frozen_models(
            x_fit,
            y_fit,
            seed=args.model_seed,
        )
        fit_record = {
            "experiment_version": EXPERIMENT_VERSION,
            "training_sampling": provenance["training_sampling"],
            "class_ratio_policy": provenance["class_ratio_policy"],
            "fit_rows": int(len(context_index)),
            "fit_row_sha256": provenance["fit_row_sha256"],
            "early_stopping": False,
            "model_contract": frozen_model_contract(args.model_seed),
            "fit_seconds": {name: float(fit_seconds[name]) for name in MODEL_ORDER},
        }
        _atomic_joblib(model_path, {"scaler": scaler, "models": models})
        _atomic_json(fit_path, fit_record)
        del x_fit, y_fit
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
        print("Building timestamp-merge features for odd canonical holdout", flush=True)
        holdout_features = build_features_keeping_index(frame.loc[~train_mask].copy())
    else:
        print("Reused all V2 prediction chunks", flush=True)
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
                        f"V2 prediction checkpoint identity drifted: {path}"
                    )
                for name in (*MODEL_ORDER, "ensemble"):
                    if stored[name].dtype != PREDICTION_DTYPE:
                        raise AssertionError(
                            f"V2 prediction dtype drifted for {name}: {path}"
                        )
        else:
            if holdout_features is None:
                raise AssertionError(
                    "missing V2 prediction chunk without holdout features"
                )
            block = holdout_features.loc[holdout_index[start:end], columns].to_numpy(
                dtype=MATRIX_DTYPE
            )
            block = _scale_matrix(scaler, block)
            score = _ensemble_probabilities(models, block)
            atomic_write_npz(
                path,
                validation_raw_index=holdout_index[start:end],
                **score,
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
        name: np.empty(len(holdout_index), dtype=PREDICTION_DTYPE)
        for name in (*MODEL_ORDER, "ensemble")
    }
    for start, end in spans:
        with np.load(chunks / f"rows_{start:09d}_{end:09d}.npz") as stored:
            for name in scores:
                scores[name][start:end] = stored[name]
    predictions_path = args.out_root / "predictions.npz"
    atomic_write_npz(
        predictions_path,
        validation_raw_index=holdout_index,
        anomaly=holdout["anomaly"],
        building_id=holdout["building_id"],
        site_id=holdout["site_id"],
        meter=holdout["meter"],
        **scores,
    )
    metrics = {
        name: evaluation_summary(holdout["anomaly"], values)
        for name, values in scores.items()
    }
    result_path = args.out_root / "cell.json"
    write_json_with_provenance(
        result_path,
        {
            "schema_version": 2,
            "experiment": "m5_building_count_v2_tree_cell",
            **provenance,
            "score_names": [*MODEL_ORDER, "ensemble"],
            "available_buildings": len(cell["available_buildings"]),
            "fit_buildings": len(cell["available_buildings"]),
            "early_stop_buildings": 0,
            "manifest_roles_ignored_for_training": True,
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
