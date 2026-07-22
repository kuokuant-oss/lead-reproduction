"""Run one resumable TabPFN context against the canonical full M3 test split.

The parent never imports torch or TabPFN. The worker fits once, atomically saves
the official TabPFN fitted state, and persists full-test predictions in
restartable checkpoint files. Query microbatches and disk checkpoints are
independent sizes.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from lead import BASELINE_FEATURE_COLS, PROC, RANDOM_STATE, ROOT, load_m3_frame
from lead.resource_guard import (
    LimitTracker,
    append_jsonl,
    atomic_write_json,
    query_gpu_memory,
    resolve_limits,
    sample_dict,
    sample_resources,
    terminate_process_tree,
)
from run_m5_tabpfn_single_context_scaling import (
    FakeTabPFNClassifier,
    Heartbeat,
    atomic_write_npz,
    batched_predict,
    evaluation_metrics,
    fixed_recall_threshold,
    fixed_score_indices,
    index_record,
    nested_balanced_indices,
    operating_metrics,
    serialize_failure,
    sha256_file,
    synthetic_frame,
    verify_fitted_context,
)


EXPERIMENT = "tabpfn_v3_canonical_m3_full_test"
DEFAULT_M3_PREDICTIONS = PROC / "m3_figure_predictions_50_50.npz"
DEFAULT_SITE_PREDICTIONS = (
    PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"
)
DEFAULT_BASELINE_PREDICTIONS = PROC / "m3_17_feature_ensemble_predictions.npz"


def default_model_path() -> Path:
    cache = os.environ.get("TABPFN_MODEL_CACHE_DIR")
    return (Path(cache) if cache else ROOT / ".tabpfn-cache") / (
        "tabpfn-v3-classifier-v3_default.ckpt"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-rows", type=int, required=True)
    parser.add_argument("--validation-rows", type=int, default=4_000)
    parser.add_argument("--query-microbatch-size", type=int, default=512)
    parser.add_argument("--min-query-microbatch-size", type=int, default=256)
    parser.add_argument("--checkpoint-rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument(
        "--canonical-m3-predictions", type=Path, default=DEFAULT_M3_PREDICTIONS
    )
    parser.add_argument(
        "--canonical-site-predictions", type=Path, default=DEFAULT_SITE_PREDICTIONS
    )
    parser.add_argument(
        "--canonical-baseline-predictions",
        type=Path,
        default=DEFAULT_BASELINE_PREDICTIONS,
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--state-out", type=Path)
    parser.add_argument("--events-out", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--gpu-soft-limit-fraction", type=float, default=0.86)
    parser.add_argument("--gpu-hard-limit-fraction", type=float, default=0.92)
    parser.add_argument("--ram-soft-limit-fraction", type=float, default=0.85)
    parser.add_argument("--ram-hard-limit-fraction", type=float, default=0.92)
    parser.add_argument("--gpu-soft-limit-mib", type=float)
    parser.add_argument("--gpu-hard-limit-mib", type=float)
    parser.add_argument("--ram-soft-limit-mib", type=float)
    parser.add_argument("--ram-hard-limit-mib", type=float)
    parser.add_argument("--soft-limit-consecutive-polls", type=int, default=4)
    parser.add_argument("--termination-grace-seconds", type=float, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.context_rows <= 0 or args.context_rows % 2:
        raise ValueError("context rows must be a positive even integer")
    if args.validation_rows <= 0:
        raise ValueError("validation rows must be positive")
    if not 0 < args.min_query_microbatch_size <= args.query_microbatch_size:
        raise ValueError("invalid query microbatch range")
    if args.checkpoint_rows < args.query_microbatch_size:
        raise ValueError("checkpoint rows must be at least one query microbatch")
    prefix = f"m5_tabpfn_canonical_full_test_context{args.context_rows}"
    args.out = args.out or PROC / f"{prefix}.json"
    args.state_out = args.state_out or PROC / f"{prefix}.state.json"
    args.events_out = args.events_out or PROC / f"{prefix}.events.jsonl"
    stem = args.out.stem
    args.work_dir = args.work_dir or args.out.with_name(f"{stem}.work")
    args.predictions_out = args.predictions_out or args.out.with_name(
        f"{stem}.predictions.npz"
    )
    return args


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def load_frame(smoke: bool) -> Any:
    return synthetic_frame() if smoke else load_m3_frame(verbose=True)


def canonical_contract(frame: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Freeze the M3 mod2 split and prove exact test row alignment."""
    if len(BASELINE_FEATURE_COLS) != 17:
        raise AssertionError("canonical run requires exactly 17 baseline features")
    train_mask = (frame["building_id"] % 2 == 0).to_numpy()
    test_mask = ~train_mask
    train_buildings = set(frame.loc[train_mask, "building_id"].unique())
    test_buildings = set(frame.loc[test_mask, "building_id"].unique())
    if train_buildings & test_buildings:
        raise AssertionError("canonical M3 train/test buildings overlap")

    train_index = frame.index[train_mask].to_numpy(dtype="int64")
    split_test_index = frame.index[test_mask].to_numpy(dtype="int64")

    if not args.smoke:
        with np.load(args.canonical_m3_predictions) as m3:
            required = {"anomaly", "m3_1_lightgbm", "lightgbm", "ensemble"}
            if missing := required - set(m3.files):
                raise ValueError(f"canonical M3 artifact missing {sorted(missing)}")
            canonical_y = np.asarray(m3["anomaly"], dtype="int8")
            canonical_ensemble = np.asarray(m3["ensemble"], dtype="float32")
            for score_name in ("m3_1_lightgbm", "lightgbm", "ensemble"):
                score = np.asarray(m3[score_name])
                if len(score) != len(canonical_y) or not np.isfinite(score).all():
                    raise AssertionError(f"invalid canonical M3 score: {score_name}")
        with np.load(args.canonical_site_predictions) as site:
            required = {
                "validation_raw_index",
                "anomaly",
                "site_id",
                "building_id",
                "ensemble",
            }
            if missing := required - set(site.files):
                raise ValueError(f"canonical site artifact missing {sorted(missing)}")
            test_index = np.asarray(site["validation_raw_index"], dtype="int64")
            test_y = np.asarray(site["anomaly"], dtype="int8")
            test_site = np.asarray(site["site_id"], dtype="int8")
            test_building = np.asarray(site["building_id"], dtype="int16")
            site_ensemble = np.asarray(site["ensemble"], dtype="float32")
        with np.load(args.canonical_baseline_predictions) as baseline:
            required = {"anomaly", "site_id", "ensemble"}
            if missing := required - set(baseline.files):
                raise ValueError(
                    f"canonical baseline artifact missing {sorted(missing)}"
                )
            baseline_y = np.asarray(baseline["anomaly"], dtype="int8")
            baseline_site = np.asarray(baseline["site_id"], dtype="int8")
        if len(np.unique(test_index)) != len(test_index):
            raise AssertionError("site artifact raw row IDs are not unique")
        if not np.array_equal(np.sort(test_index), split_test_index):
            raise AssertionError("site artifact row set differs from M3 test split")
        if not np.array_equal(test_y, canonical_y):
            raise AssertionError("M3 and site artifact labels are not aligned")
        if not np.array_equal(site_ensemble, canonical_ensemble):
            raise AssertionError("M3 and site artifact ensemble scores are not aligned")
        if not np.array_equal(test_y, baseline_y):
            raise AssertionError("17- and 137-feature labels are not aligned")
        if not np.array_equal(test_site, baseline_site):
            raise AssertionError("17- and 137-feature site IDs are not aligned")
        if not np.array_equal(
            frame.loc[test_index, "anomaly"].to_numpy(dtype="int8"), test_y
        ):
            raise AssertionError("site raw row IDs do not map to saved labels")
        if not np.array_equal(
            frame.loc[test_index, "site_id"].to_numpy(dtype="int8"), test_site
        ):
            raise AssertionError("site raw row IDs do not map to saved sites")
        if not np.array_equal(
            frame.loc[test_index, "building_id"].to_numpy(dtype="int16"),
            test_building,
        ):
            raise AssertionError("site raw row IDs do not map to saved buildings")
    else:
        test_index = split_test_index
        test_y = frame.loc[test_mask, "anomaly"].to_numpy(dtype="int8", copy=True)
        test_site = frame.loc[test_mask, "site_id"].to_numpy(dtype="int8", copy=True)
        test_building = frame.loc[test_mask, "building_id"].to_numpy(
            dtype="int16", copy=True
        )

    validation_index = fixed_score_indices(
        train_index, args.validation_rows, seed=args.seed + 20_000
    )
    candidate_mask = ~np.isin(train_index, validation_index)
    candidate_index = train_index[candidate_mask]
    candidate_y = frame.loc[candidate_index, "anomaly"].to_numpy(dtype="int8")
    context_index = nested_balanced_indices(
        candidate_index,
        candidate_y,
        [args.context_rows],
        seed=args.seed,
    )[args.context_rows]
    if np.intersect1d(context_index, validation_index).size:
        raise AssertionError("context and validation rows overlap")
    label_map = {
        int(row): int(label)
        for row, label in zip(candidate_index, candidate_y, strict=True)
    }
    return {
        "context_index": context_index,
        "validation_index": validation_index,
        "test_index": test_index,
        "test_y": test_y,
        "test_site_id": test_site,
        "test_building_id": test_building,
        "metadata": {
            "name": "m3_50_50_mod2_canonical",
            "train_rule": "building_id % 2 == 0",
            "test_rule": "building_id % 2 == 1",
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_buildings": len(train_buildings),
            "test_buildings": len(test_buildings),
            "building_overlap": 0,
            "context": index_record(context_index, label_map),
            "validation_row_sha256": array_sha256(validation_index.astype("<i8")),
            "test_row_sha256": array_sha256(test_index.astype("<i8")),
            "test_label_sha256": array_sha256(test_y),
            "test_site_sha256": array_sha256(test_site),
            "canonical_m3_scores_aligned_to_a0_order": True,
            "canonical_site_ids_aligned": True,
            "canonical_target_order_source": "m6_a0_validation_raw_index",
            "m3_validation_raw_index_used": False,
            "canonical_buildings_aligned": True,
        },
    }


def checkpoint_path(directory: Path, index: int) -> Path:
    return directory / f"chunk_{index:06d}.npz"


def save_checkpoint(
    path: Path,
    *,
    raw_index: np.ndarray,
    y: np.ndarray,
    score: np.ndarray,
    site_id: np.ndarray,
    building_id: np.ndarray,
) -> None:
    atomic_write_npz(
        path,
        raw_index=raw_index.astype("int64", copy=False),
        y=y.astype("int8", copy=False),
        score=score.astype("float32", copy=False),
        site_id=site_id.astype("int8", copy=False),
        building_id=building_id.astype("int16", copy=False),
    )


def load_checkpoint(
    path: Path, expected_index: np.ndarray
) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path) as payload:
        result = {name: np.asarray(payload[name]) for name in payload.files}
    if not np.array_equal(result.get("raw_index"), expected_index):
        raise AssertionError(f"checkpoint row identity drifted: {path}")
    if len(result.get("score", ())) != len(expected_index):
        raise AssertionError(f"checkpoint score length drifted: {path}")
    if not np.isfinite(result["score"]).all():
        raise AssertionError(f"checkpoint contains non-finite scores: {path}")
    return result


def atomic_joblib_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_save_fitted_model(model: Any, path: Path) -> None:
    from tabpfn.model_loading import save_fitted_tabpfn_model

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.tabpfn_fit")
    save_fitted_tabpfn_model(model, temporary)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def create_real_model(model_path: Path, seed: int) -> Any:
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        n_estimators=1,
        auto_scale_n_estimators=False,
        model_path=str(model_path.resolve()),
        device="cuda",
        ignore_pretraining_limits=True,
        fit_mode="low_memory",
        memory_saving_mode=True,
        keep_cache_on_device=False,
        random_state=seed,
        n_preprocessing_jobs=1,
        inference_config={"SUBSAMPLE_SAMPLES": None},
        show_progress_bar=False,
    )


def fit_or_load(
    frame: Any,
    contract: dict[str, Any],
    args: argparse.Namespace,
    heartbeat: Heartbeat,
) -> tuple[Any, StandardScaler, str, dict[str, Any]]:
    fit_state = args.work_dir / "model.tabpfn_fit"
    fake_state = args.work_dir / "fake_model.joblib"
    scaler_path = args.work_dir / "scaler.joblib"
    manifest_path = args.work_dir / "fit_manifest.json"
    expected_manifest = {
        "experiment": EXPERIMENT,
        "context_rows": args.context_rows,
        "context_sha256": contract["metadata"]["context"]["sha256"],
        "feature_names": list(BASELINE_FEATURE_COLS),
        "model_path": str(args.model_path.resolve()),
        "model_sha256": sha256_file(args.model_path) if not args.smoke else None,
        "seed": args.seed,
    }
    if manifest_path.is_file():
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed != expected_manifest:
            raise AssertionError("saved fit manifest differs from requested context")
    else:
        atomic_write_json(manifest_path, expected_manifest)

    saved_model = fake_state if args.smoke else fit_state
    if saved_model.is_file() and scaler_path.is_file():
        heartbeat.update("load_fitted_state")
        if args.smoke:
            model = joblib.load(saved_model)
        else:
            from tabpfn.model_loading import load_fitted_tabpfn_model

            model = load_fitted_tabpfn_model(saved_model, device="cuda")
        scaler = joblib.load(scaler_path)
        context = verify_fitted_context(model, args.context_rows)
        if context["status"] != "verified":
            raise RuntimeError("loaded fitted state failed context verification")
        return model, scaler, "loaded", context

    heartbeat.update("fit")
    context_index = contract["context_index"]
    x_context = frame.loc[context_index, BASELINE_FEATURE_COLS].to_numpy(
        dtype="float32", copy=True
    )
    y_context = frame.loc[context_index, "anomaly"].to_numpy(dtype="int64", copy=True)
    scaler = StandardScaler(copy=False)
    x_context = scaler.fit_transform(x_context).astype("float32", copy=False)
    model = (
        FakeTabPFNClassifier()
        if args.smoke
        else create_real_model(args.model_path, args.seed)
    )
    model.fit(x_context, y_context)
    context = verify_fitted_context(model, args.context_rows)
    if context["status"] != "verified":
        raise RuntimeError("new fitted state failed context verification")
    atomic_joblib_dump(scaler, scaler_path)
    if args.smoke:
        atomic_joblib_dump(model, fake_state)
    else:
        atomic_save_fitted_model(model, fit_state)
    del x_context, y_context
    gc.collect()
    return model, scaler, "fitted", context


def worker(args: argparse.Namespace) -> int:
    os.environ["TABPFN_NO_BROWSER"] = "1"
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.work_dir / "chunks"
    progress_path = args.work_dir / "progress.json"
    result_path = args.work_dir / "worker_result.json"
    heartbeat_path = args.work_dir / "heartbeat.json"
    stop_path = args.work_dir / "stop.json"
    stage = "data_loading"
    torch_module = None
    model = frame = contract = scaler = None
    started = time.perf_counter()
    try:
        if not args.smoke:
            import torch

            torch_module = torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
            torch.cuda.reset_peak_memory_stats()
        frame = load_frame(args.smoke)
        contract = canonical_contract(frame, args)
        progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.is_file()
            else {
                "status": "pending",
                "completed_checkpoints": [],
                "checkpoint_timings": {},
            }
        )
        with Heartbeat(heartbeat_path, args.context_rows, torch_module) as heartbeat:
            model, scaler, fit_action, context = fit_or_load(
                frame, contract, args, heartbeat
            )
            progress.update(
                {
                    "status": "fitted",
                    "fit_action": fit_action,
                    "context_contract": context,
                    "split": contract["metadata"],
                }
            )
            atomic_write_json(progress_path, progress)

            stage = "validation_predict"
            heartbeat.update(stage)
            validation_index = contract["validation_index"]
            validation_path = args.work_dir / "validation.npz"
            validation_saved = load_checkpoint(validation_path, validation_index)
            if validation_saved is not None:
                y_validation = validation_saved["y"].astype("int64", copy=False)
                validation_score = validation_saved["score"]
                validation_seconds = 0.0
                validation_action = "loaded"
            else:
                x_validation = frame.loc[
                    validation_index, BASELINE_FEATURE_COLS
                ].to_numpy(dtype="float32", copy=True)
                x_validation = scaler.transform(x_validation).astype(
                    "float32", copy=False
                )
                y_validation = frame.loc[validation_index, "anomaly"].to_numpy(
                    dtype="int64", copy=True
                )
                validation_started = time.perf_counter()
                validation_score, _ = batched_predict(
                    model,
                    x_validation,
                    initial_batch_size=args.query_microbatch_size,
                    minimum_batch_size=args.min_query_microbatch_size,
                    stop_requested=stop_path.exists,
                    heartbeat=heartbeat,
                    stage=stage,
                )
                validation_seconds = time.perf_counter() - validation_started
                save_checkpoint(
                    validation_path,
                    raw_index=validation_index,
                    y=y_validation,
                    score=validation_score,
                    site_id=frame.loc[validation_index, "site_id"].to_numpy(
                        dtype="int8", copy=True
                    ),
                    building_id=frame.loc[validation_index, "building_id"].to_numpy(
                        dtype="int16", copy=True
                    ),
                )
                del x_validation
                validation_action = "predicted"
            expected_validation_y = frame.loc[validation_index, "anomaly"].to_numpy(
                dtype="int64", copy=True
            )
            if not np.array_equal(y_validation, expected_validation_y):
                raise AssertionError("saved validation labels drifted")
            threshold = fixed_recall_threshold(y_validation, validation_score)

            stage = "full_test_predict"
            test_index = contract["test_index"]
            checkpoint_count = int(np.ceil(len(test_index) / args.checkpoint_rows))
            test_started = time.perf_counter()
            effective_batches: list[int] = []
            for checkpoint_index in range(checkpoint_count):
                start = checkpoint_index * args.checkpoint_rows
                end = min(len(test_index), start + args.checkpoint_rows)
                expected_index = test_index[start:end]
                path = checkpoint_path(chunks_dir, checkpoint_index)
                saved = load_checkpoint(path, expected_index)
                if saved is not None:
                    if checkpoint_index not in progress["completed_checkpoints"]:
                        progress["completed_checkpoints"].append(checkpoint_index)
                    heartbeat.update(stage, end)
                    continue
                if stop_path.exists():
                    raise InterruptedError(
                        "watchdog requested stop before next checkpoint"
                    )
                x_query = frame.loc[expected_index, BASELINE_FEATURE_COLS].to_numpy(
                    dtype="float32", copy=True
                )
                x_query = scaler.transform(x_query).astype("float32", copy=False)
                y_query = frame.loc[expected_index, "anomaly"].to_numpy(
                    dtype="int8", copy=True
                )
                site_query = frame.loc[expected_index, "site_id"].to_numpy(
                    dtype="int8", copy=True
                )
                building_query = frame.loc[expected_index, "building_id"].to_numpy(
                    dtype="int16", copy=True
                )
                checkpoint_started = time.perf_counter()
                score, effective_batch = batched_predict(
                    model,
                    x_query,
                    initial_batch_size=args.query_microbatch_size,
                    minimum_batch_size=args.min_query_microbatch_size,
                    stop_requested=stop_path.exists,
                    heartbeat=heartbeat,
                    stage=stage,
                    position_offset=start,
                )
                checkpoint_seconds = time.perf_counter() - checkpoint_started
                save_checkpoint(
                    path,
                    raw_index=expected_index,
                    y=y_query,
                    score=score,
                    site_id=site_query,
                    building_id=building_query,
                )
                effective_batches.append(effective_batch)
                progress["completed_checkpoints"] = sorted(
                    {*progress["completed_checkpoints"], checkpoint_index}
                )
                progress["checkpoint_timings"][str(checkpoint_index)] = {
                    "rows": int(end - start),
                    "seconds": float(checkpoint_seconds),
                    "effective_microbatch_size": int(effective_batch),
                }
                progress["status"] = "predicting"
                progress["rows_completed"] = int(end)
                atomic_write_json(progress_path, progress)
                heartbeat.update(stage, end)
                del x_query, y_query, site_query, building_query, score
                gc.collect()
            test_seconds = time.perf_counter() - test_started

            stage = "aggregate_predictions"
            heartbeat.update(stage, len(test_index))
            del frame
            gc.collect()
            chunk_payloads = [
                load_checkpoint(
                    checkpoint_path(chunks_dir, index),
                    test_index[
                        index * args.checkpoint_rows : min(
                            len(test_index), (index + 1) * args.checkpoint_rows
                        )
                    ],
                )
                for index in range(checkpoint_count)
            ]
            if any(payload is None for payload in chunk_payloads):
                raise AssertionError("full-test checkpoint reconstruction incomplete")
            payloads = [payload for payload in chunk_payloads if payload is not None]
            raw_index = np.concatenate([payload["raw_index"] for payload in payloads])
            y_test = np.concatenate([payload["y"] for payload in payloads])
            test_score = np.concatenate([payload["score"] for payload in payloads])
            site_id = np.concatenate([payload["site_id"] for payload in payloads])
            building_id = np.concatenate(
                [payload["building_id"] for payload in payloads]
            )
            if not np.array_equal(raw_index, test_index):
                raise AssertionError("aggregated full-test row order drifted")
            if not np.array_equal(y_test, contract["test_y"]):
                raise AssertionError("aggregated full-test labels drifted")
            if not np.array_equal(site_id, contract["test_site_id"]):
                raise AssertionError("aggregated full-test site order drifted")
            if not np.array_equal(building_id, contract["test_building_id"]):
                raise AssertionError("aggregated full-test buildings drifted")
            del chunk_payloads, payloads
            gc.collect()

            atomic_write_npz(
                args.predictions_out,
                raw_index=raw_index,
                anomaly=y_test,
                tabpfn=test_score.astype("float32", copy=False),
                site_id=site_id,
                building_id=building_id,
            )
            result = {
                "status": "completed",
                "experiment": EXPERIMENT,
                "context_rows": args.context_rows,
                "fit_action": fit_action,
                "context_contract": context,
                "split": contract["metadata"],
                "validation": evaluation_metrics(y_validation, validation_score),
                "test": {
                    **evaluation_metrics(y_test, test_score),
                    **operating_metrics(y_test, test_score, threshold),
                },
                "validation_prediction_seconds": float(validation_seconds),
                "validation_action": validation_action,
                "full_test_prediction_seconds_this_session": float(test_seconds),
                "checkpoint_rows": args.checkpoint_rows,
                "checkpoint_count": checkpoint_count,
                "completed_checkpoints": len(progress["completed_checkpoints"]),
                "query_microbatch_size": args.query_microbatch_size,
                "effective_query_microbatch_size": min(effective_batches)
                if effective_batches
                else args.query_microbatch_size,
                "fit_state": {
                    "path": str((args.work_dir / "model.tabpfn_fit").resolve()),
                    "sha256": sha256_file(args.work_dir / "model.tabpfn_fit")
                    if not args.smoke
                    else None,
                },
                "prediction_artifact": {
                    "path": str(args.predictions_out.resolve()),
                    "sha256": sha256_file(args.predictions_out),
                    "size_bytes": args.predictions_out.stat().st_size,
                    "rows": int(len(test_score)),
                },
                "torch_peak_allocated_bytes": int(
                    torch_module.cuda.max_memory_allocated()
                )
                if torch_module is not None
                else None,
                "torch_peak_reserved_bytes": int(
                    torch_module.cuda.max_memory_reserved()
                )
                if torch_module is not None
                else None,
                "total_seconds_this_session": float(time.perf_counter() - started),
            }
            progress["status"] = "completed"
            progress["rows_completed"] = int(len(test_index))
            atomic_write_json(progress_path, progress)
            heartbeat.update("completed", len(test_index))
        atomic_write_json(result_path, result)
        return 0
    except BaseException as error:
        failure = serialize_failure(
            error,
            budget=args.context_rows,
            stage=stage,
            predict_batch_size=args.query_microbatch_size,
            torch_module=torch_module,
        )
        atomic_write_json(result_path, failure)
        return 1
    finally:
        model = contract = scaler = None
        gc.collect()
        if torch_module is not None:
            try:
                torch_module.cuda.empty_cache()
                torch_module.cuda.ipc_collect()
            except RuntimeError:
                pass


def prepare_worker_invocation(work_dir: Path) -> None:
    """Clear prior process signals while preserving every durable checkpoint."""
    for name in ("worker_result.json", "heartbeat.json", "stop.json"):
        (work_dir / name).unlink(missing_ok=True)


def controller(args: argparse.Namespace) -> int:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    worker_result = args.work_dir / "worker_result.json"
    heartbeat = args.work_dir / "heartbeat.json"
    stop_request = args.work_dir / "stop.json"
    if args.force:
        for path in (
            args.out,
            args.state_out,
            args.events_out,
            worker_result,
            heartbeat,
            stop_request,
        ):
            path.unlink(missing_ok=True)
    elif args.out.is_file():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            print(f"Already completed: {args.out}")
            return 0
    elif args.state_out.exists() and not args.resume:
        raise FileExistsError("existing state; pass --resume")
    prepare_worker_invocation(args.work_dir)
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    )
    state = {
        "status": "running",
        "context_rows": args.context_rows,
        "worker_pid": process.pid,
        "started_at": time.time(),
        "updated_at": time.time(),
        "stop_reason": None,
    }
    atomic_write_json(args.state_out, state)
    append_jsonl(
        args.events_out,
        {
            "event": "worker_started",
            "timestamp": time.time(),
            "context_rows": args.context_rows,
            "pid": process.pid,
        },
    )
    import psutil

    initial_gpu = query_gpu_memory(process.pid)
    memory = psutil.virtual_memory()
    limits = resolve_limits(
        gpu_total_mib=initial_gpu.get("total_mib"),
        ram_total_mib=float(memory.total / 1024**2),
        gpu_soft_fraction=args.gpu_soft_limit_fraction,
        gpu_hard_fraction=args.gpu_hard_limit_fraction,
        ram_soft_fraction=args.ram_soft_limit_fraction,
        ram_hard_fraction=args.ram_hard_limit_fraction,
        gpu_soft_mib=args.gpu_soft_limit_mib,
        gpu_hard_mib=args.gpu_hard_limit_mib,
        ram_soft_mib=args.ram_soft_limit_mib,
        ram_hard_mib=args.ram_hard_limit_mib,
        soft_limit_consecutive_polls=args.soft_limit_consecutive_polls,
        timeout_seconds=None,
    )
    tracker = LimitTracker(limits)
    started = time.monotonic()
    soft_requested_at = None
    last_stage = None
    peak_gpu = peak_rss = peak_system = 0.0
    last_sample = None
    termination_reason = None
    while process.poll() is None:
        try:
            sample = sample_resources(process.pid)
            last_sample = sample_dict(sample)
            peak_gpu = max(peak_gpu, sample.gpu_used_mib or 0)
            peak_rss = max(peak_rss, sample.worker_rss_mib)
            peak_system = max(peak_system, sample.system_used_mib)
            decision = tracker.observe(
                sample, elapsed_seconds=time.monotonic() - started
            )
        except Exception:
            decision = type("Decision", (), {"action": "continue", "reason": None})()
        if heartbeat.is_file():
            try:
                current = json.loads(heartbeat.read_text(encoding="utf-8"))
                if current.get("stage") != last_stage:
                    last_stage = current.get("stage")
                    append_jsonl(
                        args.events_out,
                        {
                            "event": "stage_changed",
                            "timestamp": time.time(),
                            "context_rows": args.context_rows,
                            "stage": last_stage,
                            "position": current.get("prediction_batch_position"),
                        },
                    )
                if time.time() - heartbeat.stat().st_mtime >= max(
                    5.0, args.termination_grace_seconds
                ):
                    termination_reason = "worker no longer responsive"
            except (OSError, json.JSONDecodeError):
                pass
        if decision.action == "request_stop" and soft_requested_at is None:
            soft_requested_at = time.monotonic()
            atomic_write_json(
                stop_request, {"reason": decision.reason, "timestamp": time.time()}
            )
            append_jsonl(
                args.events_out,
                {
                    "event": "stop_requested",
                    "timestamp": time.time(),
                    "reason": decision.reason,
                },
            )
        if decision.action == "terminate":
            termination_reason = decision.reason
        elif (
            soft_requested_at is not None
            and time.monotonic() - soft_requested_at >= args.termination_grace_seconds
        ):
            termination_reason = "soft-limit grace period expired"
        if termination_reason:
            details = terminate_process_tree(
                process.pid, grace_seconds=args.termination_grace_seconds
            )
            append_jsonl(
                args.events_out,
                {
                    "event": "worker_terminated",
                    "timestamp": time.time(),
                    "reason": termination_reason,
                    "details": details,
                },
            )
            break
        time.sleep(args.poll_seconds)
    process.wait()
    result = (
        json.loads(worker_result.read_text(encoding="utf-8"))
        if worker_result.is_file()
        else {
            "status": "terminated" if termination_reason else "crashed",
            "stage": last_stage,
            "stop_reason": termination_reason,
        }
    )
    result.update(
        {
            "worker_exit_code": process.returncode,
            "watchdog_peak_gpu_mib": peak_gpu or None,
            "watchdog_peak_worker_rss_mib": peak_rss or None,
            "watchdog_peak_system_ram_mib": peak_system or None,
            "last_watchdog_sample": last_sample,
            "resource_limits": {
                "gpu_soft_mib": limits.gpu_soft_mib,
                "gpu_hard_mib": limits.gpu_hard_mib,
                "ram_soft_mib": limits.ram_soft_mib,
                "ram_hard_mib": limits.ram_hard_mib,
                "timeout_seconds": None,
            },
        }
    )
    atomic_write_json(args.out, result)
    state.update(
        {
            "status": result.get("status"),
            "worker_pid": None,
            "updated_at": time.time(),
            "stop_reason": result.get("stop_reason") or result.get("exception_message"),
        }
    )
    atomic_write_json(args.state_out, state)
    append_jsonl(
        args.events_out,
        {
            "event": "run_completed",
            "timestamp": time.time(),
            "status": state["status"],
            "context_rows": args.context_rows,
        },
    )
    print(f"Saved {args.out}; status={state['status']}")
    return 0


def parent_has_forbidden_imports() -> bool:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import) and any(
            alias.name.split(".")[0] in {"torch", "tabpfn"} for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in {
            "torch",
            "tabpfn",
        }:
            return True
    return False


def main() -> int:
    args = parse_args()
    return worker(args) if args.worker else controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
