"""Run one Site 1/Site 8 M6 B1 TabPFN context-size cell.

This additive runner reuses a prepared B1 meter manifest, keeps the target site
fully unseen, fits TabPFN once, and scores the complete target site in
resumable query chunks. It never modifies the frozen M3/M6 tree runners.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from experiment_observability import host_environment
from lead import (
    BASELINE_FEATURE_COLS,
    PROC,
    RANDOM_STATE,
    ROOT,
    SHIFTS,
    add_value_change_features,
    load_m3_frame,
    write_json_with_provenance,
)
from m6_site_transfer_protocol import (
    array_fingerprint,
    location_split_masks,
    meter_manifest_mask,
)
from m6_tabpfn_context_protocol import (
    RegionTimer,
    ResourceMonitor,
    nested_stratified_context_positions,
    timing_dict,
)
from run_m3_figure_observations import curve_summary, evaluation_summary
from run_m6_site_transfer import score_histograms


VALUE_CHANGE_REGIME = "timestamp_merge"
DIRECTION_CONTRACT = {
    "a1": {"split": "a1_even_to_odd", "target_site": 1},
    "a2": {"split": "a2_odd_to_even", "target_site": 8},
}
EXPECTED_TARGET_SUPPORT = {
    1: {"rows": 553_357, "anomalies": 77_779},
    8: {"rows": 567_915, "anomalies": 43_504},
}


def log(message: str) -> None:
    print(message, flush=True)


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def artifact_path(path: Path) -> str:
    try:
        return resolve(path).relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolve(path))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=tuple(DIRECTION_CONTRACT), required=True)
    parser.add_argument("--meter-budget", required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument(
        "--context-rows", type=int, choices=(10_000, 100_000), required=True
    )
    parser.add_argument("--manifest-in", type=Path, required=True)
    parser.add_argument("--query-chunk-size", type=int, default=4_000)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / ".tabpfn-cache" / "tabpfn-v3-classifier-v3_default.ckpt",
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    parser.add_argument(
        "--max-query-chunks",
        type=int,
        help="Smoke-only limit. Produces partial status and never whole-site metrics.",
    )
    return parser.parse_args()


def cell_stem(args: argparse.Namespace) -> str:
    return (
        f"m6_tabpfn_b1_{args.direction}_meters{str(args.meter_budget).lower()}"
        f"_seed{args.selection_seed}_context{args.context_rows}"
    )


def load_prepared_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("status") != "manifest_prepared":
        raise ValueError(f"{path} is not a prepared M6 manifest")
    if payload.get("cell") != "b1" or not payload.get("selection"):
        raise ValueError(f"{path} is not a prepared M6 B1 manifest")
    return payload, hashlib.sha256(raw).hexdigest()


def stable_context_seed(selection_seed: int) -> int:
    """Keep context ranking frozen across 10k/100k for one B1 selection seed."""
    return int(selection_seed) ^ 0x4D365450


def feature_columns(frame: pd.DataFrame) -> list[str]:
    value_cols = [column for column in frame if column.startswith("lag_value_")]
    columns = [*BASELINE_FEATURE_COLS, *value_cols]
    if len(value_cols) != 120 or len(columns) != 137:
        raise RuntimeError(
            f"Frozen feature contract mismatch: {len(value_cols)=}, {len(columns)=}"
        )
    return columns


def extract_context_features(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    context_raw_index: np.ndarray,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Compute complete-series features, then retain exact context rows."""
    requested = frame.loc[
        frame["_raw_index"].isin(context_raw_index),
        ["building_id", "meter"],
    ].drop_duplicates()
    selected_keys = pd.MultiIndex.from_frame(requested)
    all_keys = pd.MultiIndex.from_frame(frame[["building_id", "meter"]])
    complete_series_mask = train_mask & np.asarray(all_keys.isin(selected_keys))
    complete_rows = int(complete_series_mask.sum())
    started = time.perf_counter()
    featured = add_value_change_features(
        frame.loc[complete_series_mask],
        list(SHIFTS),
        value_change_regime=VALUE_CHANGE_REGIME,
    )
    columns = feature_columns(featured)
    indexed = featured.set_index("_raw_index", drop=False)
    missing = np.setdiff1d(context_raw_index, indexed.index.to_numpy())
    if len(missing):
        raise AssertionError(f"context feature extraction lost {len(missing)} rows")
    context = indexed.loc[context_raw_index].reset_index(drop=True)
    return (
        context,
        columns,
        {
            "complete_meter_series_rows": complete_rows,
            "complete_meter_series": int(len(requested)),
            "feature_seconds": float(time.perf_counter() - started),
        },
    )


def target_cache_path(target_site: int) -> Path:
    return (
        ROOT
        / ".cache"
        / "m6_tabpfn_features"
        / f"site_{target_site}_timestamp_merge.parquet"
    )


def load_or_build_target_features(
    frame: pd.DataFrame,
    target_mask: np.ndarray,
    *,
    target_site: int,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    cache = target_cache_path(target_site)
    if cache.exists():
        started = time.perf_counter()
        featured = pd.read_parquet(cache)
        columns = feature_columns(featured)
        source = "cache"
    else:
        started = time.perf_counter()
        featured = add_value_change_features(
            frame.loc[target_mask],
            list(SHIFTS),
            value_change_regime=VALUE_CHANGE_REGIME,
        )
        columns = feature_columns(featured)
        cache.parent.mkdir(parents=True, exist_ok=True)
        featured.to_parquet(cache, compression="zstd", index=False)
        source = "built"
    expected = EXPECTED_TARGET_SUPPORT[target_site]
    if len(featured) != expected["rows"]:
        raise AssertionError(
            f"Site {target_site} row drift: {len(featured):,} != {expected['rows']:,}"
        )
    anomalies = int(featured["anomaly"].sum())
    if anomalies != expected["anomalies"]:
        raise AssertionError(
            f"Site {target_site} anomaly drift: {anomalies:,} != "
            f"{expected['anomalies']:,}"
        )
    return (
        featured,
        columns,
        {
            "path": artifact_path(cache),
            "source": source,
            "seconds": float(time.perf_counter() - started),
        },
    )


def initialize_tabpfn(model_path: Path):
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        device="cuda",
        # The installed save_fitted_tabpfn_model serializes init parameters to
        # JSON and cannot encode pathlib.WindowsPath. Keep the same checkpoint
        # identity while making fitted-state resume portable.
        model_path=str(model_path),
        random_state=RANDOM_STATE,
        fit_mode="fit_preprocessors",
        memory_saving_mode="auto",
    )


def chunk_path(chunks_dir: Path, chunk_index: int) -> Path:
    return chunks_dir / f"chunk_{chunk_index:04d}.npz"


def save_chunk(
    path: Path,
    *,
    raw_index: np.ndarray,
    probability: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        raw_index=raw_index.astype("int64", copy=False),
        probability=probability.astype("float32", copy=False),
    )
    os.replace(temporary, path)


def load_valid_chunk(path: Path, expected_raw_index: np.ndarray) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path) as payload:
        raw_index = payload["raw_index"]
        probability = payload["probability"]
    if not np.array_equal(raw_index, expected_raw_index):
        raise AssertionError(f"resumed chunk row identity drifted: {path}")
    if (
        len(probability) != len(expected_raw_index)
        or not np.isfinite(probability).all()
    ):
        raise AssertionError(f"resumed chunk predictions are invalid: {path}")
    return probability.astype("float64", copy=False)


def main() -> None:
    args = parse_args()
    if args.query_chunk_size <= 0:
        raise ValueError("--query-chunk-size must be positive")
    contract = DIRECTION_CONTRACT[args.direction]
    target_site = int(contract["target_site"])
    stem = cell_stem(args)
    args.manifest_in = resolve(args.manifest_in)
    args.model_path = resolve(args.model_path)
    args.work_dir = resolve(
        args.work_dir or (ROOT / ".scratch" / "m6_tabpfn_context" / stem)
    )
    args.out = resolve(args.out or (PROC / f"{stem}.json"))
    args.predictions_out = resolve(
        args.predictions_out or (PROC / f"{stem}_predictions.npz")
    )
    progress_path = args.work_dir / "progress.json"
    context_manifest_path = args.work_dir / "context_manifest.json"
    scaler_path = args.work_dir / "scaler.joblib"
    fit_state_path = args.work_dir / "model.tabpfn_fit"
    chunks_dir = args.work_dir / "chunks"
    monitor_path = args.work_dir / f"utilization_session_{int(time.time())}.csv"
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.out.exists():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            log(f"Already completed; refusing to recompute: {args.out}")
            return

    prepared, prepared_sha256 = load_prepared_manifest(args.manifest_in)
    selection = prepared["selection"]
    if int(selection["seed"]) != args.selection_seed:
        raise AssertionError("selection seed differs from prepared B1 manifest")
    requested_budget = str(args.meter_budget).lower()
    actual_budget = (
        "all"
        if int(selection["budget"]) == int(selection["available_meters"])
        else str(selection["budget"])
    )
    if requested_budget != actual_budget:
        raise AssertionError(
            f"meter budget differs from manifest: {requested_budget} != {actual_budget}"
        )

    started = time.perf_counter()
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {
            "schema_version": 1,
            "status": "initializing",
            "stem": stem,
            "completed_chunks": [],
            "chunk_timings": {},
            "sessions": [],
        }
    )
    session = {
        "started_unix": time.time(),
        "monitor_path": artifact_path(monitor_path),
        "resumed_completed_chunks": list(progress.get("completed_chunks", [])),
    }
    progress["sessions"].append(session)
    atomic_json(progress_path, progress)

    with ResourceMonitor(
        monitor_path, interval_seconds=args.monitor_interval
    ) as monitor:
        log(f"Loading M3 frame for {stem}")
        with RegionTimer(use_cuda_events=False) as load_timer:
            frame = load_m3_frame().copy()
            frame["_raw_index"] = np.arange(len(frame), dtype="int64")

        source_mask, held_out_mask, _ = location_split_masks(
            frame, str(contract["split"])
        )
        train_mask = source_mask & meter_manifest_mask(frame, selection)
        target_mask = held_out_mask & (frame["site_id"] == target_site).to_numpy()
        source_raw_index = frame.loc[train_mask, "_raw_index"].to_numpy(
            dtype="int64", copy=True
        )
        source_labels = frame.loc[train_mask, "anomaly"].to_numpy(
            dtype="int8", copy=True
        )
        context_positions = nested_stratified_context_positions(
            source_raw_index,
            source_labels,
            context_rows=args.context_rows,
            seed=stable_context_seed(args.selection_seed),
        )
        context_raw_index = source_raw_index[context_positions]
        context_labels = source_labels[context_positions]
        context_fingerprint = array_fingerprint(context_raw_index)
        context_manifest = {
            "schema_version": 1,
            "direction": args.direction,
            "target_site": target_site,
            "meter_budget": requested_budget,
            "selection_seed": args.selection_seed,
            "context_rows": args.context_rows,
            "context_seed": stable_context_seed(args.selection_seed),
            "available_source_rows": int(train_mask.sum()),
            "available_source_anomalies": int(source_labels.sum()),
            "context_anomalies": int(context_labels.sum()),
            "context_anomaly_rate": float(context_labels.mean()),
            "context_raw_index_sha256": context_fingerprint,
            "prepared_manifest_sha256": prepared_sha256,
            "selected_meter_sha256": selection["selected_meter_sha256"],
        }
        if context_manifest_path.exists():
            frozen_context = json.loads(
                context_manifest_path.read_text(encoding="utf-8")
            )
            if frozen_context != context_manifest:
                raise AssertionError("resumed context manifest drifted")
        else:
            atomic_json(context_manifest_path, context_manifest)

        if fit_state_path.exists() and scaler_path.exists():
            log("Loading saved fitted TabPFN state")
            from tabpfn.model_loading import load_fitted_tabpfn_model

            with RegionTimer() as fit_timer:
                model = load_fitted_tabpfn_model(fit_state_path, device="cuda")
                scaler = joblib.load(scaler_path)
            fit_action = "loaded"
            context_feature_profile = progress.get("context_feature_profile")
        else:
            log(
                f"Building complete-series source features for "
                f"{args.context_rows:,}-row context"
            )
            context_feature_frame, source_feature_cols, context_feature_profile = (
                extract_context_features(frame, train_mask, context_raw_index)
            )
            if not np.array_equal(
                context_feature_frame["_raw_index"].to_numpy(dtype="int64"),
                context_raw_index,
            ):
                raise AssertionError("context feature rows are not in frozen order")
            scaler = StandardScaler()
            with RegionTimer(use_cuda_events=False) as scale_timer:
                x_context = scaler.fit_transform(
                    context_feature_frame[source_feature_cols]
                ).astype("float32", copy=False)
            y_context = context_feature_frame["anomaly"].to_numpy(
                dtype="int8", copy=True
            )
            if not np.array_equal(y_context, context_labels):
                raise AssertionError("context labels drifted during feature extraction")
            del context_feature_frame
            gc.collect()
            log(f"Fitting TabPFN once on {len(x_context):,} source rows")
            model = initialize_tabpfn(args.model_path)
            with RegionTimer() as fit_timer:
                model.fit(x_context, y_context)
            from tabpfn.model_loading import save_fitted_tabpfn_model

            joblib.dump(scaler, scaler_path)
            save_fitted_tabpfn_model(model, fit_state_path)
            fit_action = "fitted"
            progress["scaling_timing"] = timing_dict(scale_timer)
            progress["context_feature_profile"] = context_feature_profile
            del x_context, y_context
            gc.collect()
        progress["fit"] = {
            "action": fit_action,
            "timing": timing_dict(fit_timer),
            "fit_state": artifact_path(fit_state_path),
            "scaler": artifact_path(scaler_path),
        }
        progress["status"] = "fitted"
        atomic_json(progress_path, progress)

        log(f"Loading/building complete Site {target_site} feature cache")
        target_feature_frame, target_feature_cols, target_cache = (
            load_or_build_target_features(
                frame,
                target_mask,
                target_site=target_site,
            )
        )
        if (
            "source_feature_cols" in locals()
            and source_feature_cols != target_feature_cols
        ):
            raise AssertionError("source and target feature order differs")
        del frame, source_raw_index, source_labels
        gc.collect()

        target_raw_index = target_feature_frame["_raw_index"].to_numpy(
            dtype="int64", copy=True
        )
        target_y = target_feature_frame["anomaly"].to_numpy(dtype="int8", copy=True)
        target_site_id = target_feature_frame["site_id"].to_numpy(
            dtype="int16", copy=True
        )
        target_building_id = target_feature_frame["building_id"].to_numpy(
            dtype="int16", copy=True
        )
        target_meter = target_feature_frame["meter"].to_numpy(dtype="int8", copy=True)
        target_timestamp_ns = (
            pd.to_datetime(target_feature_frame["timestamp"])
            .astype("int64")
            .to_numpy(copy=True)
        )

        chunk_count = int(np.ceil(len(target_feature_frame) / args.query_chunk_size))
        run_chunk_count = (
            chunk_count
            if args.max_query_chunks is None
            else min(chunk_count, args.max_query_chunks)
        )
        probabilities: list[np.ndarray] = []
        for chunk_index in range(run_chunk_count):
            start = chunk_index * args.query_chunk_size
            end = min(len(target_feature_frame), start + args.query_chunk_size)
            expected_raw_index = target_raw_index[start:end]
            path = chunk_path(chunks_dir, chunk_index)
            probability = load_valid_chunk(path, expected_raw_index)
            if probability is not None:
                log(f"Chunk {chunk_index + 1}/{chunk_count}: resumed")
                probabilities.append(probability)
                if chunk_index not in progress["completed_chunks"]:
                    progress["completed_chunks"].append(chunk_index)
                continue

            with RegionTimer(use_cuda_events=False) as transform_timer:
                query = scaler.transform(
                    target_feature_frame.iloc[start:end][target_feature_cols]
                ).astype("float32", copy=False)
            with RegionTimer() as predict_timer:
                probability = model.predict_proba(query)[:, 1]
            if not np.isfinite(probability).all():
                raise RuntimeError(
                    f"non-finite TabPFN probabilities in chunk {chunk_index}"
                )
            save_chunk(
                path,
                raw_index=expected_raw_index,
                probability=probability,
            )
            progress["completed_chunks"].append(chunk_index)
            progress["completed_chunks"] = sorted(set(progress["completed_chunks"]))
            progress["chunk_timings"][str(chunk_index)] = {
                "rows": int(end - start),
                "transform": timing_dict(transform_timer),
                "predict": timing_dict(predict_timer),
            }
            progress["status"] = "predicting"
            atomic_json(progress_path, progress)
            probabilities.append(np.asarray(probability, dtype="float64"))
            log(
                f"Chunk {chunk_index + 1}/{chunk_count}: "
                f"wall={predict_timer.result.wall_seconds:.2f}s "
                f"gpu={predict_timer.result.gpu_seconds}"
            )

        completed_full_site = run_chunk_count == chunk_count
        if not completed_full_site:
            progress["status"] = "partial_smoke"
            progress["partial_rows"] = int(sum(len(value) for value in probabilities))
            session["elapsed_seconds"] = float(time.perf_counter() - started)
            session["monitor"] = monitor.summary()
            atomic_json(progress_path, progress)
            log(
                f"Saved partial smoke after {run_chunk_count}/{chunk_count} chunks: "
                f"{progress_path}"
            )
            return

        probability = np.concatenate(probabilities)
        if len(probability) != len(target_y):
            raise AssertionError("whole-site probability reconstruction is incomplete")
        metrics = evaluation_summary(target_y, probability)
        curves = curve_summary(target_y, probability)
        histograms = score_histograms(
            target_y,
            target_site_id,
            {"tabpfn": probability},
        )
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        temporary_predictions = args.predictions_out.with_name(
            args.predictions_out.stem + ".tmp.npz"
        )
        np.savez_compressed(
            temporary_predictions,
            validation_raw_index=target_raw_index,
            timestamp_ns=target_timestamp_ns,
            site_id=target_site_id,
            building_id=target_building_id,
            meter=target_meter,
            anomaly=target_y,
            tabpfn=probability.astype("float32"),
        )
        os.replace(temporary_predictions, args.predictions_out)

        session["elapsed_seconds"] = float(time.perf_counter() - started)
        session["monitor"] = monitor.summary()
        progress["status"] = "completed"
        atomic_json(progress_path, progress)

    payload = {
        "schema_version": 1,
        "experiment": "m6_tabpfn_context_curve",
        "status": "completed",
        "direction": args.direction,
        "target_site": target_site,
        "meter_budget": requested_budget,
        "selection_seed": args.selection_seed,
        "context": context_manifest,
        "query": {
            "rows": int(len(target_y)),
            "anomalies": int(target_y.sum()),
            "chunk_size": args.query_chunk_size,
            "chunk_count": chunk_count,
            "raw_index_sha256": array_fingerprint(target_raw_index),
        },
        "model": {
            "name": "TabPFNClassifier",
            "model_seed": RANDOM_STATE,
            "model_path": artifact_path(args.model_path),
            "fit_mode": "fit_preprocessors",
            "memory_saving_mode": "auto",
            "fit_once_per_cell": True,
        },
        "features": {
            "value_change_regime": VALUE_CHANGE_REGIME,
            "count": len(target_feature_cols),
            "names": target_feature_cols,
            "target_cache": target_cache,
            "context_profile": context_feature_profile,
            "scaler": "sklearn.preprocessing.StandardScaler",
        },
        "metrics": {"tabpfn": metrics},
        "curves": {"tabpfn": curves},
        "score_histograms": histograms,
        "timing": {
            "load_frame": timing_dict(load_timer),
            "fit_or_load": progress["fit"],
            "chunk_timings": progress["chunk_timings"],
            "sessions": progress["sessions"],
            "elapsed_seconds": float(time.perf_counter() - started),
            "semantics": {
                "wall": "time.perf_counter",
                "cpu": "psutil process user/system time",
                "gpu": "synchronized torch.cuda.Event elapsed time",
            },
        },
        "environment": host_environment(),
        "artifacts": {
            "prepared_manifest": artifact_path(args.manifest_in),
            "context_manifest": artifact_path(context_manifest_path),
            "progress": artifact_path(progress_path),
            "predictions": artifact_path(args.predictions_out),
            "fit_state": artifact_path(fit_state_path),
            "utilization_sessions": [
                item["monitor_path"] for item in progress["sessions"]
            ],
        },
        "target_label_contract": {
            "target_labels_used_for_fit": False,
            "target_labels_used_for_capacity_or_stopping": False,
            "target_labels_used_for_evaluation_only": True,
        },
    }
    write_json_with_provenance(
        args.out,
        payload,
        root=ROOT,
        provenance={
            "note": (
                "M6 Site 1/Site 8 TabPFN context-size cell; target site unseen; "
                "prepared B1 meter manifest reused."
            )
        },
    )
    log(
        f"Completed Site {target_site} {requested_budget=} "
        f"context={args.context_rows:,}: {args.out}"
    )


if __name__ == "__main__":
    main()
