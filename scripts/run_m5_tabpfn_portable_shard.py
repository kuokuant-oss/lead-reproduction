"""Predict one portable TabPFN test shard with atomic resumable checkpoints.

This file is intentionally self-contained so it can be uploaded to a Colab VM.
Torch and TabPFN are imported only inside the worker entry point. There is no
wall-time timeout; resource pressure reduces only the query microbatch, while
every completed disk checkpoint remains reusable after interruption.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


PORTABLE_DEFAULT_MICROBATCH = 1_024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--fit-state", type=Path, required=True)
    parser.add_argument(
        "--scaler",
        type=Path,
        default=None,
        help=(
            "npz with 'mean' and 'scale'; applied per microbatch when --features "
            "holds unscaled values. Omit when the matrix is already standardised."
        ),
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--context-rows", type=int, default=100_000)
    # The defaults are the canonical 17-feature contract. The 137-feature line
    # and the estimator sweep override them, and the fitted state must agree.
    parser.add_argument("--n-features", type=int, default=17)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument(
        "--query-microbatch-size", type=int, default=PORTABLE_DEFAULT_MICROBATCH
    )
    parser.add_argument("--min-query-microbatch-size", type=int, default=256)
    parser.add_argument("--checkpoint-rows", type=int, default=20_000)
    parser.add_argument(
        "--direction", choices=("forward", "reverse"), default="reverse"
    )
    parser.add_argument("--gpu-soft-limit-fraction", type=float, default=0.86)
    parser.add_argument("--gpu-hard-limit-fraction", type=float, default=0.92)
    parser.add_argument("--ram-soft-limit-fraction", type=float, default=0.85)
    parser.add_argument("--ram-hard-limit-fraction", type=float, default=0.92)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.context_rows <= 0:
        raise ValueError("context rows must be positive")
    if args.n_features <= 0:
        raise ValueError("feature count must be positive")
    if args.n_estimators <= 0:
        raise ValueError("estimator count must be positive")
    if not 0 < args.min_query_microbatch_size <= args.query_microbatch_size:
        raise ValueError("invalid query microbatch range")
    if args.checkpoint_rows < args.query_microbatch_size:
        raise ValueError("checkpoint rows must be at least one query microbatch")
    for fraction in (
        args.gpu_soft_limit_fraction,
        args.gpu_hard_limit_fraction,
        args.ram_soft_limit_fraction,
        args.ram_hard_limit_fraction,
    ):
        if not 0 < fraction <= 1:
            raise ValueError("resource fractions must be in (0, 1]")
    if args.gpu_soft_limit_fraction >= args.gpu_hard_limit_fraction:
        raise ValueError("GPU soft limit must be below hard limit")
    if args.ram_soft_limit_fraction >= args.ram_hard_limit_fraction:
        raise ValueError("RAM soft limit must be below hard limit")
    return args


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_spans(
    rows: int, checkpoint_rows: int, direction: str
) -> list[tuple[int, int]]:
    spans = [
        (start, min(rows, start + checkpoint_rows))
        for start in range(0, rows, checkpoint_rows)
    ]
    return spans if direction == "forward" else list(reversed(spans))


def checkpoint_path(directory: Path, global_start: int, global_end: int) -> Path:
    return directory / f"rows_{global_start:08d}_{global_end:08d}.npz"


def load_metadata(path: Path) -> dict[str, np.ndarray]:
    required = {
        "raw_index",
        "anomaly",
        "site_id",
        "building_id",
        "global_position",
    }
    with np.load(path) as payload:
        if missing := required - set(payload.files):
            raise ValueError(f"metadata missing arrays: {sorted(missing)}")
        result = {name: np.asarray(payload[name]) for name in required}
    lengths = {len(value) for value in result.values()}
    if len(lengths) != 1:
        raise ValueError("metadata arrays have different lengths")
    positions = result["global_position"].astype("int64", copy=False)
    if len(positions) and not np.array_equal(
        positions, np.arange(positions[0], positions[0] + len(positions))
    ):
        raise ValueError("global positions are not contiguous and ascending")
    if len(np.unique(result["raw_index"])) != len(result["raw_index"]):
        raise ValueError("raw row IDs are not unique")
    return result


def load_saved_checkpoint(
    path: Path, expected_raw_index: np.ndarray
) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path) as payload:
        required = {"raw_index", "anomaly", "score", "site_id", "building_id"}
        if missing := required - set(payload.files):
            raise ValueError(f"checkpoint missing arrays: {sorted(missing)}")
        result = {name: np.asarray(payload[name]) for name in required}
    if not np.array_equal(result["raw_index"], expected_raw_index):
        raise AssertionError(f"checkpoint row identity drifted: {path}")
    if not np.isfinite(result["score"]).all():
        raise AssertionError(f"checkpoint score contains non-finite values: {path}")
    return result


def is_oom(error: BaseException) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return isinstance(error, MemoryError) or any(
        token in text for token in ("outofmemory", "out of memory", "cuda oom")
    )


def verify_fitted_model(model: Any, context_rows: int, estimators: int = 1) -> None:
    config = getattr(model, "inference_config_", None)
    observed = {
        "context_rows": getattr(model, "n_train_samples_", None),
        "estimators": getattr(model, "n_estimators_", None),
        "subsample_samples": getattr(config, "SUBSAMPLE_SAMPLES", "missing"),
    }
    expected = {
        "context_rows": context_rows,
        "estimators": estimators,
        "subsample_samples": None,
    }
    if observed != expected:
        raise AssertionError(f"fitted model contract drifted: {observed} != {expected}")


def resource_snapshot(torch: Any, psutil: Any) -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    ram = psutil.virtual_memory()
    return {
        "gpu_used_fraction": float((total_bytes - free_bytes) / total_bytes),
        "gpu_used_mib": float((total_bytes - free_bytes) / 1024**2),
        "gpu_total_mib": float(total_bytes / 1024**2),
        "torch_allocated_mib": float(torch.cuda.memory_allocated() / 1024**2),
        "torch_reserved_mib": float(torch.cuda.memory_reserved() / 1024**2),
        "ram_used_fraction": float(ram.percent / 100),
        "ram_used_mib": float(ram.used / 1024**2),
        "ram_total_mib": float(ram.total / 1024**2),
    }


def predict_checkpoint(
    model: Any,
    features: np.ndarray,
    *,
    initial_batch_size: int,
    minimum_batch_size: int,
    args: argparse.Namespace,
    torch: Any,
    psutil: Any,
    heartbeat_path: Path,
    completed_before: int,
    scaler: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, int, dict[str, float]]:
    batch_size = initial_batch_size
    predictions: list[np.ndarray] = []
    position = 0
    peak: dict[str, float] = {}
    while position < len(features):
        snapshot = resource_snapshot(torch, psutil)
        for name, value in snapshot.items():
            peak[name] = max(peak.get(name, value), value)
        if snapshot["ram_used_fraction"] >= args.ram_hard_limit_fraction:
            raise MemoryError("system RAM hard limit exceeded")
        if snapshot["gpu_used_fraction"] >= args.gpu_hard_limit_fraction:
            if batch_size <= minimum_batch_size:
                raise MemoryError("GPU hard limit exceeded at minimum microbatch")
            batch_size = max(minimum_batch_size, batch_size // 2)
            gc.collect()
            torch.cuda.empty_cache()
            continue
        if (
            snapshot["gpu_used_fraction"] >= args.gpu_soft_limit_fraction
            or snapshot["ram_used_fraction"] >= args.ram_soft_limit_fraction
        ) and batch_size > minimum_batch_size:
            batch_size = max(minimum_batch_size, batch_size // 2)
            gc.collect()
            torch.cuda.empty_cache()

        end = min(len(features), position + batch_size)
        atomic_write_json(
            heartbeat_path,
            {
                "status": "predicting",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "completed_rows": completed_before + position,
                "microbatch_size": batch_size,
                "resource": snapshot,
            },
        )
        try:
            block = np.asarray(features[position:end], dtype="float32")
            if scaler is not None:
                # Standardising here rather than at export time lets one
                # unscaled matrix serve every context in the curve: the contexts
                # differ only by these 2 x n_features numbers.
                block = ((block - scaler[0]) / scaler[1]).astype("float32", copy=False)
            score = model.predict_proba(block)[:, 1]
        except BaseException as error:
            if not is_oom(error) or batch_size <= minimum_batch_size:
                raise
            batch_size = max(minimum_batch_size, batch_size // 2)
            gc.collect()
            torch.cuda.empty_cache()
            continue
        predictions.append(np.asarray(score, dtype="float32"))
        position = end
    return np.concatenate(predictions), batch_size, peak


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.work_dir / "chunks"
    heartbeat_path = args.work_dir / "heartbeat.json"
    progress_path = args.work_dir / "progress.json"
    result_path = args.work_dir / "result.json"
    if progress_path.exists() and not args.resume:
        raise FileExistsError("existing progress; pass --resume")

    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    os.environ["TABPFN_NO_BROWSER"] = "1"
    import psutil
    import torch
    from tabpfn.model_loading import load_fitted_tabpfn_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback forbidden")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    stage = "loading"
    model = features = metadata = None
    try:
        features = np.load(args.features, mmap_mode="r")
        if features.ndim != 2 or features.dtype != np.dtype("float32"):
            raise ValueError("features must be a two-dimensional float32 .npy")
        metadata = load_metadata(args.metadata)
        if len(features) != len(metadata["raw_index"]):
            raise ValueError("feature and metadata row counts differ")
        if features.shape[1] != args.n_features:
            raise ValueError(
                f"portable shard requires exactly {args.n_features} features, "
                f"got {features.shape[1]}"
            )

        scaler = None
        if args.scaler is not None:
            with np.load(args.scaler) as payload:
                missing = {"mean", "scale"} - set(payload.files)
                if missing:
                    raise ValueError(f"scaler npz missing {sorted(missing)}")
                mean = np.asarray(payload["mean"], dtype="float32")
                scale = np.asarray(payload["scale"], dtype="float32")
            if mean.shape != (args.n_features,) or scale.shape != (args.n_features,):
                raise ValueError(
                    f"scaler must hold {args.n_features} means and scales, got "
                    f"{mean.shape} and {scale.shape}"
                )
            if not np.isfinite(mean).all() or not np.isfinite(scale).all():
                raise ValueError("scaler contains non-finite values")
            if (scale == 0).any():
                raise ValueError("scaler has a zero scale; would divide by zero")
            scaler = (mean, scale)

        stage = "load_fitted_state"
        atomic_write_json(
            heartbeat_path,
            {"status": stage, "timestamp": time.time(), "pid": os.getpid()},
        )
        model = load_fitted_tabpfn_model(args.fit_state, device="cuda")
        verify_fitted_model(model, args.context_rows, args.n_estimators)

        spans = checkpoint_spans(len(features), args.checkpoint_rows, args.direction)
        previous_progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.is_file()
            else {}
        )
        completed_rows = 0
        completed_spans: list[list[int]] = []
        timings: dict[str, Any] = dict(previous_progress.get("timings", {}))
        # The requested microbatch wins on resume. Inheriting the previous value
        # was meant to remember a pressure-driven downshift, but it also silently
        # discards a deliberately raised value, pinning a resumed shard to the
        # old size forever. Re-discovering a downshift costs one halving step,
        # which the soft/hard limit checks below apply within the first batch.
        effective_batch = int(
            args.query_microbatch_size
            if args.query_microbatch_size != PORTABLE_DEFAULT_MICROBATCH
            else previous_progress.get(
                "effective_microbatch_size", args.query_microbatch_size
            )
        )
        peak_resources: dict[str, float] = dict(
            previous_progress.get("peak_resources", {})
        )
        progress: dict[str, Any] = {
            "status": "predicting",
            "direction": args.direction,
            "rows": len(features),
            "completed_rows": 0,
            "completed_fraction": 0.0,
            "completed_spans": [],
            "timings": timings,
            "effective_microbatch_size": effective_batch,
            "peak_resources": peak_resources,
        }
        for local_start, local_end in spans:
            raw_index = metadata["raw_index"][local_start:local_end]
            global_start = int(metadata["global_position"][local_start])
            global_end = int(metadata["global_position"][local_end - 1]) + 1
            path = checkpoint_path(chunks_dir, global_start, global_end)
            saved = load_saved_checkpoint(path, raw_index)
            if saved is not None:
                completed_rows += local_end - local_start
                completed_spans.append([global_start, global_end])
                progress.update(
                    {
                        "completed_rows": completed_rows,
                        "completed_fraction": completed_rows / len(features),
                        "completed_spans": completed_spans,
                    }
                )
                continue

            stage = "predict"
            checkpoint_started = time.perf_counter()
            score, effective_batch, peak = predict_checkpoint(
                model,
                features[local_start:local_end],
                initial_batch_size=effective_batch,
                minimum_batch_size=args.min_query_microbatch_size,
                args=args,
                torch=torch,
                psutil=psutil,
                heartbeat_path=heartbeat_path,
                completed_before=completed_rows,
                scaler=scaler,
            )
            checkpoint_seconds = time.perf_counter() - checkpoint_started
            atomic_write_npz(
                path,
                raw_index=raw_index.astype("int64", copy=False),
                anomaly=metadata["anomaly"][local_start:local_end].astype(
                    "int8", copy=False
                ),
                score=score,
                site_id=metadata["site_id"][local_start:local_end].astype(
                    "int8", copy=False
                ),
                building_id=metadata["building_id"][local_start:local_end].astype(
                    "int16", copy=False
                ),
            )
            completed_rows += local_end - local_start
            completed_spans.append([global_start, global_end])
            timings[f"{global_start}:{global_end}"] = {
                "rows": local_end - local_start,
                "seconds": checkpoint_seconds,
                "rows_per_second": (local_end - local_start) / checkpoint_seconds,
                "effective_microbatch_size": effective_batch,
            }
            for name, value in peak.items():
                peak_resources[name] = max(peak_resources.get(name, value), value)
            progress = {
                "status": "predicting",
                "direction": args.direction,
                "rows": len(features),
                "completed_rows": completed_rows,
                "completed_fraction": completed_rows / len(features),
                "completed_spans": completed_spans,
                "timings": timings,
                "effective_microbatch_size": effective_batch,
                "peak_resources": peak_resources,
            }
            atomic_write_json(progress_path, progress)
            del score
            gc.collect()

        result = {
            "status": "completed",
            "direction": args.direction,
            "rows": len(features),
            "completed_rows": completed_rows,
            "checkpoint_count": len(spans),
            "n_features": int(features.shape[1]),
            "n_estimators": args.n_estimators,
            "context_rows": args.context_rows,
            # Which standardisation produced these scores. Without it, a shard
            # scored from an unscaled matrix and one scored from a pre-scaled
            # matrix are indistinguishable after the fact.
            "scaler_applied_at_predict": args.scaler is not None,
            "scaler_sha256": (
                sha256_file(args.scaler) if args.scaler is not None else None
            ),
            "effective_microbatch_size": effective_batch,
            "elapsed_seconds_this_session": time.perf_counter() - started,
            "fit_state_sha256": sha256_file(args.fit_state),
            "metadata_sha256": sha256_file(args.metadata),
            "features_sha256": sha256_file(args.features),
            "peak_resources": peak_resources,
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        }
        atomic_write_json(result_path, result)
        atomic_write_json(
            progress_path,
            {**progress, "status": "completed", "completed_fraction": 1.0},
        )
        atomic_write_json(
            heartbeat_path,
            {"status": "completed", "timestamp": time.time(), "pid": os.getpid()},
        )
        return 0
    except BaseException as error:
        atomic_write_json(
            result_path,
            {
                "status": "failed",
                "stage": stage,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            },
        )
        raise
    finally:
        model = features = metadata = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def parent_has_forbidden_imports() -> bool:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden = {"torch", "tabpfn"}
    for node in tree.body:
        if isinstance(node, ast.Import) and any(
            alias.name.split(".")[0] in forbidden for alias in node.names
        ):
            return True
        if (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] in forbidden
        ):
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
