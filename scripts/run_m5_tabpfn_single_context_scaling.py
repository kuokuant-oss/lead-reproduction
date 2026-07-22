"""Resource-guarded TabPFN-3 single-context scaling experiment.

The controller never imports torch or TabPFN. Preflight and every budget run in
disposable subprocesses so failed CUDA contexts cannot strand the controller.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from lead import BASELINE_FEATURE_COLS, PROC, RANDOM_STATE, ROOT, load_m3_frame
from lead.resource_guard import (
    LimitTracker,
    append_jsonl,
    atomic_write_json,
    pid_exists,
    query_gpu_memory,
    resolve_limits,
    sample_dict,
    sample_resources,
    terminate_process_tree,
)


EXPERIMENT = "tabpfn_v3_single_context_scaling"
DEFAULT_BUDGETS = [100_000, 200_000, 300_000, 400_000, 500_000]
OOM_EXIT_CODE = 42


def default_model_path() -> Path:
    cache = os.environ.get("TABPFN_MODEL_CACHE_DIR")
    return (Path(cache) if cache else ROOT / ".tabpfn-cache") / (
        "tabpfn-v3-classifier-v3_default.ckpt"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--score-rows", type=int, default=4_000)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--min-predict-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument(
        "--out", type=Path, default=PROC / "m5_tabpfn_single_context_scaling.json"
    )
    parser.add_argument(
        "--state-out",
        type=Path,
        default=PROC / "m5_tabpfn_single_context_scaling.state.json",
    )
    parser.add_argument(
        "--events-out",
        type=Path,
        default=PROC / "m5_tabpfn_single_context_scaling.events.jsonl",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        help="Directory for small per-budget scoring prediction artifacts",
    )
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
    parser.add_argument("--budget-timeout-minutes", type=float, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart-budget", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-after-failure", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-budget", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--heartbeat", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stop-request", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.budgets = sorted(set(int(value) for value in args.budgets))
    if not args.budgets or any(value <= 0 or value % 2 for value in args.budgets):
        raise ValueError("budgets must be positive even integers")
    if not 0 < args.min_predict_batch_size <= args.predict_batch_size:
        raise ValueError("minimum prediction batch must be in (0, initial batch]")
    if args.restart_budget is not None and args.restart_budget not in args.budgets:
        raise ValueError("restart budget must appear in --budgets")
    if args.predictions_dir is None:
        args.predictions_dir = args.out.with_name(args.out.stem + ".predictions")
    return args


def nested_balanced_indices(
    row_index: np.ndarray,
    labels: np.ndarray,
    budgets: list[int],
    *,
    seed: int,
) -> dict[int, np.ndarray]:
    """Build deterministic balanced prefixes without replacement."""
    rows = np.asarray(row_index, dtype="int64")
    y = np.asarray(labels)
    if len(rows) != len(y):
        raise ValueError("row_index and labels must have equal length")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("source row identities must be unique")
    ordered = sorted(set(int(value) for value in budgets))
    if not ordered or any(value <= 0 or value % 2 for value in ordered):
        raise ValueError("budgets must be positive even integers")
    per_class = ordered[-1] // 2
    positive = rows[y == 1].copy()
    negative = rows[y == 0].copy()
    if len(positive) < per_class or len(negative) < per_class:
        raise ValueError(
            "insufficient unique class support for balanced sampling without "
            f"replacement: need {per_class:,} per class, found "
            f"positive={len(positive):,}, negative={len(negative):,}"
        )
    rng = np.random.RandomState(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    maximum = np.empty(ordered[-1], dtype="int64")
    maximum[0::2] = positive[:per_class]
    maximum[1::2] = negative[:per_class]
    return {budget: maximum[:budget].copy() for budget in ordered}


def fixed_score_indices(row_index: np.ndarray, rows: int, *, seed: int) -> np.ndarray:
    values = np.asarray(row_index, dtype="int64")
    if len(values) <= rows:
        return values.copy()
    return (
        np.random.RandomState(seed)
        .choice(values, rows, replace=False)
        .astype("int64", copy=False)
    )


def index_record(indices: np.ndarray, labels_by_row: dict[int, int]) -> dict[str, Any]:
    values = np.asarray(indices, dtype="int64")
    labels = np.asarray([labels_by_row[int(row)] for row in values], dtype="int8")
    return {
        "count": int(len(values)),
        "unique_count": int(len(np.unique(values))),
        "sha256": hashlib.sha256(
            values.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "first_10": [int(value) for value in values[:10]],
        "last_10": [int(value) for value in values[-10:]],
        "class_counts": {
            "negative": int((labels == 0).sum()),
            "positive": int((labels == 1).sum()),
        },
    }


def build_split(frame: Any) -> dict[str, Any]:
    masks = {
        "fit": (frame["building_id"] % 4 == 0).to_numpy(),
        "validation": (frame["building_id"] % 4 == 2).to_numpy(),
        "test": (frame["building_id"] % 2 == 1).to_numpy(),
    }
    buildings = {
        name: set(int(value) for value in frame.loc[mask, "building_id"].unique())
        for name, mask in masks.items()
    }
    for left, right in (
        ("fit", "validation"),
        ("fit", "test"),
        ("validation", "test"),
    ):
        overlap = buildings[left] & buildings[right]
        if overlap:
            raise AssertionError(
                f"{left}/{right} building overlap: {sorted(overlap)[:5]}"
            )
    return {
        **{f"{name}_mask": mask for name, mask in masks.items()},
        "metadata": {
            "name": "50_50_mod2_with_nested_mod4_val",
            "fit_rule": "building_id % 4 == 0",
            "validation_rule": "building_id % 4 == 2",
            "test_rule": "building_id % 2 == 1",
            "n_buildings": {name: len(value) for name, value in buildings.items()},
            "n_rows": {name: int(mask.sum()) for name, mask in masks.items()},
            "building_overlaps": {
                "fit_validation": 0,
                "fit_test": 0,
                "validation_test": 0,
            },
        },
    }


def prepare_row_contract(frame: Any, args: argparse.Namespace) -> dict[str, Any]:
    if len(BASELINE_FEATURE_COLS) != 17:
        raise AssertionError(
            f"expected 17 baseline features, got {len(BASELINE_FEATURE_COLS)}"
        )
    split = build_split(frame)
    fit = frame.loc[split["fit_mask"]]
    validation = frame.loc[split["validation_mask"]]
    test = frame.loc[split["test_mask"]]
    fit_indices = nested_balanced_indices(
        fit.index.to_numpy(), fit["anomaly"].to_numpy(), args.budgets, seed=args.seed
    )
    val_indices = fixed_score_indices(
        validation.index.to_numpy(), args.score_rows, seed=args.seed + 20_000
    )
    test_indices = fixed_score_indices(
        test.index.to_numpy(), args.score_rows, seed=args.seed + 30_000
    )
    label_map = {
        int(row): int(label)
        for row, label in zip(fit.index, fit["anomaly"], strict=True)
    }
    return {
        "split": split,
        "fit_indices": fit_indices,
        "validation_indices": val_indices,
        "test_indices": test_indices,
        "budget_records": {
            str(budget): index_record(indices, label_map)
            for budget, indices in fit_indices.items()
        },
        "score_records": {
            "validation": score_record(val_indices, validation),
            "test": score_record(test_indices, test),
        },
    }


def score_record(indices: np.ndarray, frame: Any) -> dict[str, Any]:
    values = np.asarray(indices, dtype="int64")
    return {
        "count": int(len(values)),
        "sha256": hashlib.sha256(
            values.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "first_10": [int(value) for value in values[:10]],
        "last_10": [int(value) for value in values[-10:]],
        "prevalence": float(frame.loc[values, "anomaly"].mean()),
    }


def synthetic_frame(rows_per_building: int = 200) -> Any:
    import pandas as pd

    rng = np.random.RandomState(1234)
    building_id = np.repeat(np.arange(24, dtype="int16"), rows_per_building)
    size = len(building_id)
    data: dict[str, Any] = {
        "building_id": building_id,
        "site_id": building_id % 16,
        "anomaly": np.arange(size) % 2,
    }
    for position, feature in enumerate(BASELINE_FEATURE_COLS):
        data[feature] = (rng.normal(size=size) + position / 20).astype("float32")
    return pd.DataFrame(data)


def load_experiment_frame(*, smoke: bool) -> Any:
    return synthetic_frame() if smoke else load_m3_frame(verbose=True)


def make_matrices(frame: Any, contract: dict[str, Any], budget: int) -> dict[str, Any]:
    fit_idx = contract["fit_indices"][budget]
    val_idx = contract["validation_indices"]
    test_idx = contract["test_indices"]
    x_fit = frame.loc[fit_idx, BASELINE_FEATURE_COLS].to_numpy(
        dtype="float32", copy=True
    )
    y_fit = frame.loc[fit_idx, "anomaly"].to_numpy(dtype="int64", copy=True)
    x_val = frame.loc[val_idx, BASELINE_FEATURE_COLS].to_numpy(
        dtype="float32", copy=True
    )
    y_val = frame.loc[val_idx, "anomaly"].to_numpy(dtype="int64", copy=True)
    x_test = frame.loc[test_idx, BASELINE_FEATURE_COLS].to_numpy(
        dtype="float32", copy=True
    )
    y_test = frame.loc[test_idx, "anomaly"].to_numpy(dtype="int64", copy=True)
    scaler = StandardScaler(copy=False)
    x_fit = scaler.fit_transform(x_fit).astype("float32", copy=False)
    x_val = scaler.transform(x_val).astype("float32", copy=False)
    x_test = scaler.transform(x_test).astype("float32", copy=False)
    return {
        "x_fit": x_fit,
        "y_fit": y_fit,
        "x_validation": x_val,
        "y_validation": y_val,
        "validation_row_index": np.asarray(val_idx, dtype="int64"),
        "validation_building_id": frame.loc[val_idx, "building_id"].to_numpy(
            dtype="int16", copy=True
        ),
        "validation_site_id": frame.loc[val_idx, "site_id"].to_numpy(
            dtype="int8", copy=True
        ),
        "x_test": x_test,
        "y_test": y_test,
        "test_row_index": np.asarray(test_idx, dtype="int64"),
        "test_building_id": frame.loc[test_idx, "building_id"].to_numpy(
            dtype="int16", copy=True
        ),
        "test_site_id": frame.loc[test_idx, "site_id"].to_numpy(
            dtype="int8", copy=True
        ),
    }


class FakeTabPFNClassifier:
    """Deterministic classifier for subprocess smoke testing only."""

    def __init__(self) -> None:
        self.n_estimators = 1
        self.inference_config_ = type("Config", (), {"SUBSAMPLE_SAMPLES": None})()

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FakeTabPFNClassifier":
        self.n_train_samples_ = int(len(x))
        self.n_estimators_ = 1
        self.offset_ = float(np.nanmean(x[:, 0]))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        score = 1 / (1 + np.exp(-(x[:, 0] - self.offset_)))
        return np.column_stack([1 - score, score])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    """Atomically persist the small scoring arrays needed for later curves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def prediction_artifact_path(args: argparse.Namespace, budget: int) -> Path:
    return args.predictions_dir / f"budget-{budget}-scoring-predictions.npz"


def tabpfn_api_evidence(model_path: Path) -> dict[str, Any]:
    """Inspect active checkpoint config and source; worker-only."""
    import tabpfn
    from tabpfn import TabPFNClassifier

    signature = inspect.signature(TabPFNClassifier)
    required = {
        "n_estimators",
        "auto_scale_n_estimators",
        "model_path",
        "device",
        "ignore_pretraining_limits",
        "fit_mode",
        "memory_saving_mode",
        "random_state",
        "n_preprocessing_jobs",
        "inference_config",
    }
    missing = sorted(required - set(signature.parameters))
    source_file = Path(inspect.getsourcefile(TabPFNClassifier) or "")
    source = source_file.read_text(encoding="utf-8") if source_file.is_file() else ""
    routed = "subsample_samples=self.inference_config_.SUBSAMPLE_SAMPLES" in source
    probe = TabPFNClassifier(
        n_estimators=1,
        auto_scale_n_estimators=False,
        model_path=model_path,
        device="cpu",
        inference_config={"SUBSAMPLE_SAMPLES": None},
    )
    config = probe.get_inference_config()
    verified = not missing and config.SUBSAMPLE_SAMPLES is None and routed
    return {
        "status": "verified" if verified else "blocked_unverified_context",
        "version": str(tabpfn.__version__),
        "constructor_signature": str(signature),
        "missing_required_parameters": missing,
        "inference_config": {"SUBSAMPLE_SAMPLES": config.SUBSAMPLE_SAMPLES},
        "source_file": str(source_file),
        "source_routes_subsample_config": routed,
        "checkpoint_path": str(model_path.resolve()),
        "checkpoint_sha256": sha256_file(model_path) if model_path.is_file() else None,
    }


def tabpfn_constructor(model_path: Path, seed: int) -> tuple[Any, dict[str, Any]]:
    from tabpfn import TabPFNClassifier

    kwargs = {
        "n_estimators": 1,
        "auto_scale_n_estimators": False,
        "model_path": model_path,
        "device": "cuda",
        "ignore_pretraining_limits": True,
        "fit_mode": "low_memory",
        "memory_saving_mode": True,
        "keep_cache_on_device": False,
        "random_state": seed,
        "n_preprocessing_jobs": 1,
        "inference_config": {"SUBSAMPLE_SAMPLES": None},
        "show_progress_bar": False,
    }
    return TabPFNClassifier(**kwargs), kwargs


def verify_fitted_context(model: Any, requested_rows: int) -> dict[str, Any]:
    rows = getattr(model, "n_train_samples_", None)
    estimators = getattr(model, "n_estimators_", None)
    config = getattr(model, "inference_config_", None)
    subsampling = getattr(config, "SUBSAMPLE_SAMPLES", "unavailable")
    verified = (
        rows == requested_rows
        and estimators == 1
        and subsampling is None
        and getattr(model, "n_estimators", None) == 1
    )
    return {
        "status": "verified" if verified else "blocked_unverified_context",
        "requested_context_rows": int(requested_rows),
        "effective_context_rows": int(rows) if rows is not None else None,
        "external_sharding": False,
        "sample_subsampling": subsampling,
        "sample_subsampling_disabled": subsampling is None,
        "effective_estimators": estimators,
    }


class Heartbeat:
    def __init__(
        self, path: Path | None, budget: int, torch_module: Any = None
    ) -> None:
        self.path = path
        self.budget = budget
        self.torch = torch_module
        self.stage = "starting"
        self.position = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def update(self, stage: str, position: int = 0) -> None:
        self.stage, self.position = stage, int(position)
        self.write()

    def write(self) -> None:
        if self.path is None:
            return
        allocated = reserved = None
        if self.torch is not None and self.torch.cuda.is_available():
            allocated = int(self.torch.cuda.memory_allocated())
            reserved = int(self.torch.cuda.memory_reserved())
        atomic_write_json(
            self.path,
            {
                "pid": os.getpid(),
                "budget": self.budget,
                "stage": self.stage,
                "timestamp": time.time(),
                "prediction_batch_position": self.position,
                "torch_allocated_bytes": allocated,
                "torch_reserved_bytes": reserved,
            },
        )

    def __enter__(self) -> "Heartbeat":
        def pulse() -> None:
            while not self._stop.wait(1):
                self.write()

        self.write()
        self._thread = threading.Thread(target=pulse, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.write()


def prediction_batch_sizes(initial: int, minimum: int) -> list[int]:
    values = [int(initial)]
    while values[-1] > minimum:
        values.append(max(minimum, values[-1] // 2))
    return values


def is_oom_exception(error: BaseException) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return (
        isinstance(error, MemoryError)
        or "outofmemory" in text
        or "out of memory" in text
        or "cuda oom" in text
    )


def batched_predict(
    model: Any,
    matrix: np.ndarray,
    *,
    initial_batch_size: int,
    minimum_batch_size: int,
    stop_requested: Callable[[], bool],
    heartbeat: Heartbeat,
    stage: str,
) -> tuple[np.ndarray, int]:
    for batch_size in prediction_batch_sizes(initial_batch_size, minimum_batch_size):
        predictions: list[np.ndarray] = []
        try:
            for start in range(0, len(matrix), batch_size):
                if stop_requested():
                    raise InterruptedError("resource watchdog requested graceful stop")
                heartbeat.update(stage, start)
                predictions.append(
                    np.asarray(
                        model.predict_proba(matrix[start : start + batch_size])[:, 1],
                        dtype="float64",
                    )
                )
            return np.concatenate(predictions), batch_size
        except BaseException as error:
            if not is_oom_exception(error) or batch_size <= minimum_batch_size:
                raise
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
    raise AssertionError("prediction batching produced no attempt")


def fixed_recall_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    _, recall, thresholds = precision_recall_curve(y, scores)
    candidates = thresholds[recall[:-1] >= 0.9]
    return float(candidates.max()) if len(candidates) else 0.0


def evaluation_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
    }


def operating_metrics(
    y: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    labels = (scores >= threshold).astype("int8")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, labels, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, labels, labels=[0, 1]).ravel()
    return {
        "threshold_from_validation_fixed_recall_0_90": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def serialize_failure(
    error: BaseException,
    *,
    budget: int,
    stage: str,
    predict_batch_size: int,
    torch_module: Any = None,
) -> dict[str, Any]:
    allocated = reserved = None
    if torch_module is not None and torch_module.cuda.is_available():
        allocated = int(torch_module.cuda.max_memory_allocated())
        reserved = int(torch_module.cuda.max_memory_reserved())
    return {
        "status": "oom" if is_oom_exception(error) else "failed",
        "stage": stage,
        "budget": budget,
        "predict_batch_size": predict_batch_size,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback.format_exc(limit=12),
        "torch_peak_allocated_bytes": allocated,
        "torch_peak_reserved_bytes": reserved,
        "last_watchdog_gpu_mib": None,
        "last_watchdog_ram_mib": None,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Worker preflight; smoke uses synthetic data and never imports torch."""
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    if args.smoke:
        contract = prepare_row_contract(synthetic_frame(), args)
        return {
            "status": "ready",
            "smoke": True,
            "checks": {"synthetic_contract": {"passed": True}},
            "environment": {"python": sys.version, "platform": platform.platform()},
            "tabpfn": {"status": "fake_model"},
            "split": contract["split"]["metadata"],
            "row_contract": {
                "budget_records": contract["budget_records"],
                "score_records": contract["score_records"],
            },
        }

    os.environ["TABPFN_NO_BROWSER"] = "1"
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    model_path = args.model_path.resolve()
    check(
        "feature_count_exactly_17",
        len(BASELINE_FEATURE_COLS) == 17,
        len(BASELINE_FEATURE_COLS),
    )
    check(
        "local_checkpoint",
        model_path.is_file() and os.access(model_path, os.R_OK),
        str(model_path),
    )
    try:
        api = tabpfn_api_evidence(model_path)
        check("tabpfn_context_api", api["status"] == "verified", api)
    except Exception as error:
        api = {
            "status": "blocked_unverified_context",
            "error": f"{type(error).__name__}: {error}",
        }
        check("tabpfn_context_api", False, api)
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        check(
            "cuda_available",
            cuda,
            {
                "torch_version": str(torch.__version__),
                "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
                "total_vram_mib": float(
                    torch.cuda.get_device_properties(0).total_memory / 1024**2
                )
                if cuda
                else None,
            },
        )
    except Exception as error:
        cuda = False
        check("cuda_available", False, f"{type(error).__name__}: {error}")
    gpu = query_gpu_memory(os.getpid())
    check("nvidia_smi", bool(gpu["available"]), gpu)
    check(
        "output_directory_writable",
        os.access(args.out.parent, os.W_OK),
        str(args.out.parent),
    )
    # Full-data checks intentionally occur only when the operator invokes preflight.
    try:
        frame = load_experiment_frame(smoke=False)
        contract = prepare_row_contract(frame, args)
        matrices = make_matrices(frame, contract, min(args.budgets))
        check(
            "matrix_float32",
            all(
                matrices[name].dtype == np.float32
                for name in ("x_fit", "x_validation", "x_test")
            ),
            {
                name: str(matrices[name].dtype)
                for name in ("x_fit", "x_validation", "x_test")
            },
        )
        check("balanced_context_available", True, contract["budget_records"])
        split = contract["split"]["metadata"]
        row_contract = {
            "budget_records": contract["budget_records"],
            "score_records": contract["score_records"],
        }
    except Exception as error:
        split, row_contract = {}, {}
        check("balanced_context_available", False, f"{type(error).__name__}: {error}")
    import psutil

    memory = psutil.virtual_memory()
    try:
        limits = resolve_limits(
            gpu_total_mib=gpu.get("total_mib"),
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
            timeout_seconds=args.budget_timeout_minutes * 60,
        )
        margin_ok = (
            limits.gpu_hard_mib is None
            or gpu.get("used_mib") is None
            or limits.gpu_hard_mib > gpu["used_mib"] + 128
        )
        check(
            "resource_thresholds",
            margin_ok,
            {"limits": limits.__dict__, "current_gpu": gpu},
        )
    except Exception as error:
        check("resource_thresholds", False, f"{type(error).__name__}: {error}")
    status = "ready" if all(item["passed"] for item in checks.values()) else "blocked"
    if api.get("status") == "blocked_unverified_context":
        status = "blocked_unverified_context"
    return {
        "status": status,
        "smoke": False,
        "checks": checks,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_available": cuda,
        },
        "tabpfn": api,
        "split": split,
        "row_contract": row_contract,
    }


def run_budget_worker(args: argparse.Namespace) -> int:
    if args.worker_result is None:
        raise ValueError("worker requires --worker-result")
    os.environ["TABPFN_NO_BROWSER"] = "1"
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    budget = int(args.worker_budget)
    stage = "data_loading"
    torch_module = None
    frame = contract = matrices = model = val_scores = test_scores = None
    try:
        frame = load_experiment_frame(smoke=args.smoke)
        contract = prepare_row_contract(frame, args)
        matrices = make_matrices(frame, contract, budget)
        del frame
        gc.collect()
        if args.smoke:
            model = FakeTabPFNClassifier()
            constructor_kwargs = {
                "fake_model": True,
                "n_estimators": 1,
                "inference_config": {"SUBSAMPLE_SAMPLES": None},
            }
            api = {"status": "fake_model"}
        else:
            import torch

            torch_module = torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
            torch.cuda.reset_peak_memory_stats()
            api = tabpfn_api_evidence(args.model_path.resolve())
            if api["status"] != "verified":
                atomic_write_json(
                    args.worker_result,
                    {
                        "status": "blocked_unverified_context",
                        "budget": budget,
                        "tabpfn": api,
                    },
                )
                return 0
            model, constructor_kwargs = tabpfn_constructor(
                args.model_path.resolve(), args.seed
            )

        started = time.perf_counter()
        with Heartbeat(args.heartbeat, budget, torch_module) as heartbeat:
            stage = "fit"
            heartbeat.update(stage)
            fit_started = time.perf_counter()
            model.fit(matrices["x_fit"], matrices["y_fit"])
            fit_seconds = time.perf_counter() - fit_started
            context = verify_fitted_context(model, budget)
            if context["status"] != "verified":
                atomic_write_json(
                    args.worker_result,
                    {
                        "status": "blocked_unverified_context",
                        "budget": budget,
                        "context_contract": context,
                        "tabpfn": api,
                    },
                )
                return 0

            stage = "validation_predict"
            prediction_started = time.perf_counter()
            val_scores, val_batch = batched_predict(
                model,
                matrices["x_validation"],
                initial_batch_size=args.predict_batch_size,
                minimum_batch_size=args.min_predict_batch_size,
                stop_requested=lambda: bool(
                    args.stop_request and args.stop_request.exists()
                ),
                heartbeat=heartbeat,
                stage=stage,
            )
            val_seconds = time.perf_counter() - prediction_started
            stage = "test_predict"
            prediction_started = time.perf_counter()
            test_scores, test_batch = batched_predict(
                model,
                matrices["x_test"],
                initial_batch_size=min(args.predict_batch_size, val_batch),
                minimum_batch_size=args.min_predict_batch_size,
                stop_requested=lambda: bool(
                    args.stop_request and args.stop_request.exists()
                ),
                heartbeat=heartbeat,
                stage=stage,
            )
            test_seconds = time.perf_counter() - prediction_started
            threshold = fixed_recall_threshold(matrices["y_validation"], val_scores)
            total_seconds = time.perf_counter() - started
            stage = "persist_predictions"
            heartbeat.update(stage, len(matrices["x_test"]))
            predictions_path = prediction_artifact_path(args, budget)
            atomic_write_npz(
                predictions_path,
                budget=np.asarray([budget], dtype="int64"),
                validation_y=matrices["y_validation"],
                validation_score=val_scores.astype("float32", copy=False),
                validation_row_index=matrices["validation_row_index"],
                validation_building_id=matrices["validation_building_id"],
                validation_site_id=matrices["validation_site_id"],
                test_y=matrices["y_test"],
                test_score=test_scores.astype("float32", copy=False),
                test_row_index=matrices["test_row_index"],
                test_building_id=matrices["test_building_id"],
                test_site_id=matrices["test_site_id"],
            )
            result = {
                "status": "completed",
                "budget": budget,
                "fit_completed": True,
                "validation_prediction_completed": len(val_scores) > 0,
                "test_prediction_completed": len(test_scores) > 0,
                "fit_seconds": float(fit_seconds),
                "validation_prediction_seconds": float(val_seconds),
                "test_prediction_seconds": float(test_seconds),
                "total_seconds": float(total_seconds),
                "rows_per_second": float(budget / total_seconds),
                "effective_prediction_batch_size": int(min(val_batch, test_batch)),
                "validation": evaluation_metrics(matrices["y_validation"], val_scores),
                "test": {
                    **evaluation_metrics(matrices["y_test"], test_scores),
                    **operating_metrics(matrices["y_test"], test_scores, threshold),
                },
                "context_contract": context,
                "row_contract": contract["budget_records"][str(budget)],
                "score_rows": contract["score_records"],
                "constructor_kwargs": json_safe(constructor_kwargs),
                "tabpfn": api,
                "prediction_artifact": {
                    "path": str(predictions_path.resolve()),
                    "sha256": sha256_file(predictions_path),
                    "size_bytes": predictions_path.stat().st_size,
                    "rows_saved": {
                        "validation": int(len(val_scores)),
                        "test": int(len(test_scores)),
                    },
                    "contains_training_matrix": False,
                    "curve_inputs_complete": True,
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
            }
            heartbeat.update("completed", len(matrices["x_test"]))
        atomic_write_json(args.worker_result, result)
        return 0
    except BaseException as error:
        result = serialize_failure(
            error,
            budget=budget,
            stage=stage,
            predict_batch_size=args.predict_batch_size,
            torch_module=torch_module,
        )
        atomic_write_json(args.worker_result, result)
        return OOM_EXIT_CODE if result["status"] == "oom" else 1
    finally:
        # Drop every potentially large reference before CUDA cleanup. Process
        # exit remains the definitive VRAM/RAM release boundary.
        model = matrices = contract = val_scores = test_scores = None
        gc.collect()
        if torch_module is not None:
            try:
                torch_module.cuda.empty_cache()
                torch_module.cuda.ipc_collect()
            except RuntimeError:
                pass


def headline_500k_success(summary: dict[str, Any]) -> bool:
    result = summary.get("budget_results", {}).get("500000", {})
    context = result.get("context_contract", {})
    metrics = all(
        key in result.get(section, {})
        for section in ("validation", "test")
        for key in ("roc_auc", "pr_auc")
    )
    return bool(
        result.get("status") == "completed"
        and result.get("row_contract", {}).get("count") == 500_000
        and result.get("row_contract", {}).get("unique_count") == 500_000
        and context.get("requested_context_rows") == 500_000
        and context.get("effective_context_rows") == 500_000
        and context.get("external_sharding") is False
        and context.get("sample_subsampling_disabled") is True
        and context.get("effective_estimators") == 1
        and result.get("fit_completed") is True
        and result.get("validation_prediction_completed") is True
        and result.get("test_prediction_completed") is True
        and result.get("prediction_artifact", {}).get("curve_inputs_complete") is True
        and metrics
    )


def initial_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "external_sharding": False,
        "feature_set": "raw_baseline_17",
        "requested_budgets": args.budgets,
        "split": {},
        "environment": {},
        "tabpfn": {},
        "resource_limits": {},
        "row_contract": {},
        "budget_results": {},
        "last_safe_budget": None,
        "headline_500k_success": False,
    }


def initial_state(args: argparse.Namespace) -> dict[str, Any]:
    now = time.time()
    return {
        "status": "pending",
        "pending_budgets": list(args.budgets),
        "running_budget": None,
        "completed_budgets": [],
        "failed_budgets": [],
        "last_safe_budget": None,
        "worker_pid": None,
        "started_at": now,
        "updated_at": now,
        "stop_reason": None,
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def mark_stale_worker(state: dict[str, Any], summary: dict[str, Any]) -> None:
    budget = state.get("running_budget")
    worker_pid = state.get("worker_pid")
    if budget is None or (worker_pid and pid_exists(int(worker_pid))):
        return
    summary.setdefault("budget_results", {})[str(budget)] = {
        "status": "interrupted_previous_run",
        "budget": int(budget),
    }
    state["failed_budgets"] = sorted({*state.get("failed_budgets", []), int(budget)})
    state.update(
        {
            "running_budget": None,
            "worker_pid": None,
            "stop_reason": "interrupted_previous_run",
        }
    )


def budgets_to_run(args: argparse.Namespace, state: dict[str, Any]) -> list[int]:
    completed = set(int(value) for value in state.get("completed_budgets", []))
    budgets = list(args.budgets)
    if args.restart_budget is not None:
        budgets = budgets[budgets.index(args.restart_budget) :]
        completed -= set(budgets)
        state["completed_budgets"] = sorted(completed)
    return [budget for budget in budgets if budget not in completed]


def public_worker_arguments(args: argparse.Namespace) -> list[str]:
    values = [
        "--budgets",
        *[str(value) for value in args.budgets],
        "--score-rows",
        str(args.score_rows),
        "--predict-batch-size",
        str(args.predict_batch_size),
        "--min-predict-batch-size",
        str(args.min_predict_batch_size),
        "--seed",
        str(args.seed),
        "--model-path",
        str(args.model_path),
        "--out",
        str(args.out),
        "--state-out",
        str(args.state_out),
        "--events-out",
        str(args.events_out),
        "--predictions-dir",
        str(args.predictions_dir),
        "--poll-seconds",
        str(args.poll_seconds),
        "--gpu-soft-limit-fraction",
        str(args.gpu_soft_limit_fraction),
        "--gpu-hard-limit-fraction",
        str(args.gpu_hard_limit_fraction),
        "--ram-soft-limit-fraction",
        str(args.ram_soft_limit_fraction),
        "--ram-hard-limit-fraction",
        str(args.ram_hard_limit_fraction),
        "--soft-limit-consecutive-polls",
        str(args.soft_limit_consecutive_polls),
        "--termination-grace-seconds",
        str(args.termination_grace_seconds),
        "--budget-timeout-minutes",
        str(args.budget_timeout_minutes),
    ]
    for name in (
        "gpu_soft_limit_mib",
        "gpu_hard_limit_mib",
        "ram_soft_limit_mib",
        "ram_hard_limit_mib",
    ):
        if (value := getattr(args, name)) is not None:
            values.extend(["--" + name.replace("_", "-"), str(value)])
    if args.smoke:
        values.append("--smoke")
    return values


def run_preflight_subprocess(args: argparse.Namespace) -> dict[str, Any]:
    path = args.out.with_name(args.out.stem + ".preflight.worker.json")
    path.unlink(missing_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            *public_worker_arguments(args),
            "--worker",
            "--preflight-only",
            "--worker-result",
            str(path),
        ],
        check=False,
    )
    if completed.returncode != 0 or not path.is_file():
        return {
            "status": "blocked",
            "error": f"preflight worker exited {completed.returncode}",
        }
    return load_json(path)


def worker_paths(args: argparse.Namespace, budget: int) -> dict[str, Path]:
    prefix = args.out.with_name(f"{args.out.stem}.budget-{budget}")
    return {
        "result": prefix.with_suffix(".worker.json"),
        "heartbeat": prefix.with_suffix(".heartbeat.json"),
        "stop": prefix.with_suffix(".stop.json"),
    }


def monitor_budget(
    args: argparse.Namespace,
    budget: int,
    state: dict[str, Any],
    *,
    worker_launcher: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    resource_sampler: Callable[[int], Any] = sample_resources,
    process_terminator: Callable[..., dict[str, Any]] = terminate_process_tree,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    paths = worker_paths(args, budget)
    for path in paths.values():
        path.unlink(missing_ok=True)
    worker = worker_launcher(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            *public_worker_arguments(args),
            "--worker",
            "--worker-budget",
            str(budget),
            "--worker-result",
            str(paths["result"]),
            "--heartbeat",
            str(paths["heartbeat"]),
            "--stop-request",
            str(paths["stop"]),
        ]
    )
    started = clock()
    state.update(
        {
            "status": "running",
            "running_budget": budget,
            "worker_pid": worker.pid,
            "updated_at": time.time(),
        }
    )
    atomic_write_json(args.state_out, state)
    append_jsonl(
        args.events_out,
        {
            "event": "worker_started",
            "timestamp": time.time(),
            "budget": budget,
            "pid": worker.pid,
        },
    )
    import psutil

    memory = psutil.virtual_memory()
    initial_gpu = query_gpu_memory(worker.pid)
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
        timeout_seconds=args.budget_timeout_minutes * 60,
    )
    tracker = LimitTracker(limits)
    peak_gpu = peak_rss = peak_system = 0.0
    last_sample = None
    last_stage = None
    heartbeat_seen = False
    soft_requested_at = None
    termination_reason = None
    while worker.poll() is None:
        try:
            sample = resource_sampler(worker.pid)
            last_sample = sample_dict(sample)
            peak_gpu = max(peak_gpu, sample.gpu_used_mib or 0)
            peak_rss = max(peak_rss, sample.worker_rss_mib)
            peak_system = max(peak_system, sample.system_used_mib)
            decision = tracker.observe(sample, elapsed_seconds=clock() - started)
        except Exception:
            decision = type("Decision", (), {"action": "continue", "reason": None})()
        if paths["heartbeat"].is_file():
            try:
                heartbeat = load_json(paths["heartbeat"])
                heartbeat_seen = True
                if heartbeat.get("stage") != last_stage:
                    last_stage = heartbeat.get("stage")
                    append_jsonl(
                        args.events_out,
                        {
                            "event": "stage_changed",
                            "timestamp": time.time(),
                            "budget": budget,
                            "stage": last_stage,
                        },
                    )
                if time.time() - paths["heartbeat"].stat().st_mtime >= max(
                    5.0, args.termination_grace_seconds
                ):
                    termination_reason = "worker no longer responsive"
            except (OSError, json.JSONDecodeError):
                pass
        if decision.action == "request_stop" and soft_requested_at is None:
            soft_requested_at = clock()
            atomic_write_json(
                paths["stop"], {"reason": decision.reason, "timestamp": time.time()}
            )
            append_jsonl(
                args.events_out,
                {
                    "event": "soft_limit_exceeded",
                    "timestamp": time.time(),
                    "budget": budget,
                    "reason": decision.reason,
                },
            )
            append_jsonl(
                args.events_out,
                {
                    "event": "stop_requested",
                    "timestamp": time.time(),
                    "budget": budget,
                    "reason": decision.reason,
                },
            )
        if decision.action == "terminate":
            termination_reason = decision.reason
            if decision.reason and "hard limit" in decision.reason:
                append_jsonl(
                    args.events_out,
                    {
                        "event": "hard_limit_exceeded",
                        "timestamp": time.time(),
                        "budget": budget,
                        "reason": decision.reason,
                    },
                )
        elif (
            soft_requested_at is not None
            and clock() - soft_requested_at >= args.termination_grace_seconds
        ):
            termination_reason = "soft-limit grace period expired"
        if termination_reason:
            termination = process_terminator(
                worker.pid, grace_seconds=args.termination_grace_seconds
            )
            append_jsonl(
                args.events_out,
                {
                    "event": "worker_terminated",
                    "timestamp": time.time(),
                    "budget": budget,
                    "reason": termination_reason,
                    "details": termination,
                },
            )
            if termination.get("killed"):
                append_jsonl(
                    args.events_out,
                    {
                        "event": "worker_killed",
                        "timestamp": time.time(),
                        "budget": budget,
                        "reason": termination_reason,
                        "pids": termination["killed"],
                    },
                )
            break
        time.sleep(args.poll_seconds)
    try:
        exit_code = worker.wait(timeout=max(1.0, args.termination_grace_seconds * 2))
    except subprocess.TimeoutExpired:
        termination_reason = termination_reason or "worker no longer responsive"
        process_terminator(worker.pid, grace_seconds=args.termination_grace_seconds)
        exit_code = worker.poll()
    result = (
        load_json(paths["result"])
        if paths["result"].is_file()
        else {
            "status": "terminated" if termination_reason else "crashed",
            "budget": budget,
            "stage": last_stage,
            "stop_reason": termination_reason,
        }
    )
    result.update(
        {
            "worker_exit_code": exit_code,
            "watchdog_peak_gpu_mib": peak_gpu or None,
            "watchdog_peak_worker_rss_mib": peak_rss or None,
            "watchdog_peak_system_ram_mib": peak_system or None,
            "last_watchdog_sample": last_sample,
            "monitoring_scope": (
                last_sample.get("monitoring_scope")
                if last_sample
                else initial_gpu.get("monitoring_scope")
            ),
            "heartbeat_seen": heartbeat_seen,
        }
    )
    if result.get("status") == "oom":
        append_jsonl(
            args.events_out,
            {
                "event": "oom",
                "timestamp": time.time(),
                "budget": budget,
                "stage": result.get("stage"),
                "reason": result.get("error", result.get("stop_reason")),
            },
        )
    return result


def controller(args: argparse.Namespace) -> int:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.force = True
    if args.force:
        state, summary = initial_state(args), initial_summary(args)
        args.events_out.unlink(missing_ok=True)
    elif args.resume and args.state_out.is_file():
        state = load_json(args.state_out)
        summary = load_json(args.out) if args.out.is_file() else initial_summary(args)
        mark_stale_worker(state, summary)
        if state.get("running_budget") is not None and state.get("worker_pid"):
            state["status"] = "worker_still_running"
            atomic_write_json(args.state_out, state)
            return 0
    elif args.state_out.exists() or args.out.exists():
        raise FileExistsError("existing experiment state; pass --resume or --force")
    else:
        state, summary = initial_state(args), initial_summary(args)
    atomic_write_json(args.state_out, state)
    preflight = run_preflight_subprocess(args)
    summary.update(
        {
            "split": preflight.get("split", {}),
            "environment": preflight.get("environment", {}),
            "tabpfn": preflight.get("tabpfn", {}),
            "row_contract": preflight.get("row_contract", {}),
            "resource_limits": preflight.get("checks", {})
            .get("resource_thresholds", {})
            .get("detail", {}),
            "preflight": preflight,
        }
    )
    atomic_write_json(args.out, summary)
    if args.preflight_only:
        state.update(
            {
                "status": preflight.get("status"),
                "updated_at": time.time(),
                "stop_reason": None
                if preflight.get("status") == "ready"
                else preflight.get("status"),
            }
        )
        atomic_write_json(args.state_out, state)
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    if preflight.get("status") != "ready":
        state.update(
            {
                "status": "blocked",
                "updated_at": time.time(),
                "stop_reason": preflight.get("status"),
            }
        )
        atomic_write_json(args.state_out, state)
        return 0
    if args.restart_budget is not None:
        for budget in args.budgets[args.budgets.index(args.restart_budget) :]:
            summary["budget_results"].pop(str(budget), None)
    for budget in budgets_to_run(args, state):
        result = monitor_budget(args, budget, state)
        summary["budget_results"][str(budget)] = result
        if result.get("status") == "completed":
            state["completed_budgets"] = sorted(
                {*state.get("completed_budgets", []), budget}
            )
            state["last_safe_budget"] = summary["last_safe_budget"] = budget
            append_jsonl(
                args.events_out,
                {
                    "event": "budget_completed",
                    "timestamp": time.time(),
                    "budget": budget,
                },
            )
        else:
            state["failed_budgets"] = sorted({*state.get("failed_budgets", []), budget})
            state["stop_reason"] = result.get("status")
        state.update(
            {
                "running_budget": None,
                "worker_pid": None,
                "pending_budgets": [
                    value
                    for value in args.budgets
                    if value not in state.get("completed_budgets", [])
                ],
                "updated_at": time.time(),
            }
        )
        summary["headline_500k_success"] = headline_500k_success(summary)
        atomic_write_json(args.out, summary)
        atomic_write_json(args.state_out, state)
        if result.get("status") != "completed" and not args.continue_after_failure:
            break
    state["status"] = (
        "completed" if not state["pending_budgets"] else "stopped_after_failure"
    )
    state["updated_at"] = time.time()
    summary["headline_500k_success"] = headline_500k_success(summary)
    atomic_write_json(args.out, summary)
    atomic_write_json(args.state_out, state)
    append_jsonl(
        args.events_out,
        {
            "event": "run_completed",
            "timestamp": time.time(),
            "status": state["status"],
            "last_safe_budget": state["last_safe_budget"],
        },
    )
    print(
        f"Saved {args.out}; status={state['status']} "
        f"last_safe_budget={state['last_safe_budget']}"
    )
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
    if args.worker:
        if args.preflight_only:
            if args.worker_result is None:
                raise ValueError("preflight worker requires --worker-result")
            try:
                atomic_write_json(args.worker_result, run_preflight(args))
                return 0
            except BaseException as error:
                atomic_write_json(
                    args.worker_result,
                    {
                        "status": "blocked",
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(limit=12),
                    },
                )
                return 1
        return run_budget_worker(args)
    return controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
