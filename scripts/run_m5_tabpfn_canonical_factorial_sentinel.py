"""Establish a reproducible TabPFN execution protocol for one Path-A sentinel.

The legacy low-memory audit is intentionally not reused.  This script creates a
new, immutable run root for one predeclared F4 factorial cell and implements
R1--R7: three live predictions, state save, same-process reload, and two
independent fresh-process GPU reloads.  It never writes to the legacy factorial
or recovery roots, never scores the independent query, and never fits a second
factorial cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lead import ROOT, array_sha256, load_m3_frame, validate_context_manifest
from lead.m5_context import feature_names, query_paths

try:  # Supports both ``python scripts/...`` and package-based tests.
    from scripts.analyze_m5_hotwater_label_role_factorial import metrics, query_frame
    from scripts.run_m5_story_ae_probe import (
        build_feature_matrix,
        validate_feature_matrix,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct invocation.
    from analyze_m5_hotwater_label_role_factorial import metrics, query_frame
    from run_m5_story_ae_probe import build_feature_matrix, validate_feature_matrix


FACTORIAL = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"
CANONICAL = FACTORIAL / "canonical_tabpfn_protocol"
CHECKPOINT = ROOT / ".tabpfn-cache" / "tabpfn-v3-classifier-v3_default.ckpt"
RUN_ID = "v1_tabpfn_8_0_8_fit_preprocessors_seed42_pooled_cell_specific"
SENTINEL = "hw_pos_present__hw_neg_present"
SEED = 42
MODEL_SEED = 42
QUERY_BATCH_SIZE = 352
PREDICTION_STAGES = (
    "R1_fit_gpu",
    "R2_live_gpu_repeat",
    "R3_live_gpu_repeat",
    "R5_same_process_gpu_reload",
    "R6_fresh_gpu_reload_1",
    "R7_fresh_gpu_reload_2",
)
PRIMARY = ("hw01_within_rank_gap", "hw01_pair_auc", "steam_pos_vs_hw_neg_auc")
TOLERANCES = {
    "probability_mae_max": 1e-4,
    "probability_max_abs_max": 0.005,
    "spearman_min": 0.99999,
    "primary_estimand_abs_delta_max": 0.002,
    "require_primary_direction_match": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--stage", choices=("full", "fresh"), default="full")
    parser.add_argument(
        "--fresh-slot", choices=("R6_fresh_gpu_reload_1", "R7_fresh_gpu_reload_2")
    )
    parser.add_argument("--expected-tabpfn-version")
    parser.add_argument(
        "--reuse-assembly-from",
        type=Path,
        help="immutable canonical run whose frozen arrays/scaler must be reused",
    )
    args = parser.parse_args()
    if args.stage == "fresh" and args.fresh_slot is None:
        parser.error("--fresh-slot is required for --stage fresh")
    return args


def run_root(run_id: str) -> Path:
    return CANONICAL / run_id


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def finite_profile(values: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(values)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "nan": int(np.isnan(contiguous).sum()),
        "inf": int(np.isinf(contiguous).sum()),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [sanitize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def backend_config(torch: Any) -> dict[str, Any]:
    """Set and report the fixed CUDA execution settings before any model call."""
    if not torch.cuda.is_available():
        raise RuntimeError("canonical sentinel requires CUDA")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    cuda_backend = torch.backends.cuda
    if hasattr(cuda_backend, "enable_flash_sdp"):
        cuda_backend.enable_flash_sdp(False)
        cuda_backend.enable_mem_efficient_sdp(False)
        cuda_backend.enable_math_sdp(True)
    return {
        "device": "cuda:0",
        "gpu_count": int(torch.cuda.device_count()),
        "gpu": torch.cuda.get_device_name(0),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "flash_sdp": bool(cuda_backend.flash_sdp_enabled())
        if hasattr(cuda_backend, "flash_sdp_enabled")
        else None,
        "mem_efficient_sdp": bool(cuda_backend.mem_efficient_sdp_enabled())
        if hasattr(cuda_backend, "mem_efficient_sdp_enabled")
        else None,
        "math_sdp": bool(cuda_backend.math_sdp_enabled())
        if hasattr(cuda_backend, "math_sdp_enabled")
        else None,
    }


def model_kwargs(torch: Any) -> dict[str, Any]:
    return {
        "n_estimators": 8,
        "auto_scale_n_estimators": False,
        "model_path": str(CHECKPOINT),
        "device": "cuda:0",
        "random_state": MODEL_SEED,
        "fit_mode": "fit_preprocessors",
        "memory_saving_mode": False,
        "keep_cache_on_device": True,
        "ignore_pretraining_limits": True,
        "n_preprocessing_jobs": 1,
        "inference_config": {"SUBSAMPLE_SAMPLES": None},
        "inference_precision": torch.float32,
        "show_progress_bar": False,
    }


def initialize(torch: Any) -> Any:
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(**model_kwargs(torch))


def stage_paths(root: Path, stage: str) -> tuple[Path, Path]:
    return root / "stages" / f"{stage}.npz", root / "stages" / f"{stage}.json"


def require_new_stage(root: Path, stage: str) -> None:
    artifact, metadata = stage_paths(root, stage)
    if artifact.exists() or metadata.exists():
        raise FileExistsError(
            f"refusing to overwrite completed or partial stage: {stage}"
        )


def load_query(root: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(root / "frozen_arrays.npz") as payload:
        query = np.asarray(payload["scaled_query"], dtype="float32")
        raw_index = np.asarray(payload["query_raw_index"], dtype="int64")
    if len(query) != QUERY_BATCH_SIZE:
        raise AssertionError(
            f"fixed query batch drifted: {len(query)} != {QUERY_BATCH_SIZE}"
        )
    return query, raw_index


def stage_environment(
    model: Any, backend: dict[str, Any], state_sha: str | None
) -> dict[str, Any]:
    import tabpfn
    import torch

    return {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "command": sys.argv,
        "timestamp_epoch": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "tabpfn": tabpfn.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "backend": backend,
        "constructor": sanitize(model.get_params(deep=False)),
        "actual": {
            "n_estimators": int(getattr(model, "n_estimators_", -1)),
            "random_state": sanitize(getattr(model, "random_state", None)),
            "fit_mode": sanitize(getattr(model, "fit_mode", None)),
            "inference_config": sanitize(getattr(model, "inference_config_", None)),
            "query_batch_size": QUERY_BATCH_SIZE,
        },
        "state_sha256": state_sha,
    }


def capture(
    root: Path, model: Any, stage: str, state_sha: str | None
) -> dict[str, Any]:
    """Predict exactly one frozen batch and atomically persist an immutable stage."""
    import torch

    require_new_stage(root, stage)
    query, raw_index = load_query(root)
    backend = backend_config(torch)
    started = time.time()
    aggregate = np.asarray(model.predict_proba(query), dtype="float32")
    raw_logits = model._raw_predict(query, return_logits=False, return_raw_logits=True)
    raw_logits = raw_logits.detach().cpu().numpy().astype("float32")
    temperature = float(
        getattr(model, "softmax_temperature_", model.softmax_temperature)
    )
    per_estimator = (
        torch.softmax(torch.from_numpy(raw_logits) / temperature, dim=-1)
        .numpy()
        .astype("float32")
    )
    if (
        aggregate.shape != (QUERY_BATCH_SIZE, 2)
        or raw_logits.shape[1:] != aggregate.shape
    ):
        raise AssertionError("unexpected aggregate/raw-logit shape")
    if not np.isfinite(aggregate).all() or not np.isfinite(raw_logits).all():
        raise AssertionError("non-finite canonical stage prediction")
    artifact, metadata = stage_paths(root, stage)
    atomic_npz(
        artifact,
        raw_index=raw_index,
        aggregate_probability=aggregate,
        positive_score=aggregate[:, 1],
        raw_logits=raw_logits,
        per_estimator_probability=per_estimator,
    )
    primary = {
        key: float(value)
        for key, value in metrics(aggregate[:, 1], query_frame()).items()
        if key in PRIMARY
    }
    contract_path = root / "state_contract.json"
    contract = json.loads(
        (
            contract_path
            if contract_path.is_file()
            else root / "assembly_contract.json"
        ).read_text(encoding="utf-8")
    )
    payload = {
        "stage": stage,
        "status": "completed",
        "started_epoch": started,
        "completed_epoch": time.time(),
        "artifact_sha256": file_sha256(artifact),
        "aggregate": finite_profile(aggregate),
        "raw_logits": finite_profile(raw_logits),
        "per_estimator_probability": finite_profile(per_estimator),
        "primary_estimands": primary,
        "frozen_contract": {
            "scaled_query_sha256": contract["scaled_query"]["sha256"],
            "scaler_sha256": contract["scaler_sha256"],
            "checkpoint_sha256": contract["checkpoint"]["sha256"],
            "state_sha256": state_sha,
        },
        **stage_environment(model, backend, state_sha),
    }
    atomic_json(metadata, payload)
    return payload


def load_manifest_and_arrays(
    root: Path,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    manifest_path = FACTORIAL / "manifests" / f"seed{SEED}" / f"{SENTINEL}.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = load_m3_frame(verbose=True)
    validate_context_manifest(frame, payload)
    fit = frame.loc[frame["building_id"] % 2 == 0]
    holdout = frame.loc[frame["building_id"] % 2 == 1]
    context_raw = np.asarray(payload["raw_index"], dtype="int64")
    _, query_path = query_paths(
        ROOT / "data" / "processed" / "m5_context_stories", "screening"
    )
    with np.load(query_path) as query_payload:
        query_raw = np.asarray(query_payload["raw_index"], dtype="int64")
        query_y = np.asarray(query_payload["anomaly"], dtype="int8")
    raw_train = build_feature_matrix(fit, context_raw, "F4", full_frame=fit)
    raw_query = build_feature_matrix(holdout, query_raw, "F4", full_frame=holdout)
    validate_feature_matrix(raw_train, matrix_name="canonical sentinel train")
    validate_feature_matrix(raw_query, matrix_name="canonical sentinel query")
    labels = frame.iloc[context_raw]["anomaly"].to_numpy(dtype="int8")
    if raw_train.shape != (20_000, 137) or raw_query.shape != (QUERY_BATCH_SIZE, 137):
        raise AssertionError(
            f"unexpected F4 array shapes: {raw_train.shape}, {raw_query.shape}"
        )
    if int(labels.sum()) != 10_000:
        raise AssertionError("canonical sentinel lost 50/50 labels")
    return payload, context_raw, query_raw, query_y, raw_train, raw_query, labels


def create_contract(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import tabpfn
    import torch

    payload, context_raw, query_raw, query_y, raw_train, raw_query, labels = (
        load_manifest_and_arrays(root)
    )
    scaler = StandardScaler().fit(raw_train)
    scaled_train = scaler.transform(raw_train).astype("float32")
    scaled_query = scaler.transform(raw_query).astype("float32")
    root.mkdir(parents=True, exist_ok=False)
    joblib.dump(scaler, root / "scaler.joblib")
    atomic_npz(
        root / "frozen_arrays.npz",
        context_raw_index=context_raw,
        query_raw_index=query_raw,
        anomaly=query_y,
        labels=labels,
        raw_train=raw_train,
        scaled_train=scaled_train,
        raw_query=raw_query,
        scaled_query=scaled_query,
    )
    backend = backend_config(torch)
    contract = {
        "run_id": root.name,
        "protocol": "fit_preprocessors_single_gpu_float32_fixed_batch",
        "sentinel": {
            "context_seed": SEED,
            "factorial_cell": SENTINEL,
            "scaler_arm": "cell_specific",
            "model_seed": MODEL_SEED,
            "features": "F4",
            "n_context": 20_000,
        },
        "manifest_path": str(
            (FACTORIAL / "manifests" / f"seed{SEED}" / f"{SENTINEL}.json").relative_to(
                ROOT
            )
        ),
        "manifest_sha256": file_sha256(
            FACTORIAL / "manifests" / f"seed{SEED}" / f"{SENTINEL}.json"
        ),
        "manifest_raw_index_sha256": payload["raw_index_sha256"],
        "context_raw_index_sha256": array_sha256(context_raw),
        "query_raw_index_sha256": array_sha256(query_raw),
        "feature_names": feature_names("F4"),
        "feature_order_sha256": hashlib.sha256(
            "\n".join(feature_names("F4")).encode()
        ).hexdigest(),
        "raw_train": finite_profile(raw_train),
        "scaled_train": finite_profile(scaled_train),
        "raw_query": finite_profile(raw_query),
        "scaled_query": finite_profile(scaled_query),
        "scaler_sha256": file_sha256(root / "scaler.joblib"),
        "checkpoint": {
            "path": str(CHECKPOINT.relative_to(ROOT)),
            "sha256": file_sha256(CHECKPOINT),
        },
        "package_versions": {
            "python": sys.version,
            "tabpfn": tabpfn.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "backend": backend,
        "constructor_requested": sanitize(model_kwargs(torch)),
        "query_batch_size": QUERY_BATCH_SIZE,
    }
    atomic_json(root / "assembly_contract.json", contract)
    return scaled_train, labels, scaled_query


def create_reused_contract(
    root: Path, source: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a new version root while reusing byte-identical frozen inputs."""
    import tabpfn
    import torch

    source = source.resolve()
    source_contract_path = source / "assembly_contract.json"
    if not source_contract_path.is_file():
        raise FileNotFoundError(
            f"missing source assembly contract: {source_contract_path}"
        )
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    required = ("frozen_arrays.npz", "scaler.joblib")
    if any(not (source / name).is_file() for name in required):
        raise FileNotFoundError("source canonical assembly is incomplete")
    root.mkdir(parents=True, exist_ok=False)
    for name in required:
        target = root / name
        try:
            os.link(source / name, target)
        except OSError:
            shutil.copy2(source / name, target)
        if file_sha256(target) != file_sha256(source / name):
            raise AssertionError(f"reused artifact digest drifted: {name}")
    with np.load(root / "frozen_arrays.npz") as payload:
        scaled_train = np.asarray(payload["scaled_train"], dtype="float32")
        labels = np.asarray(payload["labels"], dtype="int8")
        scaled_query = np.asarray(payload["scaled_query"], dtype="float32")
    if (
        finite_profile(scaled_train) != source_contract["scaled_train"]
        or finite_profile(scaled_query) != source_contract["scaled_query"]
    ):
        raise AssertionError("reused frozen-array contract mismatch")
    if file_sha256(root / "scaler.joblib") != source_contract["scaler_sha256"]:
        raise AssertionError("reused scaler contract mismatch")
    contract = dict(source_contract)
    contract.update(
        {
            "run_id": root.name,
            "reused_assembly": {
                "source_run": str(source.relative_to(ROOT)),
                "source_assembly_contract_sha256": file_sha256(source_contract_path),
                "frozen_arrays_sha256": file_sha256(root / "frozen_arrays.npz"),
                "scaler_sha256": file_sha256(root / "scaler.joblib"),
            },
            "package_versions": {
                "python": sys.version,
                "tabpfn": tabpfn.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            },
            "backend": backend_config(torch),
            "constructor_requested": sanitize(model_kwargs(torch)),
        }
    )
    atomic_json(root / "assembly_contract.json", contract)
    return scaled_train, labels, scaled_query


def state_contract(root: Path) -> dict[str, Any]:
    base = json.loads((root / "assembly_contract.json").read_text(encoding="utf-8"))
    base["state_sha256"] = file_sha256(root / "model.tabpfn_fit")
    atomic_json(root / "state_contract.json", base)
    return base


def comparison(left: str, right: str, root: Path) -> dict[str, Any]:
    with (
        np.load(stage_paths(root, left)[0]) as first,
        np.load(stage_paths(root, right)[0]) as second,
    ):
        aggregate_left = np.asarray(first["aggregate_probability"], dtype="float64")
        aggregate_right = np.asarray(second["aggregate_probability"], dtype="float64")
        raw_left = np.asarray(first["raw_logits"], dtype="float64")
        raw_right = np.asarray(second["raw_logits"], dtype="float64")
    delta = np.abs(aggregate_left - aggregate_right)
    raw_delta = np.abs(raw_left - raw_right)
    meta_left = json.loads(stage_paths(root, left)[1].read_text(encoding="utf-8"))
    meta_right = json.loads(stage_paths(root, right)[1].read_text(encoding="utf-8"))
    primary_delta = {
        key: abs(
            meta_left["primary_estimands"][key] - meta_right["primary_estimands"][key]
        )
        for key in PRIMARY
    }
    direction_match = {
        key: bool(
            np.sign(meta_left["primary_estimands"][key])
            == np.sign(meta_right["primary_estimands"][key])
        )
        for key in PRIMARY
    }
    spearman = float(
        pd.Series(aggregate_left[:, 1]).corr(
            pd.Series(aggregate_right[:, 1]), method="spearman"
        )
    )
    passes = {
        "mae": float(delta.mean()) <= TOLERANCES["probability_mae_max"],
        "max_abs": float(delta.max()) <= TOLERANCES["probability_max_abs_max"],
        "spearman": spearman >= TOLERANCES["spearman_min"],
        "primary_delta": max(primary_delta.values())
        <= TOLERANCES["primary_estimand_abs_delta_max"],
        "primary_direction": all(direction_match.values()),
    }
    return {
        "left": left,
        "right": right,
        "proba_mae": float(delta.mean()),
        "proba_max_abs": float(delta.max()),
        "proba_spearman": spearman,
        "changed_rows": int(np.count_nonzero(delta.max(axis=1))),
        "raw_logit_mae": float(raw_delta.mean()),
        "raw_logit_max_abs": float(raw_delta.max()),
        "primary_estimand_abs_delta": primary_delta,
        "primary_direction_match": direction_match,
        "passes": passes,
        "passed": bool(all(passes.values())),
    }


def write_gate(root: Path) -> bool:
    comparisons = [
        comparison(PREDICTION_STAGES[i], PREDICTION_STAGES[j], root)
        for i in range(len(PREDICTION_STAGES))
        for j in range(i + 1, len(PREDICTION_STAGES))
    ]
    passed = all(item["passed"] for item in comparisons)
    atomic_json(
        root / "reproducibility_gate.json",
        {
            "passed": passed,
            "tolerances_predeclared": TOLERANCES,
            "stages": list(PREDICTION_STAGES),
            "comparisons": comparisons,
            "first_failed_comparison": next(
                (item for item in comparisons if not item["passed"]), None
            ),
        },
    )
    return passed


def fresh(root: Path, stage: str) -> int:
    import tabpfn
    import torch
    from tabpfn.model_loading import load_fitted_tabpfn_model

    if not (root / "state_contract.json").is_file():
        raise FileNotFoundError("fresh reload requires immutable state contract")
    state = json.loads((root / "state_contract.json").read_text(encoding="utf-8"))
    if tabpfn.__version__ != state["package_versions"]["tabpfn"]:
        raise RuntimeError(
            "fresh process TabPFN version drifted: "
            f"{tabpfn.__version__} != {state['package_versions']['tabpfn']}"
        )
    if file_sha256(root / "model.tabpfn_fit") != state["state_sha256"]:
        raise AssertionError("fitted state digest drifted")
    backend_config(torch)
    model = load_fitted_tabpfn_model(root / "model.tabpfn_fit", device="cuda:0")
    capture(root, model, stage, state["state_sha256"])
    return 0


def full(
    root: Path, expected_tabpfn_version: str | None, reuse_assembly_from: Path | None
) -> int:
    import tabpfn
    import torch
    from tabpfn.model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model

    if root.exists():
        raise FileExistsError(f"new canonical run root already exists: {root}")
    if (
        expected_tabpfn_version is not None
        and tabpfn.__version__ != expected_tabpfn_version
    ):
        raise RuntimeError(
            "requested TabPFN version does not match runtime: "
            f"{expected_tabpfn_version} != {tabpfn.__version__}"
        )
    scaled_train, labels, _ = (
        create_reused_contract(root, reuse_assembly_from)
        if reuse_assembly_from is not None
        else create_contract(root)
    )
    model = initialize(torch)
    model.fit(scaled_train, labels)
    if int(getattr(model, "n_train_samples_", -1)) != len(labels):
        raise AssertionError("TabPFN fit row count drifted")
    capture(root, model, "R1_fit_gpu", None)
    capture(root, model, "R2_live_gpu_repeat", None)
    capture(root, model, "R3_live_gpu_repeat", None)
    save_fitted_tabpfn_model(model, root / "model.tabpfn_fit")
    state = state_contract(root)
    atomic_json(
        root / "R4_state_save.json",
        {
            "stage": "R4_state_save",
            "status": "completed",
            "state_sha256": state["state_sha256"],
            "checkpoint_sha256": state["checkpoint"]["sha256"],
            "timestamp_epoch": time.time(),
        },
    )
    same_process = load_fitted_tabpfn_model(root / "model.tabpfn_fit", device="cuda:0")
    capture(root, same_process, "R5_same_process_gpu_reload", state["state_sha256"])
    for slot in ("R6_fresh_gpu_reload_1", "R7_fresh_gpu_reload_2"):
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--run-id",
                root.name,
                "--stage",
                "fresh",
                "--fresh-slot",
                slot,
            ],
            check=True,
        )
    passed = write_gate(root)
    print(
        f"canonical sentinel reproducibility gate {'PASSED' if passed else 'FAILED'}: {root}",
        flush=True,
    )
    return 0 if passed else 2


def main() -> int:
    args = parse_args()
    root = run_root(args.run_id)
    return (
        fresh(root, args.fresh_slot)
        if args.stage == "fresh"
        else full(root, args.expected_tabpfn_version, args.reuse_assembly_from)
    )


if __name__ == "__main__":
    raise SystemExit(main())
