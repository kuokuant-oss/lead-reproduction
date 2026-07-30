"""Sentinel-only TabPFN fit/save/load determinism audit for M5 Path A.

The script touches one predeclared factorial cell only. It never scores the
192-row independent query and never expands to the remaining 23 TabPFN cells.
All arrays and fitted states remain under ignored ``data/processed``; the small
JSON summary is designed to be copied into a report after review.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from lead import ROOT, array_sha256, load_m3_frame, validate_context_manifest
from lead.m5_context import feature_names, query_paths
from run_m5_story_ae_probe import build_feature_matrix, validate_feature_matrix


FACTORIAL = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"
AUDIT = FACTORIAL / "deterministic_execution_audit"
SENTINEL = "hw_pos_present__hw_neg_present"
SEED = 42
MODEL_SEED = 42
CHECKPOINT = ROOT / ".tabpfn-cache" / "tabpfn-v3-classifier-v3_default.ckpt"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("full", "p5"), default="full")
    return parser.parse_args()


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


def manifest() -> dict[str, Any]:
    path = FACTORIAL / "manifests" / f"seed{SEED}" / f"{SENTINEL}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_context_manifest(load_m3_frame(verbose=True), payload)
    return payload


def initialize(device: str):
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        n_estimators=8,
        auto_scale_n_estimators=False,
        model_path=str(CHECKPOINT),
        device=device,
        random_state=MODEL_SEED,
        fit_mode="low_memory",
        memory_saving_mode=True,
        keep_cache_on_device=False,
        ignore_pretraining_limits=True,
        n_preprocessing_jobs=1,
        inference_config={"SUBSAMPLE_SAMPLES": None},
        show_progress_bar=False,
    )


def finite_profile(values: np.ndarray) -> dict[str, int | str | list[int]]:
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "nan": int(np.isnan(values).sum()),
        "inf": int(np.isinf(values).sum()),
        "sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
    }


def sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def capture(model: Any, query: np.ndarray, stage: str) -> dict[str, Any]:
    """Store public aggregate probabilities plus raw per-estimator softmaxes."""
    import torch

    aggregate = np.asarray(model.predict_proba(query), dtype="float32")
    raw = model._raw_predict(query, return_logits=False, return_raw_logits=True)
    temperature = float(
        getattr(model, "softmax_temperature_", model.softmax_temperature)
    )
    per_estimator = (
        torch.softmax(raw / temperature, dim=-1)
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
    )
    atomic_npz(
        AUDIT / "predictions" / f"{stage}.npz",
        aggregate_probability=aggregate,
        positive_score=aggregate[:, 1],
        per_estimator_probability=per_estimator,
    )
    return {
        "stage": stage,
        "aggregate": finite_profile(aggregate),
        "per_estimator": finite_profile(per_estimator),
        "public_vs_raw_call_separate": True,
        "actual_n_estimators": int(getattr(model, "n_estimators_", -1)),
        "device": str(getattr(model, "device", "unknown")),
    }


def compare(a: str, b: str) -> dict[str, Any]:
    with (
        np.load(AUDIT / "predictions" / f"{a}.npz") as left,
        np.load(AUDIT / "predictions" / f"{b}.npz") as right,
    ):
        x, y = (
            np.asarray(left["aggregate_probability"], dtype="float64"),
            np.asarray(right["aggregate_probability"], dtype="float64"),
        )
    delta = np.abs(x - y)
    return {
        "from": a,
        "to": b,
        "bit_exact": bool(np.array_equal(x, y)),
        "mae": float(delta.mean()),
        "max_abs": float(delta.max()),
        "changed_entries": int(np.count_nonzero(delta)),
    }


def provenance(
    model: Any,
    raw_train: np.ndarray,
    scaled_train: np.ndarray,
    raw_query: np.ndarray,
    scaled_query: np.ndarray,
    query_raw: np.ndarray,
    manifest_payload: dict[str, Any],
) -> dict[str, Any]:
    import tabpfn
    import torch

    return {
        "sentinel": {
            "context_seed": SEED,
            "factorial_cell": SENTINEL,
            "scaler_arm": "cell_specific",
            "model_seed": MODEL_SEED,
        },
        "manifest_raw_index_sha256": manifest_payload["raw_index_sha256"],
        "query_raw_index_sha256": array_sha256(query_raw),
        "feature_order": feature_names("F4"),
        "raw_train": finite_profile(raw_train),
        "scaled_train": finite_profile(scaled_train),
        "raw_query": finite_profile(raw_query),
        "scaled_query": finite_profile(scaled_query),
        "scaler_sha256": file_sha256(AUDIT / "scaler.joblib"),
        "constructor": sanitize(model.get_params(deep=False)),
        "api": {
            "predict_proba": str(inspect.signature(model.predict_proba)),
            "save_load": "tabpfn.model_loading.save_fitted_tabpfn_model / load_fitted_tabpfn_model",
            "call_order": "P1 fit-predict; P2 live-predict; P3 save-live-predict; P4 same-process load-predict; P5 fresh-process load-predict; P6 CPU load-predict-twice; P7 deterministic GPU load-predict-twice",
        },
        "actual": {
            "n_estimators": int(getattr(model, "n_estimators_", -1)),
            "random_state": model.random_state,
            "fit_mode": model.fit_mode,
            "inference_config": sanitize(vars(getattr(model, "inference_config_", {}))),
        },
        "checkpoint": {"name": CHECKPOINT.name, "sha256": file_sha256(CHECKPOINT)},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "tabpfn": tabpfn.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "attention_backend": str(getattr(model, "inference_config_", "unknown")),
            "query_batch_rows": len(scaled_query),
        },
        "original_factorial_environment": "not persisted in first-pass result.json; version/checkpoint equality cannot be established retrospectively",
        "recovery_environment": json.loads(
            (FACTORIAL / "recovery" / "environment_provenance_tabpfn.json").read_text(
                encoding="utf-8"
            )
        ),
    }


def load_inputs() -> tuple[np.ndarray, np.ndarray]:
    with np.load(AUDIT / "inputs.npz") as payload:
        return np.asarray(payload["scaled_query"], dtype="float32"), np.asarray(
            payload["query_raw_index"], dtype="int64"
        )


def p5() -> int:
    from tabpfn.model_loading import load_fitted_tabpfn_model

    query, _ = load_inputs()
    model = load_fitted_tabpfn_model(AUDIT / "model.tabpfn_fit", device="cuda")
    atomic_json(
        AUDIT / "stage_p5.json", capture(model, query, "P5_fresh_process_gpu_load")
    )
    return 0


def main() -> int:
    parsed = args()
    if parsed.stage == "p5":
        return p5()
    import torch
    from tabpfn.model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the sentinel fit")
    payload = manifest()
    frame = load_m3_frame(verbose=True)
    fit, holdout = (
        frame.loc[frame["building_id"] % 2 == 0],
        frame.loc[frame["building_id"] % 2 == 1],
    )
    context_raw = np.asarray(payload["raw_index"], dtype="int64")
    _, query_path = query_paths(
        ROOT / "data" / "processed" / "m5_context_stories", "screening"
    )
    with np.load(query_path) as query_payload:
        query_raw, y_query = (
            np.asarray(query_payload["raw_index"], dtype="int64"),
            np.asarray(query_payload["anomaly"], dtype="int8"),
        )
    raw_train = build_feature_matrix(fit, context_raw, "F4", full_frame=fit)
    raw_query = build_feature_matrix(holdout, query_raw, "F4", full_frame=holdout)
    validate_feature_matrix(raw_train, matrix_name="deterministic sentinel train")
    validate_feature_matrix(raw_query, matrix_name="deterministic sentinel query")
    labels = frame.iloc[context_raw]["anomaly"].to_numpy(dtype="int8")
    scaler = StandardScaler().fit(raw_train)
    scaled_train, scaled_query = (
        scaler.transform(raw_train).astype("float32"),
        scaler.transform(raw_query).astype("float32"),
    )
    AUDIT.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, AUDIT / "scaler.joblib")
    atomic_npz(
        AUDIT / "inputs.npz",
        scaled_query=scaled_query,
        query_raw_index=query_raw,
        anomaly=y_query,
    )
    model = initialize("cuda")
    model.fit(scaled_train, labels)
    atomic_json(
        AUDIT / "provenance.json",
        provenance(
            model, raw_train, scaled_train, raw_query, scaled_query, query_raw, payload
        ),
    )
    stages: list[dict[str, Any]] = [
        capture(model, scaled_query, "P1_fit_immediate"),
        capture(model, scaled_query, "P2_live_repeat"),
    ]
    save_fitted_tabpfn_model(model, AUDIT / "model.tabpfn_fit")
    stages.append(capture(model, scaled_query, "P3_live_after_save"))
    same_process = load_fitted_tabpfn_model(AUDIT / "model.tabpfn_fit", device="cuda")
    stages.append(capture(same_process, scaled_query, "P4_same_process_gpu_load"))
    subprocess.run([sys.executable, __file__, "--stage", "p5"], check=True)
    cpu = load_fitted_tabpfn_model(AUDIT / "model.tabpfn_fit", device="cpu")
    stages.extend(
        [
            capture(cpu, scaled_query, "P6_cpu_load_first"),
            capture(cpu, scaled_query, "P6_cpu_load_second"),
        ]
    )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    deterministic = load_fitted_tabpfn_model(AUDIT / "model.tabpfn_fit", device="cuda")
    stages.extend(
        [
            capture(deterministic, scaled_query, "P7_deterministic_gpu_first"),
            capture(deterministic, scaled_query, "P7_deterministic_gpu_second"),
        ]
    )
    order = [stage["stage"] for stage in stages]
    pairs = [compare(order[index], order[index + 1]) for index in range(len(order) - 1)]
    first = next((pair for pair in pairs if not pair["bit_exact"]), None)
    atomic_json(
        AUDIT / "summary.json",
        {
            "provenance": "provenance.json",
            "stages": stages,
            "adjacent_comparisons": pairs,
            "first_prediction_difference": first,
            "gate": "Do not reopen Path A unless P1-P7 and three sentinels satisfy a new predeclared tolerance.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
