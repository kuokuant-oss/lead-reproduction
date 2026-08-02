"""Frozen, fail-closed protocol primitives for M5 E7.

This module intentionally contains no model fitting and no holdout label access.
The executable runner imports these small, directly-testable primitives.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "data" / "processed" / "m5_e7_full_capacity_tree_strategy"
PHASE0_COMMIT = "2b4c9bc77c274912da9786c3f6b1cfaa43cfc84c"
MODEL_ORDER = ("lightgbm", "xgboost", "catboost", "hist_gradient_boosting")
SUPPORT_CELLS = ("00", "01", "10", "11")
NEUTRAL_SLOTS = ("n00", "n01", "n10", "n11")
CS = (0.001, 0.01, 0.1, 1.0, 10.0)
EXPECTED_OOF_COMPONENTS = 5 * 8 * 4
EXPECTED_FINAL_COMPONENTS = 8 * 4
EXPECTED_COMPONENTS = EXPECTED_OOF_COMPONENTS + EXPECTED_FINAL_COMPONENTS


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(values: np.ndarray, dtype: str = "<i8") -> str:
    return sha256_bytes(np.ascontiguousarray(values).astype(dtype).tobytes())


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def atomic_json(path: Path, payload: Any) -> str:
    """Write LF JSON atomically and return its post-rename digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json(payload))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return sha256_file(path)


def atomic_npy(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npy", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return sha256_file(path)


def atomic_text(path: Path, value: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value.replace("\r\n", "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return sha256_file(path)


def require_local_cpu() -> None:
    """Fail before work if any forbidden execution topology is configured."""
    if os.environ.get("NO_REMOTE") != "1":
        raise SystemExit("E7 requires NO_REMOTE=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("E7 requires CUDA_VISIBLE_DEVICES to be empty")
    forbidden = ROOT / "data" / "processed" / "m5_e6"
    if forbidden.exists():
        raise SystemExit("E7 refuses an active M5 E6 artifact root")


def resource_environment() -> dict[str, str]:
    return {
        "NO_REMOTE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def component_params(component: str) -> dict[str, Any]:
    params = {
        "lightgbm": {
            "n_estimators": 100,
            "verbose": -1,
            "random_state": 42,
            "num_threads": 1,
        },
        "xgboost": {
            "n_estimators": 100,
            "eval_metric": "logloss",
            "verbosity": 0,
            "random_state": 42,
            "nthread": 1,
        },
        "catboost": {
            "iterations": 1000,
            "verbose": False,
            "random_seed": 42,
            "allow_writing_files": False,
            "thread_count": 1,
        },
        "hist_gradient_boosting": {"max_iter": 100, "random_state": 42},
    }
    try:
        return params[component]
    except KeyError as error:
        raise ValueError(f"unknown E7 component: {component}") from error


def unit_id(phase: str, fold: str, family: str, slot: str, component: str) -> str:
    return f"{phase}__{fold}__{family}__{slot}__{component}"


def expected_units(phase: str) -> list[str]:
    folds = [f"fold{number}" for number in range(5)] if phase == "oof" else ["all_even"]
    output: list[str] = []
    for fold in folds:
        for family, slots in (("support", SUPPORT_CELLS), ("neutral", NEUTRAL_SLOTS)):
            for slot in slots:
                output.extend(
                    unit_id(phase, fold, family, slot, component)
                    for component in MODEL_ORDER
                )
    return output


def environment_manifest() -> dict[str, Any]:
    import catboost
    import joblib
    import lightgbm
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "lightgbm": lightgbm.__version__,
            "xgboost": xgboost.__version__,
            "catboost": catboost.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "resource_environment": resource_environment(),
        "tree_device": "cpu_only",
    }


def protocol() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "M5 E7 Historical-Full-Budget Support-Aware Tree Strategy",
        "phase0_commit": PHASE0_COMMIT,
        "execution": {
            "local_cpu_only": True,
            "remote_commands_allowed": False,
            "tabpfn_allowed": False,
            "active_e6_allowed": False,
            "process_pool_allowed": False,
            "subprocess_component_execution": True,
        },
        "split": {
            "fit": "building_id % 2 == 0",
            "holdout": "building_id % 2 == 1",
            "outer_folds": 5,
            "unit": "building_id",
            "balance": ["steam_rows", "steam_anomalies", "site_distribution"],
        },
        "features": {
            "tag": "F4",
            "count": 137,
            "value_change_regime": "timestamp_merge",
            "dtype": "float32",
        },
        "historical_downsampling": {
            "seeds": [10, 20],
            "order": "[negative_seed10, positive, negative_seed20, positive]",
            "positive_rows_duplicated": True,
        },
        "experts": {
            "support_slots": list(SUPPORT_CELLS),
            "neutral_slots": list(NEUTRAL_SLOTS),
            "components": list(MODEL_ORDER),
            "ensemble": "equal_weight_probability_mean",
            "component_params": {name: component_params(name) for name in MODEL_ORDER},
        },
        "meta_model": {
            "features": [
                "s11",
                "grand_mean",
                "negative_support_response",
                "positive_support_response",
                "interaction",
            ],
            "estimator": "StandardScaler + LogisticRegression(lbfgs, L2)",
            "cs": list(CS),
            "tie_rule": "within 1e-4 mean AP choose smaller C",
            "fold_isolation": True,
        },
        "census": {
            "oof_component_fits": EXPECTED_OOF_COMPONENTS,
            "final_component_fits": EXPECTED_FINAL_COMPONENTS,
            "total_component_fits": EXPECTED_COMPONENTS,
        },
        "score_firewall": {
            "fresh_process": True,
            "odd_anomaly_labels_read": False,
            "outputs": [
                "support experts",
                "neutral experts",
                "support_stack",
                "neutral_stack",
                "component scores",
                "canonical raw_index",
            ],
        },
        "bootstrap": {
            "draws": 1000,
            "master_seed": 20260802,
            "namespace_code": 7007,
            "cluster_code": 1,
            "cluster": "odd steam building_id",
            "estimator": "exact_weighted_average_precision",
        },
        "decision": {
            "delta_ap_min": 0.01,
            "labels": [
                "E7_FULL_CAPACITY_CONTEXT_STRATEGY_CONFIRMED",
                "E7_GAIN_NOT_CONTEXT_SPECIFIC",
                "E7_DIRECTIONALLY_CONSISTENT_BUT_INCONCLUSIVE",
                "E7_NOT_CONFIRMED",
                "E7_EXECUTION_INCOMPLETE",
            ],
        },
    }


def factor_features(scores: dict[str, np.ndarray], prefix: str = "s") -> np.ndarray:
    a, b, c, d = (scores[f"{prefix}{cell}"] for cell in SUPPORT_CELLS)
    grand = (a + b + c + d) / 4.0
    negative = ((b - a) + (d - c)) / 2.0
    positive = ((c - a) + (d - b)) / 2.0
    interaction = d - c - b + a
    return np.column_stack((d, grand, negative, positive, interaction)).astype(
        "float32"
    )
