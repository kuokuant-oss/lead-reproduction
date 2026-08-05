"""External-building-validation early stopping for the M5 tree ensemble."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


MODEL_ORDER = (
    "lightgbm",
    "xgboost",
    "catboost",
    "hist_gradient_boosting",
)
DEFAULT_CEILINGS = {
    "lightgbm": 5_000,
    "xgboost": 5_000,
    "catboost": 5_000,
    "hist_gradient_boosting": 1_000,
}


def model_matrix(model_name: str, values: np.ndarray) -> np.ndarray:
    if model_name == "hist_gradient_boosting":
        # Earlier third-party models can mark a shared array read-only. Reuse
        # writable storage, but make the required private copy at this final
        # consumer when an in-place NaN conversion would otherwise fail.
        writable = values if values.flags.writeable else values.copy()
        return np.nan_to_num(writable, nan=0.0, copy=False)
    return values


def early_stopping_contract(
    *,
    seed: int = 42,
    patience: int = 100,
    hist_patience: int = 20,
    min_delta: float = 1e-5,
    ceilings: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    limits = {**DEFAULT_CEILINGS, **(ceilings or {})}
    return {
        "lightgbm": {
            "class": "LGBMClassifier",
            "ceiling": int(limits["lightgbm"]),
            "selection_metric": "roc_auc",
            "patience": int(patience),
            "min_delta": float(min_delta),
            "params": {
                "n_estimators": int(limits["lightgbm"]),
                "verbose": -1,
                "random_state": seed,
            },
        },
        "xgboost": {
            "class": "XGBClassifier",
            "ceiling": int(limits["xgboost"]),
            "selection_metric": "roc_auc",
            "patience": int(patience),
            "min_delta": 0.0,
            "params": {
                "n_estimators": int(limits["xgboost"]),
                "eval_metric": "auc",
                "early_stopping_rounds": int(patience),
                "verbosity": 0,
                "random_state": seed,
            },
        },
        "catboost": {
            "class": "CatBoostClassifier",
            "ceiling": int(limits["catboost"]),
            "selection_metric": "roc_auc",
            "patience": int(patience),
            "min_delta": 0.0,
            "params": {
                "iterations": int(limits["catboost"]),
                "eval_metric": "AUC",
                "verbose": False,
                "random_seed": seed,
                "allow_writing_files": False,
            },
        },
        "hist_gradient_boosting": {
            "class": "HistGradientBoostingClassifier",
            "ceiling": int(limits["hist_gradient_boosting"]),
            "selection_metric": "roc_auc",
            "patience": int(hist_patience),
            "min_delta": float(min_delta),
            "params": {
                "max_iter": int(limits["hist_gradient_boosting"]),
                "early_stopping": True,
                "scoring": "roc_auc",
                "n_iter_no_change": int(hist_patience),
                "tol": float(min_delta),
                "validation_fraction": None,
                "random_state": seed,
            },
        },
    }


def _validate_inputs(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_early_stop: np.ndarray,
    y_early_stop: np.ndarray,
) -> None:
    if x_fit.ndim != 2 or x_early_stop.ndim != 2:
        raise ValueError("tree matrices must be two-dimensional")
    if x_fit.shape[1] != x_early_stop.shape[1]:
        raise ValueError("tree fit and early-stop feature counts differ")
    if len(x_fit) != len(y_fit) or len(x_early_stop) != len(y_early_stop):
        raise ValueError("tree matrix and label lengths differ")
    if len(np.unique(y_fit)) != 2:
        raise ValueError("tree fit rows must contain both classes")
    if len(np.unique(y_early_stop)) != 2:
        raise ValueError("tree early-stop rows must contain both classes")


def predict_probability(model_name: str, model: Any, values: np.ndarray) -> np.ndarray:
    return np.asarray(
        model.predict_proba(model_matrix(model_name, values))[:, 1], dtype="float64"
    )


def _history_for(model_name: str, model: Any) -> dict[str, Any]:
    if model_name == "lightgbm":
        return getattr(model, "evals_result_", {})
    if model_name == "xgboost":
        return model.evals_result()
    if model_name == "catboost":
        return model.get_evals_result()
    return {
        "train_score": [float(value) for value in model.train_score_],
        "validation_score": [float(value) for value in model.validation_score_],
    }


def _best_iteration(model_name: str, model: Any) -> int:
    if model_name == "lightgbm":
        return int(model.best_iteration_)
    if model_name == "xgboost":
        return int(model.best_iteration) + 1
    if model_name == "catboost":
        return int(model.get_best_iteration()) + 1
    return int(model.n_iter_)


def fit_early_stopped_models(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_early_stop: np.ndarray,
    y_early_stop: np.ndarray,
    *,
    seed: int = 42,
    patience: int = 100,
    hist_patience: int = 20,
    min_delta: float = 1e-5,
    ceilings: dict[str, int] | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Fit all components against one explicit, building-disjoint ES set."""
    y_fit = np.asarray(y_fit, dtype="int8")
    y_early_stop = np.asarray(y_early_stop, dtype="int8")
    _validate_inputs(x_fit, y_fit, x_early_stop, y_early_stop)
    contract = early_stopping_contract(
        seed=seed,
        patience=patience,
        hist_patience=hist_patience,
        min_delta=min_delta,
        ceilings=ceilings,
    )
    models: dict[str, Any] = {}
    records: dict[str, Any] = {}
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for name in MODEL_ORDER:
        spec = contract[name]
        model_checkpoint = (
            checkpoint_dir / f"{name}.joblib" if checkpoint_dir is not None else None
        )
        record_checkpoint = (
            checkpoint_dir / f"{name}.json" if checkpoint_dir is not None else None
        )
        if (
            resume
            and model_checkpoint is not None
            and record_checkpoint is not None
            and model_checkpoint.exists()
            and record_checkpoint.exists()
        ):
            models[name] = joblib.load(model_checkpoint)
            records[name] = json.loads(record_checkpoint.read_text(encoding="utf-8"))
            continue
        if name == "lightgbm":
            model = lgb.LGBMClassifier(**spec["params"])
            callbacks = [
                lgb.early_stopping(
                    spec["patience"],
                    first_metric_only=True,
                    min_delta=spec["min_delta"],
                    verbose=False,
                )
            ]
            started = time.perf_counter()
            model.fit(
                x_fit,
                y_fit,
                eval_set=[(x_early_stop, y_early_stop)],
                eval_names=["early_stop"],
                eval_metric="auc",
                callbacks=callbacks,
            )
        elif name == "xgboost":
            model = xgb.XGBClassifier(**spec["params"])
            started = time.perf_counter()
            model.fit(
                x_fit,
                y_fit,
                eval_set=[(x_early_stop, y_early_stop)],
                verbose=False,
            )
        elif name == "catboost":
            model = CatBoostClassifier(**spec["params"])
            started = time.perf_counter()
            model.fit(
                x_fit,
                y_fit,
                eval_set=(x_early_stop, y_early_stop),
                use_best_model=True,
                early_stopping_rounds=spec["patience"],
                verbose=False,
            )
        else:
            model = HistGradientBoostingClassifier(**spec["params"])
            started = time.perf_counter()
            model.fit(
                model_matrix(name, x_fit),
                y_fit,
                X_val=model_matrix(name, x_early_stop),
                y_val=y_early_stop,
            )
        fit_seconds = time.perf_counter() - started
        scores = predict_probability(name, model, x_early_stop)
        best_iteration = _best_iteration(name, model)
        records[name] = {
            "fit_seconds": fit_seconds,
            "best_iteration": best_iteration,
            "iteration_ceiling": int(spec["ceiling"]),
            "ceiling_hit": bool(best_iteration >= int(spec["ceiling"])),
            "stop_reason": (
                "iteration_ceiling"
                if best_iteration >= int(spec["ceiling"])
                else "early_stopping"
            ),
            "early_stop_roc_auc": float(roc_auc_score(y_early_stop, scores)),
            "early_stop_pr_auc": float(average_precision_score(y_early_stop, scores)),
            "history": _history_for(name, model),
        }
        models[name] = model
        if model_checkpoint is not None and record_checkpoint is not None:
            temporary_model = model_checkpoint.with_suffix(".joblib.tmp")
            joblib.dump(model, temporary_model)
            with temporary_model.open("rb+") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary_model, model_checkpoint)
            temporary_record = record_checkpoint.with_suffix(".json.tmp")
            with temporary_record.open("w", encoding="utf-8") as stream:
                json.dump(records[name], stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_record, record_checkpoint)
    return models, records, contract


def ensemble_probabilities(
    models: dict[str, Any], values: np.ndarray
) -> dict[str, np.ndarray]:
    scores = {
        name: predict_probability(name, models[name], values) for name in MODEL_ORDER
    }
    scores["ensemble"] = np.mean([scores[name] for name in MODEL_ORDER], axis=0)
    return scores


def refit_models_at_selected_iterations(
    values: np.ndarray,
    labels: np.ndarray,
    records: dict[str, Any],
    contract: dict[str, dict[str, Any]],
    *,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Refit on every available row after ES has selected iteration counts."""
    if len(np.unique(labels)) != 2:
        raise ValueError("full refit rows must contain both classes")
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    for name in MODEL_ORDER:
        checkpoint = (
            checkpoint_dir / f"{name}.joblib" if checkpoint_dir is not None else None
        )
        if resume and checkpoint is not None and checkpoint.exists():
            models[name] = joblib.load(checkpoint)
            continue
        iterations = int(records[name]["best_iteration"])
        params = dict(contract[name]["params"])
        if name == "lightgbm":
            params["n_estimators"] = iterations
            model = lgb.LGBMClassifier(**params)
        elif name == "xgboost":
            params["n_estimators"] = iterations
            params.pop("early_stopping_rounds", None)
            model = xgb.XGBClassifier(**params)
        elif name == "catboost":
            params["iterations"] = iterations
            model = CatBoostClassifier(**params)
        else:
            params["max_iter"] = iterations
            params["early_stopping"] = False
            params.pop("validation_fraction", None)
            model = HistGradientBoostingClassifier(**params)
        started = time.perf_counter()
        model.fit(model_matrix(name, values), labels)
        records[name]["full_refit_seconds"] = time.perf_counter() - started
        models[name] = model
        if checkpoint is not None:
            temporary = checkpoint.with_suffix(".joblib.tmp")
            joblib.dump(model, temporary)
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, checkpoint)
    return models
