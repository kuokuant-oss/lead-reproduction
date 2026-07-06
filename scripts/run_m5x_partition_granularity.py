"""M5.x partition-granularity comparison under timestamp-merge causal lags."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
import traceback
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from lead import (
    BASELINE_FEATURE_COLS,
    DOWNSAMPLE_SEEDS,
    MODEL_SEEDS,
    PAST_SHIFTS,
    PROC,
    RANDOM_STATE,
    ROOT,
    SHIFTS,
    add_value_change_features,
    load_m3_frame,
    write_json_with_provenance,
)


EXPERIMENT = "m5x_partition_granularity"
VALUE_CHANGE_REGIME = "timestamp_merge"
REGIME_LABEL = "timestamp_merge(causal77)"
TARGET_RECALL = 0.90
MODEL_ORDER = (
    "lightgbm",
    "xgboost",
    "catboost",
    "hist_gradient_boosting",
    "ensemble",
    "tabpfn",
)
TREE_MODEL_ORDER = MODEL_ORDER[:4]
CONFIGS: dict[str, tuple[str, ...]] = {
    "C1": ("building_id", "meter"),
    "C2": ("site_id", "meter"),
    "C3": ("meter",),
    "C4": ("primary_use", "meter"),
}
CONFIG_LABELS = {
    "C1": "(building_id, meter)",
    "C2": "(site_id, meter)",
    "C3": "(meter,)",
    "C4": "(primary_use, meter)",
}
TIME_SPLIT = {
    "train": "2016-01..2016-08",
    "calib": "2016-09..2016-10",
    "test": "2016-11..2016-12",
}


def log(message: str) -> None:
    print(message, flush=True)


def default_model_path() -> Path:
    if os.environ.get("TABPFN_MODEL_CACHE_DIR"):
        return Path(os.environ["TABPFN_MODEL_CACHE_DIR"]) / (
            "tabpfn-v3-classifier-v3_default.ckpt"
        )
    return ROOT / ".tabpfn-cache" / "tabpfn-v3-classifier-v3_default.ckpt"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "m5x_partition_granularity.json",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=tuple(CONFIGS),
        default=["C1", "C2", "C3", "C4"],
    )
    parser.add_argument("--fit-rows", type=int, default=10_000)
    parser.add_argument("--eval-rows", type=int, default=4_000)
    parser.add_argument("--calib-rows", type=int, default=4_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(MODEL_SEEDS))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--max-units", type=int, default=400)
    parser.add_argument("--skip-tabpfn", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    args = parser.parse_args(argv)
    if args.smoke:
        args.fit_rows = min(args.fit_rows, 400)
        args.eval_rows = min(args.eval_rows, 400)
        args.calib_rows = min(args.calib_rows, 400)
        args.seeds = list(args.seeds)[:2]
        args.max_units = min(args.max_units, 4)
    return args


def torch_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "tabpfn_installed": importlib.util.find_spec("tabpfn") is not None,
        "tabpfn_no_browser": bool(os.environ.get("TABPFN_NO_BROWSER")),
        "tabpfn_disable_telemetry": bool(os.environ.get("TABPFN_DISABLE_TELEMETRY")),
    }
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        env["nvidia_smi_gpus"] = [
            line.strip() for line in smi.stdout.splitlines() if line.strip()
        ]
    except (OSError, subprocess.CalledProcessError):
        env["nvidia_smi_gpus"] = []
    if not env["torch_installed"]:
        env["device"] = "cpu"
        return env

    import torch

    env.update(
        {
            "tabpfn_version": metadata.version("tabpfn")
            if env["tabpfn_installed"]
            else None,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
    )
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        env.update(
            {
                "device": "cuda",
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_vram_mib": int(props.total_memory // (1024 * 1024)),
            }
        )
    else:
        env["device"] = "cpu"
    return env


def tabpfn_classifier(device: str, model_path: Path | None):
    from tabpfn import TabPFNClassifier

    kwargs: dict[str, Any] = {"device": device}
    if model_path is not None:
        kwargs["model_path"] = model_path
    try:
        return TabPFNClassifier(**kwargs)
    except TypeError:
        if model_path is not None:
            raise
        return TabPFNClassifier()


class Runner:
    def __init__(self, args: argparse.Namespace, env: dict[str, Any]) -> None:
        self.args = args
        self.env = env
        self.device = str(env.get("device", "cpu"))
        self.model_path = (
            args.model_path.resolve() if args.model_path is not None else None
        )
        self.local_ckpt = bool(self.model_path and self.model_path.is_file())
        self.tabpfn_ok = (
            not args.skip_tabpfn
            and env.get("tabpfn_installed")
            and env.get("torch_installed")
            and self.local_ckpt
        )
        self._tabpfn_model = None

    def fit_tabpfn(self, x_train, y_train):
        if not self.tabpfn_ok:
            return None
        if self._tabpfn_model is None:
            self._tabpfn_model = tabpfn_classifier(
                self.device,
                self.model_path if self.local_ckpt else None,
            )
        model = self._tabpfn_model
        model.fit(x_train, y_train)
        return model


def threshold_for_recall(y_true, pred: np.ndarray, target_recall: float) -> float:
    y = np.asarray(y_true, dtype=np.int8)
    if int(y.sum()) == 0:
        return 1.0
    order = np.argsort(-pred)
    y_sorted = y[order]
    pred_sorted = pred[order]
    recall = np.cumsum(y_sorted) / y.sum()
    idx = int(np.searchsorted(recall, target_recall, side="left"))
    idx = min(idx, len(pred_sorted) - 1)
    return float(pred_sorted[idx])


def binary_summary(y_true, pred: np.ndarray, *, threshold: float) -> dict[str, Any]:
    pred_label = (pred >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, pred_label, labels=[0, 1]).ravel()
    total = int(tn + fp + fn + tp)
    anomalies = int(tp + fn)
    normal = int(tn + fp)
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / anomalies) if anomalies else 0.0
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / total) if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "total": total,
            "anomalies_total": anomalies,
            "normal_total": normal,
            "false_alarm_rate": float(fp / normal) if normal else 0.0,
            "anomaly_capture_rate": float(tp / anomalies) if anomalies else 0.0,
        },
    }


def metric_summary(
    y_true,
    pred: np.ndarray,
    *,
    calib_y=None,
    calib_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        roc = None
        pr = None
    else:
        roc = float(roc_auc_score(y_true, pred))
        pr = float(average_precision_score(y_true, pred))
    threshold_source_y = y_true if calib_y is None else calib_y
    threshold_source_pred = pred if calib_pred is None else calib_pred
    recall_threshold = threshold_for_recall(
        threshold_source_y,
        np.asarray(threshold_source_pred),
        TARGET_RECALL,
    )
    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "operation_points": {
            "threshold_0_5": binary_summary(y_true, pred, threshold=0.5),
            "calib_fixed_recall_0_90": binary_summary(
                y_true,
                pred,
                threshold=recall_threshold,
            ),
        },
    }


def balanced_subsample_indices(
    ds_idx: np.ndarray, y, max_rows: int, seed: int
) -> np.ndarray:
    if len(ds_idx) == 0:
        return ds_idx
    rng = np.random.RandomState(seed)
    y_ds = y.loc[ds_idx].to_numpy()
    pos_positions = np.flatnonzero(y_ds == 1)
    neg_positions = np.flatnonzero(y_ds == 0)
    if len(pos_positions) == 0 or len(neg_positions) == 0:
        return np.asarray([], dtype=ds_idx.dtype)
    per_class = (
        max_rows // 2 if max_rows > 0 else min(len(pos_positions), len(neg_positions))
    )
    n_pos = min(per_class, len(pos_positions))
    n_neg = min(per_class, len(neg_positions))
    chosen_pos = rng.choice(pos_positions, size=n_pos, replace=False)
    chosen_neg = rng.choice(neg_positions, size=n_neg, replace=False)
    chosen = np.concatenate([chosen_neg, chosen_pos])
    rng.shuffle(chosen)
    return ds_idx[chosen]


def index_record(index: np.ndarray) -> dict[str, Any]:
    values = [int(i) for i in index]
    payload = ",".join(map(str, values)).encode("utf-8")
    return {
        "count": len(values),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "first10": values[:10],
        "last10": values[-10:],
    }


def tree_models(seed: int, *, smoke: bool = False) -> dict[str, Any]:
    if smoke:
        return {
            "lightgbm": lgb.LGBMClassifier(
                n_estimators=10,
                verbose=-1,
                random_state=seed,
            )
        }
    n_estimators = 10 if smoke else 100
    catboost_iterations = 20 if smoke else 1000
    hgb_iterations = 10 if smoke else 100
    return {
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=n_estimators,
            verbose=-1,
            random_state=seed,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=n_estimators,
            eval_metric="logloss",
            verbosity=0,
            random_state=seed,
        ),
        "catboost": CatBoostClassifier(
            iterations=catboost_iterations,
            verbose=False,
            random_seed=seed,
            allow_writing_files=False,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=hgb_iterations,
            random_state=seed,
        ),
    }


def safe_predict_proba(
    model: Any, x_score: np.ndarray, *, fill_nan: bool
) -> np.ndarray:
    x = np.nan_to_num(x_score, nan=0.0) if fill_nan else x_score
    return model.predict_proba(x)[:, 1]


def fit_score_all_models(
    runner: Runner,
    matrices: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tree_preds: dict[str, dict[str, np.ndarray]] = {}
    models = tree_models(seed, smoke=bool(getattr(runner.args, "smoke", False)))
    for name, model in models.items():
        fill_nan = name == "hist_gradient_boosting"
        x_fit = (
            np.nan_to_num(matrices["x_fit"], nan=0.0) if fill_nan else matrices["x_fit"]
        )
        t0 = time.perf_counter()
        model.fit(x_fit, matrices["y_fit"])
        pred_calib = safe_predict_proba(model, matrices["x_calib"], fill_nan=fill_nan)
        pred_eval = safe_predict_proba(model, matrices["x_eval"], fill_nan=fill_nan)
        elapsed = time.perf_counter() - t0
        tree_preds[name] = {"calib": pred_calib, "eval": pred_eval}
        out[name] = {
            "status": "completed",
            "fit_predict_seconds": float(elapsed),
            "calib_idx_sha256": matrices["calib_idx_record"]["sha256"],
            "calib_pred": pred_calib,
            "eval_pred": pred_eval,
        }
    if all(name in tree_preds for name in TREE_MODEL_ORDER):
        out["ensemble"] = {
            "status": "completed",
            "fit_predict_seconds": None,
            "calib_idx_sha256": matrices["calib_idx_record"]["sha256"],
            "calib_pred": np.mean(
                [tree_preds[name]["calib"] for name in TREE_MODEL_ORDER],
                axis=0,
            ),
            "eval_pred": np.mean(
                [tree_preds[name]["eval"] for name in TREE_MODEL_ORDER],
                axis=0,
            ),
        }
    else:
        out["ensemble"] = {
            "status": "skipped",
            "reason": "tree models unavailable",
        }
    for name in TREE_MODEL_ORDER:
        out.setdefault(
            name,
            {
                "status": "skipped",
                "reason": "model disabled in smoke mode",
            },
        )
    if runner.tabpfn_ok:
        t0 = time.perf_counter()
        try:
            model = runner.fit_tabpfn(matrices["x_fit"], matrices["y_fit"])
            pred_calib = model.predict_proba(matrices["x_calib"])[:, 1]
            pred_eval = model.predict_proba(matrices["x_eval"])[:, 1]
            out["tabpfn"] = {
                "status": "completed",
                "fit_predict_seconds": float(time.perf_counter() - t0),
                "calib_idx_sha256": matrices["calib_idx_record"]["sha256"],
                "calib_pred": pred_calib,
                "eval_pred": pred_eval,
            }
        except Exception as exc:  # pragma: no cover - GPU/runtime dependent.
            out["tabpfn"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=6),
            }
    else:
        out["tabpfn"] = {
            "status": "skipped",
            "reason": "tabpfn unavailable, --skip-tabpfn, or no local checkpoint",
        }
    return out


def past_feature_cols() -> list[str]:
    past_cols = [f"lag_value_diff_{n}" for n in PAST_SHIFTS] + [
        f"lag_value_ratio_{n}" for n in PAST_SHIFTS
    ]
    feature_cols = list(BASELINE_FEATURE_COLS) + past_cols
    if len(feature_cols) != 77:
        raise AssertionError(f"Expected 77 causal features, got {len(feature_cols)}")
    return feature_cols


def add_primary_use(df: pd.DataFrame) -> pd.DataFrame:
    if "primary_use" in df.columns:
        return df
    meta = pd.read_csv(ROOT / "data" / "raw" / "m3" / "building_metadata.csv")[
        ["building_id", "primary_use"]
    ]
    return df.merge(meta, on="building_id", how="left")


def build_causal_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = add_primary_use(df.copy())
    frame = add_value_change_features(
        frame,
        list(SHIFTS),
        value_change_regime=VALUE_CHANGE_REGIME,
    )
    features = past_feature_cols()
    missing = [col for col in features if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing causal feature columns: {missing[:5]}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame, features


def assign_windows(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["timestamp"])
    window = pd.Series("unused", index=df.index, dtype="object")
    window.loc[(ts >= "2016-01-01") & (ts < "2016-09-01")] = "train"
    window.loc[(ts >= "2016-09-01") & (ts < "2016-11-01")] = "calib"
    window.loc[(ts >= "2016-11-01") & (ts < "2017-01-01")] = "test"
    return window


def eval_indices(df: pd.DataFrame, *, eval_rows: int, seed: int) -> np.ndarray:
    test_idx = df.index[df["window"].eq("test")].to_numpy()
    if eval_rows <= 0 or len(test_idx) <= eval_rows:
        return test_idx
    rng = np.random.RandomState(seed)
    return np.asarray(rng.choice(test_idx, size=eval_rows, replace=False))


def sample_calib_indices(
    calib_idx: np.ndarray, *, calib_rows: int, seed: int
) -> np.ndarray:
    if calib_rows <= 0 or len(calib_idx) <= calib_rows:
        return calib_idx
    rng = np.random.RandomState(seed)
    return np.asarray(rng.choice(calib_idx, size=calib_rows, replace=False))


def nan_fraction(df: pd.DataFrame, index: np.ndarray, feature_cols: list[str]) -> float:
    if len(index) == 0:
        return 0.0
    return float(df.loc[index, feature_cols].isna().to_numpy().mean())


def unit_keys(df: pd.DataFrame, fields: tuple[str, ...]) -> pd.Series:
    if len(fields) == 1:
        return df[fields[0]]
    return pd.util.hash_pandas_object(df.loc[:, list(fields)], index=False)


def group_indices_for_keys(
    index: np.ndarray,
    key_values: np.ndarray,
) -> dict[Any, np.ndarray]:
    if len(index) == 0:
        return {}
    grouped = pd.Series(key_values).groupby(key_values, sort=False).indices
    return {unit: np.asarray(index[positions]) for unit, positions in grouped.items()}


def build_unit_index_map(
    df: pd.DataFrame,
    keys: pd.Series,
    eval_idx: np.ndarray,
    selected_units: set[Any],
) -> dict[Any, dict[str, np.ndarray]]:
    index = df.index.to_numpy()
    key_values = keys.to_numpy()
    eval_keys = keys.loc[eval_idx].to_numpy()
    eval_units = sorted(set(eval_keys))
    eval_map = {
        unit: eval_idx[np.flatnonzero(eval_keys == unit)] for unit in eval_units
    }
    selected_mask = keys.isin(selected_units).to_numpy()
    train_mask = df["window"].eq("train").to_numpy() & selected_mask
    calib_mask = df["window"].eq("calib").to_numpy() & selected_mask
    train_map = group_indices_for_keys(index[train_mask], key_values[train_mask])
    calib_map = group_indices_for_keys(index[calib_mask], key_values[calib_mask])
    return {
        unit: {
            "train": train_map.get(unit, np.asarray([], dtype=df.index.dtype)),
            "calib": calib_map.get(unit, np.asarray([], dtype=df.index.dtype)),
            "eval": eval_map[unit],
        }
        for unit in eval_units
    }


def select_units(
    units: list[Any],
    *,
    max_units: int,
    seed: int,
) -> tuple[set[Any], dict[str, Any]]:
    if max_units <= 0 or len(units) <= max_units:
        return set(units), {
            "limited": False,
            "n_available": len(units),
            "n_selected": len(units),
        }
    rng = np.random.RandomState(seed)
    chosen = sorted(
        rng.choice(np.asarray(units), size=max_units, replace=False).tolist()
    )
    return set(chosen), {
        "limited": True,
        "n_available": len(units),
        "n_selected": len(chosen),
    }


def scorable_train_index(df: pd.DataFrame, idx: np.ndarray) -> bool:
    if len(idx) == 0:
        return False
    labels = df.loc[idx, "anomaly"]
    return bool((labels == 1).any() and (labels == 0).any())


def make_matrices(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    eval_idx: np.ndarray,
    fit_rows: int,
    calib_rows: int,
    seed: int,
) -> dict[str, Any] | None:
    calib_idx = sample_calib_indices(
        calib_idx,
        calib_rows=calib_rows,
        seed=seed,
    )
    fit_idx = balanced_subsample_indices(train_idx, df["anomaly"], fit_rows, seed)
    if len(fit_idx) == 0 or len(calib_idx) == 0 or len(eval_idx) == 0:
        return None
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(df.loc[fit_idx, feature_cols])
    return {
        "fit_idx": fit_idx,
        "calib_idx": calib_idx,
        "calib_idx_record": index_record(calib_idx),
        "x_fit": x_fit,
        "y_fit": df.loc[fit_idx, "anomaly"],
        "x_calib": scaler.transform(df.loc[calib_idx, feature_cols]),
        "y_calib": df.loc[calib_idx, "anomaly"],
        "x_eval": scaler.transform(df.loc[eval_idx, feature_cols]),
        "y_eval": df.loc[eval_idx, "anomaly"],
    }


def fit_unit_models(
    runner: Runner,
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    eval_idx: np.ndarray,
    fit_rows: int,
    calib_rows: int,
    seed: int,
) -> dict[str, Any]:
    matrices = make_matrices(
        df,
        feature_cols,
        train_idx=train_idx,
        calib_idx=calib_idx,
        eval_idx=eval_idx,
        fit_rows=fit_rows,
        calib_rows=calib_rows,
        seed=seed,
    )
    if matrices is None:
        return {
            "status": "skipped",
            "reason": "empty fit/calib/eval after scorable check",
        }
    model_cells = fit_score_all_models(runner, matrices, seed=seed)
    return {
        "status": "completed",
        "fit_idx": matrices["fit_idx"],
        "calib_idx": matrices["calib_idx"],
        "calib_idx_record": index_record(matrices["calib_idx"]),
        "eval_idx": eval_idx,
        "y_calib": matrices["y_calib"].to_numpy(),
        "y_eval": matrices["y_eval"].to_numpy(),
        "models": model_cells,
    }


def summarize_unit_rows(
    unit_index_map: dict[Any, dict[str, np.ndarray]],
    units: set[Any],
) -> dict[str, Any]:
    rows = []
    for unit in sorted(units):
        splits = unit_index_map.get(unit)
        if splits is None:
            rows.append((0, 0, 0))
        else:
            rows.append(
                (
                    int(len(splits["train"])),
                    int(len(splits["calib"])),
                    int(len(splits["eval"])),
                )
            )
    if not rows:
        return {"n_units": 0}
    arr = np.asarray(rows)
    return {
        "n_units": int(len(rows)),
        "train_rows": quantile_summary(arr[:, 0]),
        "calib_rows": quantile_summary(arr[:, 1]),
        "test_rows": quantile_summary(arr[:, 2]),
    }


def quantile_summary(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {}
    return {
        "min": int(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": int(np.max(values)),
    }


def fraction_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(len(values)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def empty_nan_records() -> dict[str, dict[str, list[float]]]:
    return {
        cohort: {"fit": [], "calib": [], "eval": []}
        for cohort in ("non_fallback", "fallback")
    }


def summarize_nan_records(
    records: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        cohort: {
            split: fraction_summary(values) for split, values in split_records.items()
        }
        for cohort, split_records in records.items()
    }


def aggregate_nan_summaries(
    seed_cells: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    records = empty_nan_records()
    for cell in seed_cells:
        summary = cell.get("per_unit_nan_fraction_summary", {})
        for cohort in records:
            for split in records[cohort]:
                values = summary.get(cohort, {}).get(split, {})
                for key in ("min", "median", "max"):
                    value = values.get(key)
                    if value is not None:
                        records[cohort][split].append(float(value))
    return summarize_nan_records(records)


def aggregate_count_assert(
    seed_cells: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    checked = sum(
        int(cell.get("fairness_asserts", {}).get(key, {}).get("checked_units", 0))
        for cell in seed_cells
    )
    passed = sum(
        int(cell.get("fairness_asserts", {}).get(key, {}).get("passed_units", 0))
        for cell in seed_cells
    )
    return {
        "passed": bool(checked > 0 and checked == passed),
        "checked_units": int(checked),
        "passed_units": int(passed),
    }


def eval_idx_sha_assert(
    global_eval_record: dict[str, Any],
    config_eval_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    global_sha = global_eval_record["sha256"]
    config_sha = {
        name: record["sha256"] for name, record in config_eval_records.items()
    }
    return {
        "passed": bool(
            config_sha and all(sha == global_sha for sha in config_sha.values())
        ),
        "global_sha256": global_sha,
        "config_sha256": config_sha,
    }


def calib_idx_sha_all_models_equal_assert(
    model_cells: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    shas = [
        model_cell.get("calib_idx_sha256")
        for model_cell in model_cells.values()
        if model_cell.get("status") == "completed"
    ]
    completed_shas = {sha for sha in shas if sha is not None}
    return {
        "passed": bool(completed_shas and len(completed_shas) == 1),
        "checked_models": int(len(shas)),
        "sha256": sorted(str(sha) for sha in completed_shas),
    }


def no_future_leak_for_unit(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> bool:
    if len(train_idx) == 0 or len(test_idx) == 0:
        return False
    return bool(
        df.loc[train_idx, "timestamp"].max() < df.loc[test_idx, "timestamp"].min()
    )


def jsonable_unit(value: Any) -> str:
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


def empty_calib_idx_records() -> dict[str, list[dict[str, Any]]]:
    return {"non_fallback": [], "fallback": []}


def append_calib_idx_record(
    records: dict[str, list[dict[str, Any]]],
    seen: set[tuple[str, str]],
    *,
    cohort: str,
    unit: Any,
    source: str,
    record: dict[str, Any],
) -> None:
    key = (cohort, record["sha256"])
    if key in seen:
        return
    seen.add(key)
    records[cohort].append(
        {
            "unit": jsonable_unit(unit),
            "source": source,
            **record,
        }
    )


def aggregate_seed_values(cells: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "roc_auc",
        "pr_auc",
        "macro_roc_auc_mean",
        "macro_roc_auc_median",
        "macro_pr_auc_mean",
        "macro_pr_auc_median",
        "coverage",
        "fallback_rate",
        "fit_predict_seconds",
    ]
    out: dict[str, Any] = {"n_seeds": len(cells)}
    for metric in metrics:
        vals = [cell.get(metric) for cell in cells if cell.get(metric) is not None]
        if vals:
            out[metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        else:
            out[metric] = {"mean": None, "std": None}
    return out


def model_metric_from_predictions(
    y_eval: np.ndarray,
    eval_pred: np.ndarray,
    y_calib: np.ndarray,
    calib_pred: np.ndarray,
    *,
    per_unit_scores: list[dict[str, Any]],
    fallback_count: int,
    n_eval: int,
    fit_predict_seconds: float | None,
) -> dict[str, Any]:
    pooled = metric_summary(y_eval, eval_pred, calib_y=y_calib, calib_pred=calib_pred)
    unit_roc = [
        score["roc_auc"] for score in per_unit_scores if score["roc_auc"] is not None
    ]
    unit_pr = [
        score["pr_auc"] for score in per_unit_scores if score["pr_auc"] is not None
    ]
    covered = n_eval - fallback_count
    out = {
        "roc_auc": pooled["roc_auc"],
        "pr_auc": pooled["pr_auc"],
        "operation_points": pooled["operation_points"],
        "macro": {
            "roc_auc_mean": float(np.mean(unit_roc)) if unit_roc else None,
            "roc_auc_median": float(np.median(unit_roc)) if unit_roc else None,
            "pr_auc_mean": float(np.mean(unit_pr)) if unit_pr else None,
            "pr_auc_median": float(np.median(unit_pr)) if unit_pr else None,
            "n_scorable": int(len(per_unit_scores)),
            "coverage": float(covered / n_eval) if n_eval else 0.0,
        },
        "coverage": float(covered / n_eval) if n_eval else 0.0,
        "fallback_rate": float(fallback_count / n_eval) if n_eval else 0.0,
        "fit_predict_seconds": fit_predict_seconds,
    }
    return out


def seed_summary(
    df: pd.DataFrame,
    *,
    config: str,
    unit_index_map: dict[Any, dict[str, np.ndarray]],
    eval_idx: np.ndarray,
    selected_units: set[Any],
    c3_cache: dict[tuple[int, int], dict[str, Any]],
    runner: Runner,
    feature_cols: list[str],
    fit_rows: int,
    calib_rows: int,
    seed: int,
) -> dict[str, Any]:
    y_eval = df.loc[eval_idx, "anomaly"].to_numpy()
    model_eval_preds = {
        name: np.zeros(len(eval_idx), dtype=float) for name in MODEL_ORDER
    }
    model_calib_preds = {name: [] for name in MODEL_ORDER}
    calib_labels = {name: [] for name in MODEL_ORDER}
    per_unit_scores = {name: [] for name in MODEL_ORDER}
    fallback_counts = {name: 0 for name in MODEL_ORDER}
    fit_seconds = {name: [] for name in MODEL_ORDER}
    n_units_trained = 0
    failures: dict[str, list[dict[str, Any]]] = {name: [] for name in MODEL_ORDER}
    skipped: dict[str, list[dict[str, Any]]] = {name: [] for name in MODEL_ORDER}
    nan_records = empty_nan_records()
    calib_idx_records = empty_calib_idx_records()
    seen_calib_idx_records: set[tuple[str, str]] = set()
    calib_idx_checked_units = 0
    calib_idx_passed_units = 0
    no_future_checked_units = 0
    no_future_passed_units = 0

    eval_position_by_index = pd.Series(
        np.arange(len(eval_idx), dtype=np.int64),
        index=eval_idx,
    )
    for unit, splits in unit_index_map.items():
        unit_eval_idx = splits["eval"]
        positions = eval_position_by_index.loc[unit_eval_idx].to_numpy()
        meter = int(df.loc[unit_eval_idx[0], "meter"])
        fallback = False
        cell: dict[str, Any] | None = None
        if unit not in selected_units:
            fallback = True
        if not fallback:
            train_idx = splits["train"]
            calib_idx = splits["calib"]
            fallback = not scorable_train_index(df, train_idx) or len(calib_idx) == 0
        if not fallback:
            cell = fit_unit_models(
                runner,
                df,
                feature_cols,
                train_idx=train_idx,
                calib_idx=calib_idx,
                eval_idx=unit_eval_idx,
                fit_rows=fit_rows,
                calib_rows=calib_rows,
                seed=seed,
            )
            fallback = cell.get("status") != "completed"
        if fallback:
            cell = c3_cache[(meter, seed)]
        else:
            n_units_trained += 1
            no_future_checked_units += 1
            if no_future_leak_for_unit(df, train_idx, unit_eval_idx):
                no_future_passed_units += 1

        assert cell is not None
        cohort = "fallback" if fallback else "non_fallback"
        nan_records[cohort]["fit"].append(
            nan_fraction(df, cell["fit_idx"], feature_cols)
        )
        nan_records[cohort]["calib"].append(
            nan_fraction(df, cell["calib_idx"], feature_cols)
        )
        nan_records[cohort]["eval"].append(
            nan_fraction(df, unit_eval_idx, feature_cols)
        )
        append_calib_idx_record(
            calib_idx_records,
            seen_calib_idx_records,
            cohort=cohort,
            unit=unit,
            source="C3_fallback" if fallback else "unit_model",
            record=cell["calib_idx_record"],
        )
        unit_calib_assert = calib_idx_sha_all_models_equal_assert(cell["models"])
        if unit_calib_assert["sha256"]:
            calib_idx_checked_units += 1
            if unit_calib_assert["passed"]:
                calib_idx_passed_units += 1
        for model_name in MODEL_ORDER:
            model_cell = cell["models"].get(model_name, {})
            if model_cell.get("status") != "completed":
                record = {"unit": unit, **model_cell}
                fallback_counts[model_name] += len(positions)
                c3_model = c3_cache[(meter, seed)]["models"].get(model_name, {})
                if c3_model.get("status") != "completed":
                    if (
                        model_cell.get("status") == "skipped"
                        or c3_model.get("status") == "skipped"
                    ):
                        skipped[model_name].append(record)
                    else:
                        failures[model_name].append(record)
                    continue
                model_cell = c3_model
            if fallback:
                fallback_counts[model_name] += len(positions)
            if fallback:
                c3_eval_idx = c3_cache[(meter, seed)]["eval_idx"]
                pred_positions = pd.Index(c3_eval_idx).get_indexer(unit_eval_idx)
                if np.any(pred_positions < 0):
                    raise RuntimeError("C3 fallback eval index alignment failed")
            else:
                pred_positions = np.arange(len(positions))
            unit_eval_pred = model_cell["eval_pred"][pred_positions]
            model_eval_preds[model_name][positions] = unit_eval_pred
            model_calib_preds[model_name].append(np.asarray(model_cell["calib_pred"]))
            calib_labels[model_name].append(np.asarray(cell["y_calib"]))
            seconds = model_cell.get("fit_predict_seconds")
            if seconds is not None:
                fit_seconds[model_name].append(seconds)
            unit_y = df.loc[unit_eval_idx, "anomaly"].to_numpy()
            if len(np.unique(unit_y)) >= 2:
                per_unit_scores[model_name].append(
                    {
                        "roc_auc": float(roc_auc_score(unit_y, unit_eval_pred)),
                        "pr_auc": float(
                            average_precision_score(unit_y, unit_eval_pred)
                        ),
                    }
                )

    out: dict[str, Any] = {
        "seed": int(seed),
        "n_units_trained": int(n_units_trained),
        "per_unit_nan_fraction_summary": summarize_nan_records(nan_records),
        "calib_idx_records": calib_idx_records,
        "fairness_asserts": {
            "calib_idx_sha_all_models_equal": {
                "passed": bool(
                    calib_idx_checked_units > 0
                    and calib_idx_checked_units == calib_idx_passed_units
                ),
                "checked_units": int(calib_idx_checked_units),
                "passed_units": int(calib_idx_passed_units),
            },
            "no_future_leak_per_unit": {
                "passed": bool(
                    no_future_checked_units > 0
                    and no_future_checked_units == no_future_passed_units
                ),
                "checked_units": int(no_future_checked_units),
                "passed_units": int(no_future_passed_units),
            },
        },
        "models": {},
    }
    for model_name in MODEL_ORDER:
        if model_name == "tabpfn" and not runner.tabpfn_ok:
            out["models"][model_name] = {
                "status": "skipped",
                "reason": "tabpfn unavailable, --skip-tabpfn, or no local checkpoint",
            }
            continue
        if skipped[model_name] and not calib_labels[model_name]:
            out["models"][model_name] = {
                "status": "skipped",
                "reason": skipped[model_name][0].get("reason", "model skipped"),
            }
            continue
        if failures[model_name] and not calib_labels[model_name]:
            out["models"][model_name] = {
                "status": "failed",
                "failures": failures[model_name],
            }
            continue
        y_calib = (
            np.concatenate(calib_labels[model_name])
            if calib_labels[model_name]
            else y_eval
        )
        pred_calib = (
            np.concatenate(model_calib_preds[model_name])
            if model_calib_preds[model_name]
            else model_eval_preds[model_name]
        )
        metrics = model_metric_from_predictions(
            y_eval,
            model_eval_preds[model_name],
            y_calib,
            pred_calib,
            per_unit_scores=per_unit_scores[model_name],
            fallback_count=fallback_counts[model_name],
            n_eval=len(eval_idx),
            fit_predict_seconds=float(np.sum(fit_seconds[model_name]))
            if fit_seconds[model_name]
            else None,
        )
        metrics["status"] = "completed"
        if failures[model_name]:
            metrics["failures"] = failures[model_name]
        out["models"][model_name] = metrics
    return out


def build_c3_cache(
    df: pd.DataFrame,
    runner: Runner,
    feature_cols: list[str],
    *,
    eval_idx: np.ndarray,
    unit_index_map: dict[Any, dict[str, np.ndarray]],
    fit_rows: int,
    calib_rows: int,
    seeds: Sequence[int],
) -> dict[tuple[int, int], dict[str, Any]]:
    cache: dict[tuple[int, int], dict[str, Any]] = {}
    eval_meters = sorted(int(m) for m in pd.Series(df.loc[eval_idx, "meter"]).unique())
    for seed in seeds:
        for meter in eval_meters:
            log(f"  C3 fallback meter={meter} seed={seed}")
            splits = unit_index_map[meter]
            cache[(meter, seed)] = fit_unit_models(
                runner,
                df,
                feature_cols,
                train_idx=splits["train"],
                calib_idx=splits["calib"],
                eval_idx=splits["eval"],
                fit_rows=fit_rows,
                calib_rows=calib_rows,
                seed=seed,
            )
            if cache[(meter, seed)].get("status") != "completed":
                raise RuntimeError(f"C3 fallback failed for meter={meter} seed={seed}")
    return cache


def run_config(
    df: pd.DataFrame,
    runner: Runner,
    feature_cols: list[str],
    *,
    config: str,
    eval_idx: np.ndarray,
    c3_cache: dict[tuple[int, int], dict[str, Any]],
    fit_rows: int,
    calib_rows: int,
    seeds: Sequence[int],
    max_units: int,
    seed: int,
) -> dict[str, Any]:
    fields = CONFIGS[config]
    keys = unit_keys(df, fields)
    units_in_eval = sorted(set(keys.loc[eval_idx].to_numpy()))
    selected_units, limit_record = select_units(
        units_in_eval, max_units=max_units, seed=seed
    )
    unit_index_map = build_unit_index_map(
        df,
        keys,
        eval_idx,
        selected_units,
    )
    seed_cells = []
    for model_seed in seeds:
        log(f"Config {config} seed={model_seed}")
        seed_cells.append(
            seed_summary(
                df,
                config=config,
                unit_index_map=unit_index_map,
                eval_idx=eval_idx,
                selected_units=selected_units,
                c3_cache=c3_cache,
                runner=runner,
                feature_cols=feature_cols,
                fit_rows=fit_rows,
                calib_rows=calib_rows,
                seed=model_seed,
            )
        )
    models: dict[str, Any] = {}
    for model_name in MODEL_ORDER:
        completed = [
            {
                "roc_auc": cell["models"][model_name].get("roc_auc"),
                "pr_auc": cell["models"][model_name].get("pr_auc"),
                "macro_roc_auc_mean": cell["models"][model_name]
                .get("macro", {})
                .get("roc_auc_mean"),
                "macro_roc_auc_median": cell["models"][model_name]
                .get("macro", {})
                .get("roc_auc_median"),
                "macro_pr_auc_mean": cell["models"][model_name]
                .get("macro", {})
                .get("pr_auc_mean"),
                "macro_pr_auc_median": cell["models"][model_name]
                .get("macro", {})
                .get("pr_auc_median"),
                "coverage": cell["models"][model_name].get("coverage"),
                "fallback_rate": cell["models"][model_name].get("fallback_rate"),
                "fit_predict_seconds": cell["models"][model_name].get(
                    "fit_predict_seconds"
                ),
            }
            for cell in seed_cells
            if cell["models"][model_name].get("status") == "completed"
        ]
        models[model_name] = {
            "status": "completed" if completed else "failed",
            "by_seed": [cell["models"][model_name] for cell in seed_cells],
            "mean_std": aggregate_seed_values(completed),
        }
    return {
        "unit_key": CONFIG_LABELS[config],
        "unit_fields": list(fields),
        "eval_idx_fingerprint": index_record(eval_idx),
        "selected_units": limit_record,
        "n_units_trained_by_seed": [
            int(cell["n_units_trained"]) for cell in seed_cells
        ],
        "nan_fraction": nan_fraction(df, eval_idx, feature_cols),
        "per_unit_nan_fraction_summary": aggregate_nan_summaries(seed_cells),
        "calib_idx_records_by_seed": [
            {
                "seed": int(cell["seed"]),
                "records": cell["calib_idx_records"],
            }
            for cell in seed_cells
        ],
        "fairness_asserts": {
            "calib_idx_sha_all_models_equal_within_scope": aggregate_count_assert(
                seed_cells,
                "calib_idx_sha_all_models_equal",
            ),
            "no_future_leak_per_unit": aggregate_count_assert(
                seed_cells,
                "no_future_leak_per_unit",
            ),
        },
        "unit_row_summary": summarize_unit_rows(unit_index_map, selected_units),
        "models": models,
    }


def load_reference() -> dict[str, Any]:
    path = PROC / "m6_phaseD_50_50_full_models_timestamp_merge.json"
    if not path.exists():
        return {"status": "missing", "path": str(path.relative_to(ROOT))}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    models = payload.get("axes", {}).get("in_domain", {}).get("models", {})
    ref_models = {}
    for name in MODEL_ORDER:
        test = models.get(name, {}).get("test", {})
        ref_models[name] = {
            "roc_auc": test.get("roc_auc"),
            "pr_auc": test.get("pr_auc"),
        }
    return {
        "status": "completed",
        "path": str(path.relative_to(ROOT)),
        "regime": "timestamp_merge",
        "protocol": "pooled_building_heldout_offline137",
        "models": ref_models,
    }


def command_record(args: argparse.Namespace) -> str:
    parts = [
        "uv run python scripts/run_m5x_partition_granularity.py",
        f"--out {args.out}",
        f"--configs {' '.join(args.configs)}",
        f"--fit-rows {args.fit_rows}",
        f"--eval-rows {args.eval_rows}",
        f"--calib-rows {args.calib_rows}",
        f"--seeds {' '.join(map(str, args.seeds))}",
        f"--seed {args.seed}",
        f"--max-units {args.max_units}",
    ]
    if args.skip_tabpfn:
        parts.append("--skip-tabpfn")
    if args.smoke:
        parts.append("--smoke")
    if args.model_path:
        parts.append(f"--model-path {args.model_path}")
    return " ".join(parts)


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    stage_timings: dict[str, Any] = {}
    env = torch_environment()
    env.update(
        {
            "tabpfn_model_path": str(args.model_path) if args.model_path else None,
            "tabpfn_model_path_exists": bool(
                args.model_path and args.model_path.is_file()
            ),
        }
    )
    runner = Runner(args, env)
    log(
        f"Device={runner.device} tabpfn_ok={runner.tabpfn_ok} "
        f"fit_rows={args.fit_rows} eval_rows={args.eval_rows} seeds={args.seeds}"
    )
    stage_t0 = time.perf_counter()
    df = load_m3_frame(verbose=True)
    stage_timings["load_m3_frame_seconds"] = float(time.perf_counter() - stage_t0)
    stage_t0 = time.perf_counter()
    df, feature_cols = build_causal_frame(df)
    stage_timings["build_causal_frame_seconds"] = float(time.perf_counter() - stage_t0)
    stage_t0 = time.perf_counter()
    df["window"] = assign_windows(df)
    eval_idx = eval_indices(df, eval_rows=args.eval_rows, seed=args.seed)
    stage_timings["assign_windows_and_eval_idx_seconds"] = float(
        time.perf_counter() - stage_t0
    )
    c3_keys = unit_keys(df, CONFIGS["C3"])
    c3_units = set(c3_keys.loc[eval_idx].to_numpy())
    stage_t0 = time.perf_counter()
    c3_unit_index_map = build_unit_index_map(
        df,
        c3_keys,
        eval_idx,
        c3_units,
    )
    stage_timings["c3_unit_index_map_seconds"] = float(time.perf_counter() - stage_t0)
    stage_t0 = time.perf_counter()
    c3_cache = build_c3_cache(
        df,
        runner,
        feature_cols,
        eval_idx=eval_idx,
        unit_index_map=c3_unit_index_map,
        fit_rows=args.fit_rows,
        calib_rows=args.calib_rows,
        seeds=args.seeds,
    )
    stage_timings["c3_fallback_cache_seconds"] = float(time.perf_counter() - stage_t0)
    configs = {}
    stage_timings["configs_seconds"] = {}
    for config in args.configs:
        stage_t0 = time.perf_counter()
        configs[config] = run_config(
            df,
            runner,
            feature_cols,
            config=config,
            eval_idx=eval_idx,
            c3_cache=c3_cache,
            fit_rows=args.fit_rows,
            calib_rows=args.calib_rows,
            seeds=args.seeds,
            max_units=args.max_units,
            seed=args.seed,
        )
        stage_timings["configs_seconds"][config] = float(time.perf_counter() - stage_t0)
    global_eval_record = index_record(eval_idx)
    config_eval_records = {
        name: cfg["eval_idx_fingerprint"] for name, cfg in configs.items()
    }
    eval_assert = eval_idx_sha_assert(global_eval_record, config_eval_records)
    calib_assert_checked = sum(
        int(
            cfg.get("fairness_asserts", {})
            .get("calib_idx_sha_all_models_equal_within_scope", {})
            .get("checked_units", 0)
        )
        for cfg in configs.values()
    )
    calib_assert_passed = sum(
        int(
            cfg.get("fairness_asserts", {})
            .get("calib_idx_sha_all_models_equal_within_scope", {})
            .get("passed_units", 0)
        )
        for cfg in configs.values()
    )
    future_assert_checked = sum(
        int(
            cfg.get("fairness_asserts", {})
            .get("no_future_leak_per_unit", {})
            .get("checked_units", 0)
        )
        for cfg in configs.values()
    )
    future_assert_passed = sum(
        int(
            cfg.get("fairness_asserts", {})
            .get("no_future_leak_per_unit", {})
            .get("passed_units", 0)
        )
        for cfg in configs.values()
    )
    results = {
        "experiment": EXPERIMENT,
        "value_change_regime": REGIME_LABEL,
        "time_split": TIME_SPLIT,
        "eval_idx_fingerprint": global_eval_record,
        "global_nan_fraction": nan_fraction(df, eval_idx, feature_cols),
        "budgets": {
            "fit_rows": int(args.fit_rows),
            "eval_rows": int(args.eval_rows),
            "calib_rows": int(args.calib_rows),
            "seeds": [int(s) for s in args.seeds],
            "eval_seed": int(args.seed),
            "max_units": int(args.max_units),
            "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
        },
        "environment": env,
        "reference": {"C0_anchor": load_reference()},
        "configs": configs,
        "fairness_asserts": {
            "eval_idx_sha_all_equal": {
                **eval_assert,
            },
            "calib_idx_sha_all_models_equal_within_scope": {
                "passed": bool(
                    calib_assert_checked > 0
                    and calib_assert_checked == calib_assert_passed
                ),
                "checked_units": int(calib_assert_checked),
                "passed_units": int(calib_assert_passed),
            },
            "feature_count": len(feature_cols),
            "no_future_leak": {
                "passed": bool(
                    future_assert_checked > 0
                    and future_assert_checked == future_assert_passed
                ),
                "checked_units": int(future_assert_checked),
                "passed_units": int(future_assert_passed),
            },
            "no_future_leak_global": bool(
                df.loc[df["window"].eq("train"), "timestamp"].max()
                < df.loc[df["window"].eq("test"), "timestamp"].min()
            ),
        },
        "one_shot_discipline": {
            "leaderboard_probing": False,
            "cloud_client_used": False,
            "bdg2_used": False,
        },
        "stage_timings": stage_timings,
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    return results


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results = run(args)
    write_json_with_provenance(
        args.out,
        results,
        root=ROOT,
        provenance={"command": command_record(args)},
    )
    log(f"Saved {args.out} in {results['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
