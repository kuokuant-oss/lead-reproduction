"""M6 Phase D 50/50 model matrix with train/test confusion summaries."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from lead import (
    BASELINE_FEATURE_COLS,
    DOWNSAMPLE_SEEDS,
    PROC,
    RANDOM_STATE,
    ROOT,
    SHIFTS,
    add_value_change_features,
    assert_no_building_overlap,
    downsample_indices,
    load_m3_frame,
    write_json_with_provenance,
)


VALUE_CHANGE_REGIME = "row_offset_meter_aware"
SPLIT_NAME = "50_50_mod2"
AXES = ("in_domain", "site_transfer", "label_scarcity", "minimal_fe")
MODEL_ORDER = (
    "lightgbm",
    "xgboost",
    "catboost",
    "hist_gradient_boosting",
    "ensemble",
    "tabpfn",
)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    default_model_path = (
        Path(os.environ["TABPFN_MODEL_CACHE_DIR"])
        / "tabpfn-v3-classifier-v3_default.ckpt"
        if os.environ.get("TABPFN_MODEL_CACHE_DIR")
        else ROOT / ".tabpfn-cache" / "tabpfn-v3-classifier-v3_default.ckpt"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "m6_phaseD_50_50_full_models.json",
    )
    parser.add_argument("--fit-rows", type=int, default=10_000)
    parser.add_argument("--score-rows", type=int, default=4_000)
    parser.add_argument(
        "--scarcity-sizes",
        type=int,
        nargs="+",
        default=[200, 500, 1_000, 2_000, 5_000, 10_000],
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--skip-tabpfn", action="store_true")
    parser.add_argument("--model-path", type=Path, default=default_model_path)
    return parser.parse_args()


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


def random_indices(index: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or len(index) <= max_rows:
        return index
    rng = np.random.RandomState(seed)
    return np.asarray(rng.choice(index, size=max_rows, replace=False))


def balanced_subsample_indices(
    ds_idx: np.ndarray, y, max_rows: int, seed: int
) -> np.ndarray:
    if max_rows <= 0 or len(ds_idx) <= max_rows:
        return ds_idx
    rng = np.random.RandomState(seed)
    y_ds = y.loc[ds_idx].to_numpy()
    pos_positions = np.flatnonzero(y_ds == 1)
    neg_positions = np.flatnonzero(y_ds == 0)
    per_class = max_rows // 2
    n_pos = min(per_class, len(pos_positions))
    n_neg = min(max_rows - n_pos, len(neg_positions))
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
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "total": total,
            "tn_pct_total": float(tn / total) if total else 0.0,
            "fp_pct_total": float(fp / total) if total else 0.0,
            "fn_pct_total": float(fn / total) if total else 0.0,
            "tp_pct_total": float(tp / total) if total else 0.0,
            "anomalies_total": anomalies,
            "normal_total": normal,
            "anomalies_captured": int(tp),
            "anomalies_missed": int(fn),
            "anomaly_capture_rate": float(tp / anomalies) if anomalies else 0.0,
            "anomaly_miss_rate": float(fn / anomalies) if anomalies else 0.0,
            "normal_correct": int(tn),
            "false_alarms": int(fp),
            "false_alarm_rate": float(fp / normal) if normal else 0.0,
            "summary": (
                f"Captured {int(tp)}/{anomalies} anomalies "
                f"({(tp / anomalies * 100) if anomalies else 0.0:.1f}%) and "
                f"missed {int(fn)}/{anomalies} "
                f"({(fn / anomalies * 100) if anomalies else 0.0:.1f}%). "
                f"False alarms: {int(fp)}/{normal} normal rows "
                f"({(fp / normal * 100) if normal else 0.0:.1f}%)."
            ),
        },
    }


def metric_summary(y_true, pred: np.ndarray) -> dict[str, Any]:
    recall_threshold = threshold_for_recall(y_true, pred, 0.90)
    return {
        "roc_auc": float(roc_auc_score(y_true, pred)),
        "pr_auc": float(average_precision_score(y_true, pred)),
        "operation_points": {
            "threshold_0_5": binary_summary(y_true, pred, threshold=0.5),
            "fixed_recall_0_90": binary_summary(
                y_true,
                pred,
                threshold=recall_threshold,
            ),
        },
    }


def tree_models(seed: int) -> dict[str, Any]:
    return {
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=100,
            verbose=-1,
            random_state=seed,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=100,
            eval_metric="logloss",
            verbosity=0,
            random_state=seed,
        ),
        "catboost": CatBoostClassifier(
            iterations=1000,
            verbose=False,
            random_seed=seed,
            allow_writing_files=False,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            random_state=seed,
        ),
    }


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

    def fit_tabpfn(self, x_train, y_train):
        if not self.tabpfn_ok:
            return None
        model = tabpfn_classifier(
            self.device,
            self.model_path if self.local_ckpt else None,
        )
        model.fit(x_train, y_train)
        return model


def build_50_50_table(df, *, site_split: bool = False) -> dict[str, Any]:
    if site_split:
        mask_test = (df["site_id"] % 2 == 1).to_numpy()
        split_name = "site_id_mod2_50_50"
        train_units = set(df.loc[~mask_test, "site_id"].unique())
        test_units = set(df.loc[mask_test, "site_id"].unique())
    else:
        mask_test = (df["building_id"] % 2 == 1).to_numpy()
        split_name = SPLIT_NAME
        train_units = set(df.loc[~mask_test, "building_id"].unique())
        test_units = set(df.loc[mask_test, "building_id"].unique())
    overlap = assert_no_building_overlap(
        train_units,
        test_units,
        split_name=split_name,
    )
    train_full = add_value_change_features(
        df.loc[~mask_test],
        list(SHIFTS),
        value_change_regime=VALUE_CHANGE_REGIME,
    )
    test_full = add_value_change_features(
        df.loc[mask_test],
        list(SHIFTS),
        value_change_regime=VALUE_CHANGE_REGIME,
    )
    value_cols = [c for c in train_full.columns if c.startswith("lag_value_")]
    if len(value_cols) != 120:
        raise AssertionError(
            f"Expected 120 value-change features, got {len(value_cols)}"
        )
    return {
        "train_full": train_full,
        "test_full": test_full,
        "value_cols": value_cols,
        "ds_idx_full": downsample_indices(train_full["anomaly"]),
        "split": {
            "name": split_name,
            "value_change_regime": VALUE_CHANGE_REGIME,
            "n_train_units": int(len(train_units)),
            "n_test_units": int(len(test_units)),
            "unit_overlap": int(len(overlap)),
            "n_train_rows": int((~mask_test).sum()),
            "n_test_rows": int(mask_test.sum()),
            "train_anomaly_rate": float(df.loc[~mask_test, "anomaly"].mean()),
            "test_anomaly_rate": float(df.loc[mask_test, "anomaly"].mean()),
            "unit_type": "site_id" if site_split else "building_id",
        },
    }


def make_matrices(
    table: dict[str, Any],
    feature_cols: list[str],
    fit_idx: np.ndarray,
    train_score_idx: np.ndarray,
    test_score_idx: np.ndarray,
) -> dict[str, Any]:
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(table["train_full"].loc[fit_idx, feature_cols])
    x_train_score = scaler.transform(
        table["train_full"].loc[train_score_idx, feature_cols]
    )
    x_test_score = scaler.transform(
        table["test_full"].loc[test_score_idx, feature_cols]
    )
    return {
        "x_fit": x_fit,
        "y_fit": table["train_full"].loc[fit_idx, "anomaly"],
        "x_fit_score": x_fit,
        "y_fit_score": table["train_full"].loc[fit_idx, "anomaly"],
        "x_train_score": x_train_score,
        "y_train_score": table["train_full"].loc[train_score_idx, "anomaly"],
        "x_test_score": x_test_score,
        "y_test_score": table["test_full"].loc[test_score_idx, "anomaly"],
    }


def fit_score_all_models(
    runner: Runner,
    matrices: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tree_preds_fit = []
    tree_preds_train = []
    tree_preds_test = []
    models = tree_models(seed)
    for name, model in models.items():
        log(f"    fitting {name}")
        x_fit = matrices["x_fit"]
        x_fit_score = matrices["x_fit_score"]
        x_train_score = matrices["x_train_score"]
        x_test_score = matrices["x_test_score"]
        if name == "hist_gradient_boosting":
            x_fit = np.nan_to_num(x_fit, nan=0)
            x_fit_score = np.nan_to_num(x_fit_score, nan=0)
            x_train_score = np.nan_to_num(x_train_score, nan=0)
            x_test_score = np.nan_to_num(x_test_score, nan=0)
        t0 = time.perf_counter()
        model.fit(x_fit, matrices["y_fit"])
        pred_fit = model.predict_proba(x_fit_score)[:, 1]
        pred_train = model.predict_proba(x_train_score)[:, 1]
        pred_test = model.predict_proba(x_test_score)[:, 1]
        elapsed = time.perf_counter() - t0
        if name != "hist_gradient_boosting":
            tree_preds_fit.append(pred_fit)
        tree_preds_train.append(pred_train)
        tree_preds_test.append(pred_test)
        out[name] = {
            "fit_predict_seconds": float(elapsed),
            "fit_set": metric_summary(matrices["y_fit_score"], pred_fit),
            "train": metric_summary(matrices["y_train_score"], pred_train),
            "test": metric_summary(matrices["y_test_score"], pred_test),
        }
        if name == "hist_gradient_boosting":
            tree_preds_fit.append(pred_fit)
    ensemble_fit = sum(tree_preds_fit) / len(tree_preds_fit)
    ensemble_train = sum(tree_preds_train) / len(tree_preds_train)
    ensemble_test = sum(tree_preds_test) / len(tree_preds_test)
    out["ensemble"] = {
        "fit_predict_seconds": None,
        "fit_set": metric_summary(matrices["y_fit_score"], ensemble_fit),
        "train": metric_summary(matrices["y_train_score"], ensemble_train),
        "test": metric_summary(matrices["y_test_score"], ensemble_test),
    }
    log("    fitting tabpfn")
    if runner.tabpfn_ok:
        t0 = time.perf_counter()
        try:
            model = runner.fit_tabpfn(matrices["x_fit"], matrices["y_fit"])
            pred_fit = model.predict_proba(matrices["x_fit_score"])[:, 1]
            pred_train = model.predict_proba(matrices["x_train_score"])[:, 1]
            pred_test = model.predict_proba(matrices["x_test_score"])[:, 1]
            out["tabpfn"] = {
                "status": "completed",
                "fit_predict_seconds": float(time.perf_counter() - t0),
                "fit_set": metric_summary(matrices["y_fit_score"], pred_fit),
                "train": metric_summary(matrices["y_train_score"], pred_train),
                "test": metric_summary(matrices["y_test_score"], pred_test),
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


def run_cell(
    runner: Runner,
    table: dict[str, Any],
    feature_cols: list[str],
    *,
    fit_rows: int,
    score_rows: int,
    seed: int,
) -> dict[str, Any]:
    fit_idx = balanced_subsample_indices(
        table["ds_idx_full"],
        table["train_full"]["anomaly"],
        fit_rows,
        seed,
    )
    train_score_idx = random_indices(
        table["train_full"].index.to_numpy(),
        score_rows,
        seed + 10_000,
    )
    test_score_idx = random_indices(
        table["test_full"].index.to_numpy(),
        score_rows,
        seed + 20_000,
    )
    matrices = make_matrices(
        table,
        feature_cols,
        fit_idx,
        train_score_idx,
        test_score_idx,
    )
    return {
        "fit_rows": int(len(fit_idx)),
        "train_score_rows": int(len(train_score_idx)),
        "test_score_rows": int(len(test_score_idx)),
        "row_index_records": {
            "fit_idx": index_record(fit_idx),
            "train_score_idx": index_record(train_score_idx),
            "test_score_idx": index_record(test_score_idx),
        },
        "row_prevalence": {
            "fit_set": float(table["train_full"].loc[fit_idx, "anomaly"].mean()),
            "train_score": float(
                table["train_full"].loc[train_score_idx, "anomaly"].mean()
            ),
            "test_score": float(
                table["test_full"].loc[test_score_idx, "anomaly"].mean()
            ),
        },
        "models": fit_score_all_models(runner, matrices, seed=seed),
    }


def feature_sets(table: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "full_137": BASELINE_FEATURE_COLS + table["value_cols"],
        "raw_baseline_17": list(BASELINE_FEATURE_COLS),
    }


def run_all_axes(args: argparse.Namespace, runner: Runner, df) -> dict[str, Any]:
    axes: dict[str, Any] = {}
    table = build_50_50_table(df)
    features = feature_sets(table)

    log("Axis: in_domain")
    axes["in_domain"] = {
        "description": "50/50 building train/test split, full 137 features.",
        "split": table["split"],
        **run_cell(
            runner,
            table,
            features["full_137"],
            fit_rows=args.fit_rows,
            score_rows=args.score_rows,
            seed=args.seed,
        ),
    }

    log("Axis: label_scarcity")
    scarcity = []
    for size in args.scarcity_sizes:
        log(f"  support={size}")
        scarcity.append(
            {
                "support_size": int(size),
                **run_cell(
                    runner,
                    table,
                    features["full_137"],
                    fit_rows=size,
                    score_rows=args.score_rows,
                    seed=args.seed,
                ),
            }
        )
    axes["label_scarcity"] = {
        "description": "50/50 building train/test split, full 137 features.",
        "split": table["split"],
        "sizes": scarcity,
    }

    log("Axis: minimal_fe")
    minimal = []
    for name, cols in features.items():
        log(f"  feature_set={name}")
        minimal.append(
            {
                "feature_set": name,
                "n_features": int(len(cols)),
                **run_cell(
                    runner,
                    table,
                    cols,
                    fit_rows=args.fit_rows,
                    score_rows=args.score_rows,
                    seed=args.seed,
                ),
            }
        )
    axes["minimal_fe"] = {
        "description": "50/50 building train/test split, raw baseline vs full features.",
        "split": table["split"],
        "feature_sets": minimal,
    }

    del table
    log("Axis: site_transfer")
    site_table = build_50_50_table(df, site_split=True)
    axes["site_transfer"] = {
        "description": "50/50 site train/test split, full 137 features.",
        "split": site_table["split"],
        **run_cell(
            runner,
            site_table,
            feature_sets(site_table)["full_137"],
            fit_rows=args.fit_rows,
            score_rows=args.score_rows,
            seed=args.seed,
        ),
    }
    return axes


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
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
        f"fit_rows={args.fit_rows} score_rows={args.score_rows} seed={args.seed}"
    )
    df = load_m3_frame(verbose=True)
    axes = run_all_axes(args, runner, df)
    results = {
        "experiment": "m6_phaseD_50_50_full_models",
        "issue": 52,
        "scope": (
            "50/50 train/test split run with all tree models, ensemble, TabPFN, "
            "train/test scoring subsamples, and confusion matrix summaries."
        ),
        "value_change_regime": VALUE_CHANGE_REGIME,
        "budgets": {
            "fit_rows": int(args.fit_rows),
            "score_rows": int(args.score_rows),
            "scarcity_sizes": [int(s) for s in args.scarcity_sizes],
            "seed": int(args.seed),
            "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
            "scoring_note": (
                "The train/test split is 50/50 by held-out unit. Metrics and "
                "confusion matrices are computed on natural-prevalence scoring "
                "subsamples so TabPFN and tree models score the same rows."
            ),
        },
        "models": list(MODEL_ORDER),
        "environment": env,
        "axes": axes,
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    command = (
        "uv run python scripts/run_m6_phaseD_50_50_full_models.py "
        f"--out {args.out} --fit-rows {args.fit_rows} "
        f"--score-rows {args.score_rows} "
        f"--scarcity-sizes {' '.join(map(str, args.scarcity_sizes))} "
        f"--seed {args.seed}"
    )
    write_json_with_provenance(
        args.out,
        results,
        root=ROOT,
        provenance={"command": command},
    )
    log(f"Saved {args.out} in {results['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
