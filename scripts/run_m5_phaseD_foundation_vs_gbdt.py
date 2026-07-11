"""M5 Phase D: rigorous TabPFN vs GBDT comparison on the existing M3 GEPIII data.

This harness runs paired, multi-seed comparisons through the frozen ``src/lead``
pipeline. In the default single-``--val-seed`` path, every paired cell reuses the
same split, downsample, feature table, and fixed validation subsample so the only
variable is the model. The INV-2 ``--val-seeds`` path repeats that paired design
across validation resamples. No new dataset, no BDG2, no cloud: TabPFN runs from
local weights only.

Axes:
  1. in_domain      - TabPFN vs GBDT on the 80/20 (``building_id % 5 == 4``) split.
  2. site_transfer  - PRIMARY. Site-held-out (``site_id % 5 == 4``) split. TabPFN
                      in-context vs GBDT-retrain vs GBDT-transfer-without-retrain.
                      M3 ensemble anchor AUC 0.9774.
  3. label_scarcity - shrink the labeled support set across sizes; show degradation.
  4. minimal_fe     - TabPFN/GBDT on a reduced raw feature set vs the 137-feature
                      line; quantify the feature-engineering-burden difference.

Metrics per cell: ROC-AUC, PR-AUC (average precision), precision/recall/F1 at the
0.5 threshold, and fit+predict latency. Multiple model seeds, and optionally
multiple validation seeds, are aggregated to mean +/- std.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import time
import traceback
from importlib import metadata
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from lead import (
    BASELINE_FEATURE_COLS,
    DOWNSAMPLE_SEEDS,
    MODEL_SEEDS,
    PROC,
    RANDOM_STATE,
    ROOT,
    SHIFTS,
    add_value_change_features,
    assert_no_building_overlap,
    classification_metrics,
    downsample_indices,
    leave_site_out_mask,
    load_m3_frame,
    write_json_with_provenance,
)
from experiment_observability import host_environment, timing_protocol

DEFAULT_VALUE_CHANGE_REGIME = "timestamp_merge"
IN_DOMAIN_SPLIT = "80_20_mod5"  # building_id % 5 == 4
SITE_TRANSFER_RULE = "site_id % 5 == 4"
SITE_ANCHOR_ENSEMBLE_AUC = 0.9774  # M3 site-held-out ensemble diagnostic.
TABPFN3_LIMITS = [
    {"max_rows": 1_000_000, "max_features": 200},
    {"max_rows": 100_000, "max_features": 2_000},
    {"max_rows": 1_000, "max_features": 20_000},
]
# Metric keys aggregated across seeds (mean +/- std).
METRIC_KEYS = (
    "val_auc",
    "pr_auc",
    "precision_05",
    "recall_05",
    "f1_05",
    "fit_predict_seconds",
)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_model_path = (
        Path(os.environ["TABPFN_MODEL_CACHE_DIR"])
        / "tabpfn-v3-classifier-v3_default.ckpt"
        if os.environ.get("TABPFN_MODEL_CACHE_DIR")
        else ROOT / ".tabpfn-cache" / "tabpfn-v3-classifier-v3_default.ckpt"
    )
    parser.add_argument(
        "--out", type=Path, default=PROC / "m5_phaseD_foundation_vs_gbdt.json"
    )
    parser.add_argument(
        "--tabpfn-fit-rows",
        type=int,
        default=10_000,
        help=(
            "Balanced TabPFN/GBDT fit-set budget for the in-domain, site-transfer, "
            "and minimal-FE axes. Bounded by laptop VRAM, well under the documented "
            "TabPFN-3 1,000,000 x 200 limit."
        ),
    )
    parser.add_argument(
        "--val-rows",
        type=int,
        default=4_000,
        help="Fixed validation subsample scored per axis (natural anomaly rate).",
    )
    parser.add_argument(
        "--scarcity-sizes",
        type=int,
        nargs="+",
        default=[200, 500, 1_000, 2_000, 5_000, 10_000],
        help="Balanced support-set sizes for the label-scarcity axis.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(MODEL_SEEDS),
        help="Seeds for fit-subsample selection and model random_state.",
    )
    parser.add_argument(
        "--val-seed",
        type=int,
        default=RANDOM_STATE,
        help="Seed for the fixed validation subsample (held constant per axis).",
    )
    parser.add_argument(
        "--val-seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional INV-2 validation resampling seeds. When set, each axis is "
            "rerun for every validation seed and summarized across validation "
            "resamples. The legacy --val-seed path remains the single-seed default."
        ),
    )
    parser.add_argument(
        "--value-change-regime",
        choices=["row_offset", "row_offset_meter_aware", "timestamp_merge"],
        default=DEFAULT_VALUE_CHANGE_REGIME,
        help=(
            "Value-change feature regime. Default uses timestamp-aligned value "
            "changes for the current M5/M6 comparison matrix."
        ),
    )
    parser.add_argument(
        "--tabpfn-batch-size",
        type=int,
        default=256,
        help="Recorded TabPFN prediction batch size.",
    )
    parser.add_argument(
        "--axes",
        type=str,
        nargs="+",
        default=["in_domain", "site_transfer", "label_scarcity", "minimal_fe"],
        help="Subset of axes to run.",
    )
    parser.add_argument(
        "--skip-tabpfn",
        action="store_true",
        help="Run only the GBDT side (feature/split plumbing check).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny budgets for a fast logic check (not a real result).",
    )
    parser.add_argument("--model-path", type=Path, default=default_model_path)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# TabPFN-3 documented-limit bookkeeping
# --------------------------------------------------------------------------- #
def tabpfn_limit_fit(n_rows: int, n_features: int) -> dict[str, Any]:
    fits = [
        limit
        for limit in TABPFN3_LIMITS
        if n_rows <= limit["max_rows"] and n_features <= limit["max_features"]
    ]
    return {
        "fits_documented_tabpfn3_limit": bool(fits),
        "matching_limits": fits,
        "documented_limits": TABPFN3_LIMITS,
    }


# --------------------------------------------------------------------------- #
# Subsampling
# --------------------------------------------------------------------------- #
def balanced_subsample_indices(
    ds_idx: np.ndarray, y, max_rows: int, seed: int
) -> np.ndarray:
    """Return up to ``max_rows`` row labels balanced 50/50 by class."""
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


def random_val_indices(index: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    """Fixed natural-prevalence validation subsample."""
    if max_rows <= 0 or len(index) <= max_rows:
        return index
    rng = np.random.RandomState(seed)
    return np.asarray(rng.choice(index, size=max_rows, replace=False))


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def torch_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        **host_environment(),
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "tabpfn_installed": importlib.util.find_spec("tabpfn") is not None,
        "tabpfn_token_present": bool(os.environ.get("TABPFN_TOKEN")),
        "execution_path": "local",
        "cloud_client_used": False,
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


# --------------------------------------------------------------------------- #
# Metrics + models
# --------------------------------------------------------------------------- #
def cell_metrics(y_true, pred: np.ndarray) -> dict[str, float]:
    return {
        **classification_metrics(y_true, pred),
        "pr_auc": float(average_precision_score(y_true, pred)),
    }


def fit_gbdt(x_train, y_train, x_val, y_val, *, seed: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    model = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=seed)
    model.fit(x_train, y_train)
    pred = model.predict_proba(x_val)[:, 1]
    elapsed = time.perf_counter() - t0
    return {
        "model": model,
        "metrics": {
            **cell_metrics(y_val, pred),
            "fit_predict_seconds": float(elapsed),
        },
    }


def score_gbdt(model, x_val, y_val) -> dict[str, float]:
    t0 = time.perf_counter()
    pred = model.predict_proba(x_val)[:, 1]
    elapsed = time.perf_counter() - t0
    return {**cell_metrics(y_val, pred), "fit_predict_seconds": float(elapsed)}


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


def fit_tabpfn(
    x_train, y_train, x_val, y_val, *, device: str, model_path: Path | None
) -> dict[str, float]:
    t0 = time.perf_counter()
    model = tabpfn_classifier(device, model_path)
    init_elapsed = time.perf_counter() - t0
    t_fit = time.perf_counter()
    model.fit(x_train, y_train)
    fit_elapsed = time.perf_counter() - t_fit
    t_pred = time.perf_counter()
    pred = model.predict_proba(x_val)[:, 1]
    predict_elapsed = time.perf_counter() - t_pred
    elapsed = time.perf_counter() - t0
    return {
        **cell_metrics(y_val, pred),
        "model_init_seconds": float(init_elapsed),
        "fit_seconds": float(fit_elapsed),
        "predict_proba_seconds": float(predict_elapsed),
        "fit_predict_seconds": float(elapsed),
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed metric dicts into mean/std plus the raw runs."""
    ok = [c for c in cells if c.get("status", "completed") == "completed"]
    summary: dict[str, Any] = {
        "n_runs": len(cells),
        "n_completed": len(ok),
        "raw": cells,
    }
    if not ok:
        return summary
    extra = [
        k
        for k in ("model_init_seconds", "fit_seconds", "predict_proba_seconds")
        if all(k in c for c in ok)
    ]
    keys = list(METRIC_KEYS) + extra
    summary["mean"] = {k: float(mean([c[k] for c in ok])) for k in keys}
    summary["std"] = {k: float(pstdev([c[k] for c in ok])) for k in keys}
    return summary


def completed_raw(aggregate_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        c
        for c in aggregate_result.get("raw", [])
        if c.get("status", "completed") == "completed"
    ]


def aggregate_named_models(
    per_val_seed_axes: dict[str, dict[str, Any]],
    axis_name: str,
    model_names: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model_name in model_names:
        cells: list[dict[str, Any]] = []
        for axis in per_val_seed_axes.values():
            if axis_name in axis and model_name in axis[axis_name]:
                cells.extend(completed_raw(axis[axis_name][model_name]))
        out[model_name] = aggregate(cells)
    return out


def paired_delta_summary(
    left_model: dict[str, Any],
    right_model: dict[str, Any],
    *,
    metric: str,
    label: str,
) -> dict[str, Any]:
    left = {
        (c.get("val_seed"), c.get("seed")): c
        for c in completed_raw(left_model)
        if metric in c
    }
    right = {
        (c.get("val_seed"), c.get("seed")): c
        for c in completed_raw(right_model)
        if metric in c
    }
    pairs = []
    for key in sorted(left.keys() & right.keys()):
        val_seed, model_seed = key
        delta = float(left[key][metric] - right[key][metric])
        pairs.append(
            {
                "val_seed": int(val_seed),
                "model_seed": int(model_seed),
                "delta": delta,
                "left": float(left[key][metric]),
                "right": float(right[key][metric]),
            }
        )
    deltas = [p["delta"] for p in pairs]
    return {
        "label": label,
        "metric": metric,
        "n_pairs": len(pairs),
        "mean_delta": float(mean(deltas)) if deltas else None,
        "std_delta": float(pstdev(deltas))
        if len(deltas) > 1
        else 0.0
        if deltas
        else None,
        "min_delta": float(min(deltas)) if deltas else None,
        "max_delta": float(max(deltas)) if deltas else None,
        "pairs": pairs,
    }


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int = RANDOM_STATE,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return {"n": 1, "mean": float(arr[0]), "ci95": [float(arr[0]), float(arr[0])]}
    rng = np.random.RandomState(seed)
    means = [
        float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(n_bootstrap)
    ]
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "ci95": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
    }


def add_val_seed_to_axis(axis: dict[str, Any], val_seed: int) -> dict[str, Any]:
    for value in axis.values():
        if isinstance(value, dict) and "raw" in value:
            for cell in value["raw"]:
                cell["val_seed"] = int(val_seed)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for nested in item.values():
                        if isinstance(nested, dict) and "raw" in nested:
                            for cell in nested["raw"]:
                                cell["val_seed"] = int(val_seed)
    return axis


def summarize_val_resampling(
    per_val_seed_axes: dict[str, dict[str, Any]],
    axes_requested: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if "in_domain" in axes_requested:
        models = aggregate_named_models(
            per_val_seed_axes, "in_domain", ["gbdt", "tabpfn"]
        )
        summary["in_domain"] = {
            **models,
            "paired_deltas": {
                "tabpfn_minus_gbdt_roc_auc": paired_delta_summary(
                    models["tabpfn"],
                    models["gbdt"],
                    metric="val_auc",
                    label="TabPFN - GBDT on paired validation rows",
                ),
                "tabpfn_minus_gbdt_pr_auc": paired_delta_summary(
                    models["tabpfn"],
                    models["gbdt"],
                    metric="pr_auc",
                    label="TabPFN - GBDT on paired validation rows",
                ),
            },
        }
    if "site_transfer" in axes_requested:
        models = aggregate_named_models(
            per_val_seed_axes,
            "site_transfer",
            ["gbdt_retrain", "tabpfn_in_context", "gbdt_transfer_no_retrain"],
        )
        summary["site_transfer"] = {
            **models,
            "paired_deltas": {
                "tabpfn_minus_gbdt_retrain_roc_auc": paired_delta_summary(
                    models["tabpfn_in_context"],
                    models["gbdt_retrain"],
                    metric="val_auc",
                    label="TabPFN-in-context - GBDT-retrain",
                ),
                "tabpfn_minus_gbdt_retrain_pr_auc": paired_delta_summary(
                    models["tabpfn_in_context"],
                    models["gbdt_retrain"],
                    metric="pr_auc",
                    label="TabPFN-in-context - GBDT-retrain",
                ),
            },
        }
    if "label_scarcity" in axes_requested:
        sizes: list[dict[str, Any]] = []
        first_axis = next(iter(per_val_seed_axes.values())).get("label_scarcity")
        if first_axis:
            for i, size_cell in enumerate(first_axis["sizes"]):
                support_size = size_cell["support_size"]
                gbdt_cells: list[dict[str, Any]] = []
                tabpfn_cells: list[dict[str, Any]] = []
                for axis in per_val_seed_axes.values():
                    size_axis = axis["label_scarcity"]["sizes"][i]
                    if size_axis["support_size"] != support_size:
                        raise AssertionError("support-size order drifted")
                    gbdt_cells.extend(completed_raw(size_axis["gbdt"]))
                    tabpfn_cells.extend(completed_raw(size_axis["tabpfn"]))
                gbdt_agg = aggregate(gbdt_cells)
                tabpfn_agg = aggregate(tabpfn_cells)
                pr_delta = paired_delta_summary(
                    tabpfn_agg,
                    gbdt_agg,
                    metric="pr_auc",
                    label=f"TabPFN - GBDT PR-AUC at support={support_size}",
                )
                roc_delta = paired_delta_summary(
                    tabpfn_agg,
                    gbdt_agg,
                    metric="val_auc",
                    label=f"TabPFN - GBDT ROC-AUC at support={support_size}",
                )
                pr_values = [p["delta"] for p in pr_delta["pairs"]]
                sizes.append(
                    {
                        "support_size": int(support_size),
                        "gbdt": gbdt_agg,
                        "tabpfn": tabpfn_agg,
                        "paired_deltas": {
                            "tabpfn_minus_gbdt_roc_auc": roc_delta,
                            "tabpfn_minus_gbdt_pr_auc": pr_delta,
                            "tabpfn_minus_gbdt_pr_auc_bootstrap_ci": bootstrap_mean_ci(
                                pr_values
                            ),
                        },
                    }
                )
        summary["label_scarcity"] = {"sizes": sizes}
    if "minimal_fe" in axes_requested:
        feature_sets: list[dict[str, Any]] = []
        first_axis = next(iter(per_val_seed_axes.values())).get("minimal_fe")
        if first_axis:
            for i, feature_set in enumerate(first_axis["feature_sets"]):
                name = feature_set["name"]
                gbdt_cells = []
                tabpfn_cells = []
                for axis in per_val_seed_axes.values():
                    fs_axis = axis["minimal_fe"]["feature_sets"][i]
                    if fs_axis["name"] != name:
                        raise AssertionError("feature-set order drifted")
                    gbdt_cells.extend(completed_raw(fs_axis["gbdt"]))
                    tabpfn_cells.extend(completed_raw(fs_axis["tabpfn"]))
                gbdt_agg = aggregate(gbdt_cells)
                tabpfn_agg = aggregate(tabpfn_cells)
                feature_sets.append(
                    {
                        "name": name,
                        "n_features": int(feature_set["n_features"]),
                        "gbdt": gbdt_agg,
                        "tabpfn": tabpfn_agg,
                        "paired_deltas": {
                            "tabpfn_minus_gbdt_roc_auc": paired_delta_summary(
                                tabpfn_agg,
                                gbdt_agg,
                                metric="val_auc",
                                label=f"TabPFN - GBDT ROC-AUC for {name}",
                            ),
                            "tabpfn_minus_gbdt_pr_auc": paired_delta_summary(
                                tabpfn_agg,
                                gbdt_agg,
                                metric="pr_auc",
                                label=f"TabPFN - GBDT PR-AUC for {name}",
                            ),
                        },
                    }
                )
        summary["minimal_fe"] = {"feature_sets": feature_sets}
    return summary


# --------------------------------------------------------------------------- #
# Feature table
# --------------------------------------------------------------------------- #
def build_split_table(
    df,
    val_mask: np.ndarray,
    *,
    split_label: str,
    value_change_regime: str,
) -> dict[str, Any]:
    train_buildings = set(df.loc[~val_mask, "building_id"].unique())
    val_buildings = set(df.loc[val_mask, "building_id"].unique())
    overlap = assert_no_building_overlap(
        train_buildings, val_buildings, split_name=split_label
    )
    train_full = add_value_change_features(
        df.loc[~val_mask], list(SHIFTS), value_change_regime=value_change_regime
    )
    val_full = add_value_change_features(
        df.loc[val_mask], list(SHIFTS), value_change_regime=value_change_regime
    )
    value_cols = [c for c in train_full.columns if c.startswith("lag_value_")]
    feature_cols = BASELINE_FEATURE_COLS + value_cols
    if len(feature_cols) != 137:
        raise AssertionError(f"Expected 137 features, got {len(feature_cols)}")
    return {
        "train_full": train_full,
        "val_full": val_full,
        "feature_cols": feature_cols,
        "y_train_full": train_full["anomaly"],
        "y_val_full": val_full["anomaly"],
        "ds_idx_full": downsample_indices(train_full["anomaly"]),
        "split": {
            "name": split_label,
            "value_change_regime_effective": value_change_regime,
            "n_train_buildings": int(len(train_buildings)),
            "n_val_buildings": int(len(val_buildings)),
            "n_train_rows": int((~val_mask).sum()),
            "n_val_rows": int(val_mask.sum()),
            "train_anomaly_rate": float(df.loc[~val_mask, "anomaly"].mean()),
            "val_anomaly_rate": float(df.loc[val_mask, "anomaly"].mean()),
            "building_overlap": int(len(overlap)),
        },
    }


def make_xy(table, fit_idx, val_idx, feature_cols):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(table["train_full"].loc[fit_idx, feature_cols])
    x_val = scaler.transform(table["val_full"].loc[val_idx, feature_cols])
    y_train = table["y_train_full"].loc[fit_idx]
    y_val = table["y_val_full"].loc[val_idx]
    return x_train, y_train, x_val, y_val, scaler


# --------------------------------------------------------------------------- #
# Cell runner (one model, one seed)
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self, args, env) -> None:
        self.args = args
        self.env = env
        self.device = str(env.get("device", "cpu"))
        self.model_path = (
            args.model_path.resolve() if args.model_path is not None else None
        )
        self.local_ckpt = bool(self.model_path and self.model_path.is_file())
        self.tabpfn_ok = (
            not args.skip_tabpfn
            and env["tabpfn_installed"]
            and env["torch_installed"]
            and self.local_ckpt
        )

    def gbdt_cell(self, x_train, y_train, x_val, y_val, *, seed, fit_rows):
        out = fit_gbdt(x_train, y_train, x_val, y_val, seed=seed)
        m = out["metrics"]
        m.update({"status": "completed", "seed": seed, "fit_rows": int(fit_rows)})
        return out["model"], m

    def tabpfn_cell(self, x_train, y_train, x_val, y_val, *, seed, fit_rows):
        if not self.tabpfn_ok:
            return {
                "status": "skipped",
                "seed": seed,
                "reason": "tabpfn unavailable, --skip-tabpfn, or no local checkpoint",
            }
        try:
            m = fit_tabpfn(
                x_train,
                y_train,
                x_val,
                y_val,
                device=self.device,
                model_path=self.model_path if self.local_ckpt else None,
            )
            m.update(
                {
                    "status": "completed",
                    "seed": seed,
                    "fit_rows": int(fit_rows),
                    "device": self.device,
                    "batch_size": int(self.args.tabpfn_batch_size),
                }
            )
            return m
        except Exception as exc:  # pragma: no cover - depends on GPU/VRAM.
            return {
                "status": "failed",
                "seed": seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=6),
            }


# --------------------------------------------------------------------------- #
# Axes
# --------------------------------------------------------------------------- #
def axis_in_domain(runner, table, val_idx, args) -> dict[str, Any]:
    fc = table["feature_cols"]
    gbdt_cells, tabpfn_cells = [], []
    kept_model = None
    for seed in args.seeds:
        fit_idx = balanced_subsample_indices(
            table["ds_idx_full"], table["y_train_full"], args.tabpfn_fit_rows, seed
        )
        x_tr, y_tr, x_va, y_va, scaler = make_xy(table, fit_idx, val_idx, fc)
        model, gm = runner.gbdt_cell(
            x_tr, y_tr, x_va, y_va, seed=seed, fit_rows=len(fit_idx)
        )
        gbdt_cells.append(gm)
        log(
            f"  [in_domain] GBDT seed={seed} AUC={gm['val_auc']:.4f} PR={gm['pr_auc']:.4f}"
        )
        if kept_model is None:  # reuse seed-0 model + scaler for site transfer.
            kept_model = {
                "model": model,
                "scaler": scaler,
                "seed": seed,
                "fit_rows": int(len(fit_idx)),
            }
        tm = runner.tabpfn_cell(
            x_tr, y_tr, x_va, y_va, seed=seed, fit_rows=len(fit_idx)
        )
        tabpfn_cells.append(tm)
        if tm.get("status") == "completed":
            log(
                f"  [in_domain] TabPFN seed={seed} AUC={tm['val_auc']:.4f} "
                f"PR={tm['pr_auc']:.4f} t={tm['fit_predict_seconds']:.2f}s"
            )
    return {
        "description": "TabPFN vs GBDT, identical 80/20 table, balanced fit budget.",
        "fit_rows_budget": int(args.tabpfn_fit_rows),
        "val_rows": int(len(val_idx)),
        "gbdt": aggregate(gbdt_cells),
        "tabpfn": aggregate(tabpfn_cells),
        "_transfer_model": kept_model,
    }


def axis_site_transfer(
    runner, table, val_idx, in_domain_transfer, args
) -> dict[str, Any]:
    fc = table["feature_cols"]
    retrain_cells, tabpfn_cells, transfer_cells = [], [], []
    for seed in args.seeds:
        fit_idx = balanced_subsample_indices(
            table["ds_idx_full"], table["y_train_full"], args.tabpfn_fit_rows, seed
        )
        x_tr, y_tr, x_va, y_va, _ = make_xy(table, fit_idx, val_idx, fc)
        _, gm = runner.gbdt_cell(
            x_tr, y_tr, x_va, y_va, seed=seed, fit_rows=len(fit_idx)
        )
        retrain_cells.append(gm)
        log(f"  [site] GBDT-retrain seed={seed} AUC={gm['val_auc']:.4f}")
        tm = runner.tabpfn_cell(
            x_tr, y_tr, x_va, y_va, seed=seed, fit_rows=len(fit_idx)
        )
        tabpfn_cells.append(tm)
        if tm.get("status") == "completed":
            log(f"  [site] TabPFN-in-context seed={seed} AUC={tm['val_auc']:.4f}")
    # GBDT-transfer-without-retrain: apply the in-domain GBDT to held-out-site val.
    if in_domain_transfer is not None:
        x_va_t = in_domain_transfer["scaler"].transform(
            table["val_full"].loc[val_idx, fc]
        )
        y_va_t = table["y_val_full"].loc[val_idx]
        tm = score_gbdt(in_domain_transfer["model"], x_va_t, y_va_t)
        tm.update(
            {
                "status": "completed",
                "seed": in_domain_transfer["seed"],
                "source": "in_domain_80_20_gbdt",
                "note": "no site-aware retrain",
            }
        )
        transfer_cells.append(tm)
        log(f"  [site] GBDT-transfer(no-retrain) AUC={tm['val_auc']:.4f}")
    return {
        "description": (
            "Site-held-out transfer. GBDT-retrain and TabPFN-in-context train on "
            "source-site rows; GBDT-transfer reuses the in-domain 80/20 GBDT "
            "without retraining. All score the same held-out-site subsample."
        ),
        "split_rule": SITE_TRANSFER_RULE,
        "m3_ensemble_anchor_auc": SITE_ANCHOR_ENSEMBLE_AUC,
        "anchor_note": (
            "0.9774 is the M3 4-model ensemble site-held-out diagnostic; the "
            "single-GBDT numbers here are not the ensemble and are expected to differ."
        ),
        "fit_rows_budget": int(args.tabpfn_fit_rows),
        "val_rows": int(len(val_idx)),
        "gbdt_retrain": aggregate(retrain_cells),
        "tabpfn_in_context": aggregate(tabpfn_cells),
        "gbdt_transfer_no_retrain": aggregate(transfer_cells),
    }


def axis_label_scarcity(runner, table, val_idx, args) -> dict[str, Any]:
    fc = table["feature_cols"]
    sizes_out = []
    for size in args.scarcity_sizes:
        gbdt_cells, tabpfn_cells = [], []
        for seed in args.seeds:
            fit_idx = balanced_subsample_indices(
                table["ds_idx_full"], table["y_train_full"], size, seed
            )
            x_tr, y_tr, x_va, y_va, _ = make_xy(table, fit_idx, val_idx, fc)
            _, gm = runner.gbdt_cell(
                x_tr, y_tr, x_va, y_va, seed=seed, fit_rows=len(fit_idx)
            )
            gbdt_cells.append(gm)
            tm = runner.tabpfn_cell(
                x_tr, y_tr, x_va, y_va, seed=seed, fit_rows=len(fit_idx)
            )
            tabpfn_cells.append(tm)
        g_agg, t_agg = aggregate(gbdt_cells), aggregate(tabpfn_cells)
        g_auc = g_agg.get("mean", {}).get("val_auc", float("nan"))
        t_auc = t_agg.get("mean", {}).get("val_auc", float("nan"))
        log(f"  [scarcity] size={size:>6} GBDT_AUC={g_auc:.4f} TabPFN_AUC={t_auc:.4f}")
        sizes_out.append({"support_size": int(size), "gbdt": g_agg, "tabpfn": t_agg})
    return {
        "description": "Degradation vs labeled support size; fixed val subsample.",
        "val_rows": int(len(val_idx)),
        "sizes": sizes_out,
    }


def axis_minimal_fe(runner, table, val_idx, args) -> dict[str, Any]:
    full_fc = table["feature_cols"]
    raw_fc = list(BASELINE_FEATURE_COLS)
    feature_sets = [
        {"name": "raw_baseline", "cols": raw_fc},
        {"name": "full_137", "cols": full_fc},
    ]
    out = []
    for fs in feature_sets:
        gbdt_cells, tabpfn_cells = [], []
        for seed in args.seeds:
            fit_idx = balanced_subsample_indices(
                table["ds_idx_full"], table["y_train_full"], args.tabpfn_fit_rows, seed
            )
            x_tr, y_tr, x_va, y_va, _ = make_xy(table, fit_idx, val_idx, fs["cols"])
            _, gm = runner.gbdt_cell(
                x_tr, y_tr, x_va, y_va, seed=seed, fit_rows=len(fit_idx)
            )
            gbdt_cells.append(gm)
            tm = runner.tabpfn_cell(
                x_tr, y_tr, x_va, y_va, seed=seed, fit_rows=len(fit_idx)
            )
            tabpfn_cells.append(tm)
        out.append(
            {
                "name": fs["name"],
                "n_features": len(fs["cols"]),
                "gbdt": aggregate(gbdt_cells),
                "tabpfn": aggregate(tabpfn_cells),
            }
        )
        log(f"  [minimal_fe] {fs['name']} ({len(fs['cols'])} feats) done")
    return {
        "description": (
            "Feature-engineering burden: each model on raw 17 baseline features vs "
            "the full 137-feature value-change line. Smaller AUC drop = lower FE need."
        ),
        "val_rows": int(len(val_idx)),
        "feature_sets": out,
    }


# --------------------------------------------------------------------------- #
# One validation seed run
# --------------------------------------------------------------------------- #
def run_axes_for_val_seed(
    *,
    df,
    args,
    runner,
    val_seed: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    axes: dict[str, Any] = {}

    needs_8020 = any(
        a in args.axes for a in ("in_domain", "label_scarcity", "minimal_fe")
    )
    transfer_model = None
    table_8020_meta = None
    full_shape = None
    if needs_8020:
        mask_8020 = (df["building_id"] % 5 == 4).to_numpy()
        table = build_split_table(
            df,
            mask_8020,
            split_label=IN_DOMAIN_SPLIT,
            value_change_regime=args.value_change_regime,
        )
        table_8020_meta = table["split"]
        full_shape = {
            "downsampled_train_rows": int(len(table["ds_idx_full"])),
            "feature_count": int(len(table["feature_cols"])),
            **tabpfn_limit_fit(len(table["ds_idx_full"]), len(table["feature_cols"])),
        }
        val_idx = random_val_indices(
            table["val_full"].index.to_numpy(), args.val_rows, val_seed
        )
        if "in_domain" in args.axes:
            log(f"Axis 1: in_domain (val_seed={val_seed})")
            res = axis_in_domain(runner, table, val_idx, args)
            transfer_model = res.pop("_transfer_model", None)
            axes["in_domain"] = add_val_seed_to_axis(res, val_seed)
        elif "site_transfer" in args.axes:
            # Need a transfer model even if in_domain axis not requested.
            res = axis_in_domain(runner, table, val_idx, args)
            transfer_model = res.get("_transfer_model")
        if "label_scarcity" in args.axes:
            log(f"Axis 3: label_scarcity (val_seed={val_seed})")
            axes["label_scarcity"] = add_val_seed_to_axis(
                axis_label_scarcity(runner, table, val_idx, args),
                val_seed,
            )
        if "minimal_fe" in args.axes:
            log(f"Axis 4: minimal_fe (val_seed={val_seed})")
            axes["minimal_fe"] = add_val_seed_to_axis(
                axis_minimal_fe(runner, table, val_idx, args),
                val_seed,
            )
        del table

    if "site_transfer" in args.axes:
        val_site_ids = sorted(site for site in df["site_id"].unique() if site % 5 == 4)
        mask_site = leave_site_out_mask(df, val_site_ids)
        site_table = build_split_table(
            df,
            mask_site,
            split_label=SITE_TRANSFER_RULE,
            value_change_regime=args.value_change_regime,
        )
        if transfer_model is None and not needs_8020:
            # Build an in-domain transfer model from the 80/20 split on demand.
            mask_8020 = (df["building_id"] % 5 == 4).to_numpy()
            t8020 = build_split_table(
                df,
                mask_8020,
                split_label=IN_DOMAIN_SPLIT,
                value_change_regime=args.value_change_regime,
            )
            v8020 = random_val_indices(
                t8020["val_full"].index.to_numpy(), args.val_rows, val_seed
            )
            transfer_model = axis_in_domain(runner, t8020, v8020, args).get(
                "_transfer_model"
            )
            del t8020
        val_idx_site = random_val_indices(
            site_table["val_full"].index.to_numpy(), args.val_rows, val_seed
        )
        log(f"Axis 2: site_transfer (PRIMARY, val_seed={val_seed})")
        axes["site_transfer"] = add_val_seed_to_axis(
            axis_site_transfer(runner, site_table, val_idx_site, transfer_model, args),
            val_seed,
        )
        axes["site_transfer"]["split"] = site_table["split"]
        del site_table
    return axes, table_8020_meta, full_shape


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    if args.val_seeds is None:
        args.val_seeds = [args.val_seed]
    else:
        args.val_seed = args.val_seeds[0]
        if args.out == PROC / "m5_phaseD_foundation_vs_gbdt.json":
            args.out = PROC / "inv2_phaseD_val_variance.json"
    if args.smoke:
        args.tabpfn_fit_rows = min(args.tabpfn_fit_rows, 400)
        args.val_rows = min(args.val_rows, 400)
        args.scarcity_sizes = [200, 400]
        args.seeds = args.seeds[:2]
        args.val_seeds = args.val_seeds[:2]
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
        f"fit_rows={args.tabpfn_fit_rows} val_rows={args.val_rows} "
        f"seeds={args.seeds} val_seeds={args.val_seeds} "
        f"value_change_regime={args.value_change_regime}"
    )

    df = load_m3_frame(verbose=True)
    per_val_seed_axes: dict[str, dict[str, Any]] = {}
    table_8020_meta = None
    full_shape = None
    for val_seed in args.val_seeds:
        axes_for_seed, table_meta, shape = run_axes_for_val_seed(
            df=df,
            args=args,
            runner=runner,
            val_seed=val_seed,
        )
        per_val_seed_axes[str(val_seed)] = axes_for_seed
        if table_8020_meta is None and table_meta is not None:
            table_8020_meta = table_meta
        if full_shape is None and shape is not None:
            full_shape = shape

    axes = (
        next(iter(per_val_seed_axes.values()))
        if len(per_val_seed_axes) == 1
        else summarize_val_resampling(per_val_seed_axes, list(args.axes))
    )
    val_resampling = {
        "enabled": len(args.val_seeds) > 1,
        "val_seeds": [int(s) for s in args.val_seeds],
        "per_val_seed_axes": per_val_seed_axes if len(args.val_seeds) > 1 else {},
        "summary": axes if len(args.val_seeds) > 1 else {},
        "notes": (
            "Summary aggregates all completed model-seed x validation-seed cells; "
            "paired deltas use matching validation seed and model seed."
        ),
    }

    is_m6_meter_aware = args.value_change_regime == "row_offset_meter_aware"
    results = {
        "experiment": "m6_phaseD_meter_aware"
        if is_m6_meter_aware
        else "inv2_phaseD_val_variance"
        if len(args.val_seeds) > 1
        else "m5_phaseD_foundation_vs_gbdt",
        "issue": 52 if is_m6_meter_aware else 51 if len(args.val_seeds) > 1 else 35,
        "scope": (
            "Existing M3 GEPIII data only; no BDG2, no cloud; TabPFN local weights."
        ),
        "value_change_regime": args.value_change_regime,
        "value_change_regime_default": DEFAULT_VALUE_CHANGE_REGIME,
        "value_change_regime_self_check": {
            "requested": args.value_change_regime,
            "effective_in_domain": table_8020_meta.get("value_change_regime_effective")
            if table_8020_meta
            else None,
            "matches_requested": bool(
                table_8020_meta
                and table_8020_meta.get("value_change_regime_effective")
                == args.value_change_regime
            ),
        },
        "budgets": {
            "tabpfn_fit_rows": int(args.tabpfn_fit_rows),
            "val_rows": int(args.val_rows),
            "scarcity_sizes": list(args.scarcity_sizes),
            "seeds": list(args.seeds),
            "val_seed": int(args.val_seed),
            "val_seeds": [int(s) for s in args.val_seeds],
            "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
            "tabpfn_fit_rows_rationale": (
                "Balanced budget bounded by 8 GB laptop VRAM. 137 features <= 200 and "
                "the budget << 1,000,000 rows, so the run stays within the documented "
                "TabPFN-3 1,000,000 x 200 limit without ignore_pretraining_limits."
            ),
        },
        "in_domain_split": table_8020_meta,
        "full_downsample_shape": full_shape,
        "environment": env,
        "axes": axes,
        "val_resampling": val_resampling,
        "one_shot_discipline": {
            "leaderboard_probing": False,
            "cloud_client_used": False,
            "bdg2_used": False,
            "ignore_pretraining_limits": False,
        },
        "elapsed_seconds": float(time.perf_counter() - t0),
        "timing_protocol": timing_protocol(),
    }
    cmd = (
        "uv run python scripts/run_m5_phaseD_foundation_vs_gbdt.py "
        f"--out {args.out} "
        f"--tabpfn-fit-rows {args.tabpfn_fit_rows} --val-rows {args.val_rows} "
        f"--value-change-regime {args.value_change_regime} "
        f"--axes {' '.join(args.axes)} "
        f"--seeds {' '.join(map(str, args.seeds))} "
        f"--val-seeds {' '.join(map(str, args.val_seeds))}"
    )
    write_json_with_provenance(
        args.out, results, root=ROOT, provenance={"command": cmd}
    )
    log(f"Saved {args.out} in {results['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
