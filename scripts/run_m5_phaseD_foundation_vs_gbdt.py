"""M5 Phase D: rigorous TabPFN vs GBDT comparison on the existing M3 GEPIII data.

This harness runs paired, multi-seed comparisons through the frozen ``src/lead``
pipeline. Every paired cell reuses the same split, downsample, feature table, and
fixed validation subsample so the only variable is the model. No new dataset, no
BDG2, no cloud: TabPFN runs from local weights only.

Axes:
  1. in_domain      - TabPFN vs GBDT on the 80/20 (``building_id % 5 == 4``) split.
  2. site_transfer  - PRIMARY. Site-held-out (``site_id % 5 == 4``) split. TabPFN
                      in-context vs GBDT-retrain vs GBDT-transfer-without-retrain.
                      M3 ensemble anchor AUC 0.9774.
  3. label_scarcity - shrink the labeled support set across sizes; show degradation.
  4. minimal_fe     - TabPFN/GBDT on a reduced raw feature set vs the 137-feature
                      line; quantify the feature-engineering-burden difference.

Metrics per cell: ROC-AUC, PR-AUC (average precision), precision/recall/F1 at the
0.5 threshold, and fit+predict latency. Multiple seeds per cell are aggregated to
mean +/- std.
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

VALUE_CHANGE_REGIME = "row_offset"
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


# --------------------------------------------------------------------------- #
# Feature table
# --------------------------------------------------------------------------- #
def build_split_table(df, val_mask: np.ndarray, *, split_label: str) -> dict[str, Any]:
    train_buildings = set(df.loc[~val_mask, "building_id"].unique())
    val_buildings = set(df.loc[val_mask, "building_id"].unique())
    overlap = assert_no_building_overlap(
        train_buildings, val_buildings, split_name=split_label
    )
    train_full = add_value_change_features(
        df.loc[~val_mask], list(SHIFTS), value_change_regime=VALUE_CHANGE_REGIME
    )
    val_full = add_value_change_features(
        df.loc[val_mask], list(SHIFTS), value_change_regime=VALUE_CHANGE_REGIME
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
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    if args.smoke:
        args.tabpfn_fit_rows = min(args.tabpfn_fit_rows, 400)
        args.val_rows = min(args.val_rows, 400)
        args.scarcity_sizes = [200, 400]
        args.seeds = args.seeds[:2]
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
        f"fit_rows={args.tabpfn_fit_rows} val_rows={args.val_rows} seeds={args.seeds}"
    )

    df = load_m3_frame(verbose=True)
    axes: dict[str, Any] = {}

    needs_8020 = any(
        a in args.axes for a in ("in_domain", "label_scarcity", "minimal_fe")
    )
    transfer_model = None
    table_8020_meta = None
    full_shape = None
    if needs_8020:
        mask_8020 = (df["building_id"] % 5 == 4).to_numpy()
        table = build_split_table(df, mask_8020, split_label=IN_DOMAIN_SPLIT)
        table_8020_meta = table["split"]
        full_shape = {
            "downsampled_train_rows": int(len(table["ds_idx_full"])),
            "feature_count": int(len(table["feature_cols"])),
            **tabpfn_limit_fit(len(table["ds_idx_full"]), len(table["feature_cols"])),
        }
        val_idx = random_val_indices(
            table["val_full"].index.to_numpy(), args.val_rows, args.val_seed
        )
        if "in_domain" in args.axes:
            log("Axis 1: in_domain")
            res = axis_in_domain(runner, table, val_idx, args)
            transfer_model = res.pop("_transfer_model", None)
            axes["in_domain"] = res
        elif "site_transfer" in args.axes:
            # Need a transfer model even if in_domain axis not requested.
            res = axis_in_domain(runner, table, val_idx, args)
            transfer_model = res.get("_transfer_model")
        if "label_scarcity" in args.axes:
            log("Axis 3: label_scarcity")
            axes["label_scarcity"] = axis_label_scarcity(runner, table, val_idx, args)
        if "minimal_fe" in args.axes:
            log("Axis 4: minimal_fe")
            axes["minimal_fe"] = axis_minimal_fe(runner, table, val_idx, args)
        del table

    if "site_transfer" in args.axes:
        val_site_ids = sorted(site for site in df["site_id"].unique() if site % 5 == 4)
        mask_site = leave_site_out_mask(df, val_site_ids)
        site_table = build_split_table(df, mask_site, split_label=SITE_TRANSFER_RULE)
        if transfer_model is None and not needs_8020:
            # Build an in-domain transfer model from the 80/20 split on demand.
            mask_8020 = (df["building_id"] % 5 == 4).to_numpy()
            t8020 = build_split_table(df, mask_8020, split_label=IN_DOMAIN_SPLIT)
            v8020 = random_val_indices(
                t8020["val_full"].index.to_numpy(), args.val_rows, args.val_seed
            )
            transfer_model = axis_in_domain(runner, t8020, v8020, args).get(
                "_transfer_model"
            )
            del t8020
        val_idx_site = random_val_indices(
            site_table["val_full"].index.to_numpy(), args.val_rows, args.val_seed
        )
        log("Axis 2: site_transfer (PRIMARY)")
        axes["site_transfer"] = axis_site_transfer(
            runner, site_table, val_idx_site, transfer_model, args
        )
        axes["site_transfer"]["split"] = site_table["split"]
        del site_table

    results = {
        "experiment": "m5_phaseD_foundation_vs_gbdt",
        "issue": 35,
        "scope": (
            "Existing M3 GEPIII data only; no BDG2, no cloud; TabPFN local weights."
        ),
        "value_change_regime": VALUE_CHANGE_REGIME,
        "budgets": {
            "tabpfn_fit_rows": int(args.tabpfn_fit_rows),
            "val_rows": int(args.val_rows),
            "scarcity_sizes": list(args.scarcity_sizes),
            "seeds": list(args.seeds),
            "val_seed": int(args.val_seed),
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
        "one_shot_discipline": {
            "leaderboard_probing": False,
            "cloud_client_used": False,
            "bdg2_used": False,
            "ignore_pretraining_limits": False,
        },
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    cmd = (
        "python scripts/run_m5_phaseD_foundation_vs_gbdt.py "
        f"--tabpfn-fit-rows {args.tabpfn_fit_rows} --val-rows {args.val_rows} "
        f"--seeds {' '.join(map(str, args.seeds))}"
    )
    write_json_with_provenance(
        args.out, results, root=ROOT, provenance={"command": cmd}
    )
    log(f"Saved {args.out} in {results['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
