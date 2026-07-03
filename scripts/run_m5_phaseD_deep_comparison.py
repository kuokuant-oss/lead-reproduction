"""M5 Phase D deep comparison axes on the existing GEPIII/M3 table.

This is an additive experiment line. It keeps the existing M6 50/50 artifact
untouched while writing paired train/val/test numbers for follow-up analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sklearn.base import clone
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
    downsample_indices,
    load_m3_frame,
    write_json_with_provenance,
)
from run_m6_phaseD_50_50_full_models import (
    Runner,
    balanced_subsample_indices,
    index_record,
    metric_summary,
    random_indices,
    torch_environment,
    tree_models,
)


VALUE_CHANGE_REGIME = "row_offset_meter_aware"
MODEL_ORDER = (
    "lightgbm",
    "xgboost",
    "catboost",
    "hist_gradient_boosting",
    "ensemble",
    "tabpfn",
)
AXIS_ORDER = (
    "default_vs_tuned",
    "sample_efficiency_fine",
    "dimensionality_at_small_n",
    "stability_multiseed",
)
TUNE_SPACE: dict[str, dict[str, list[Any]]] = {
    "lightgbm": {
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "num_leaves": [15, 31, 63],
        "n_estimators": [100, 200, 400],
        "min_child_samples": [10, 20, 50],
        "reg_lambda": [0.0, 1.0, 5.0],
    },
    "xgboost": {
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 200, 400],
        "min_child_weight": [1, 3, 7],
        "reg_lambda": [0.0, 1.0, 5.0],
    },
    "catboost": {
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "depth": [4, 6, 8],
        "iterations": [200, 500, 1000],
        "l2_leaf_reg": [1.0, 3.0, 10.0],
    },
    "hist_gradient_boosting": {
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "max_leaf_nodes": [15, 31, 63],
        "max_iter": [100, 200, 400],
        "min_samples_leaf": [10, 20, 50],
        "l2_regularization": [0.0, 1.0, 5.0],
    },
}


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
        default=PROC / "m5_phaseD_deep_comparison.json",
    )
    parser.add_argument(
        "--handoff",
        type=Path,
        default=ROOT / "docs" / "handoffs" / "m5-phaseD-deep-comparison.md",
    )
    parser.add_argument("--fit-rows", type=int, default=10_000)
    parser.add_argument("--score-rows", type=int, default=4_000)
    parser.add_argument(
        "--scarcity-sizes",
        type=int,
        nargs="+",
        default=[20, 50, 100, 150, 300, 500, 1_000, 2_000],
    )
    parser.add_argument("--tune-trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--skip-tabpfn", action="store_true")
    parser.add_argument("--model-path", type=Path, default=default_model_path)
    parser.add_argument(
        "--axes",
        nargs="+",
        choices=AXIS_ORDER,
        default=list(AXIS_ORDER),
        help=(
            "Subset of axes to run. Existing unrequested axes in --out are "
            "preserved so long runs can be completed one shard at a time."
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def smoke_adjust(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.fit_rows = min(args.fit_rows, 200)
    args.score_rows = min(args.score_rows, 200)
    args.tune_trials = min(args.tune_trials, 3)
    args.scarcity_sizes = [50, 200]


def build_split_table(df) -> dict[str, Any]:
    fit_mask = (df["building_id"] % 4 == 0).to_numpy()
    val_mask = (df["building_id"] % 4 == 2).to_numpy()
    test_mask = (df["building_id"] % 2 == 1).to_numpy()
    fit_buildings = set(int(x) for x in df.loc[fit_mask, "building_id"].unique())
    val_buildings = set(int(x) for x in df.loc[val_mask, "building_id"].unique())
    test_buildings = set(int(x) for x in df.loc[test_mask, "building_id"].unique())
    assert_no_building_overlap(fit_buildings, val_buildings, split_name="fit_vs_val")
    assert_no_building_overlap(fit_buildings, test_buildings, split_name="fit_vs_test")
    assert_no_building_overlap(val_buildings, test_buildings, split_name="val_vs_test")

    fit_full = add_value_change_features(
        df.loc[fit_mask],
        list(SHIFTS),
        value_change_regime=VALUE_CHANGE_REGIME,
    )
    val_full = add_value_change_features(
        df.loc[val_mask],
        list(SHIFTS),
        value_change_regime=VALUE_CHANGE_REGIME,
    )
    test_full = add_value_change_features(
        df.loc[test_mask],
        list(SHIFTS),
        value_change_regime=VALUE_CHANGE_REGIME,
    )
    value_cols = [c for c in fit_full.columns if c.startswith("lag_value_")]
    if len(value_cols) != 120:
        raise AssertionError(
            f"Expected 120 value-change columns, got {len(value_cols)}"
        )
    return {
        "fit_full": fit_full,
        "val_full": val_full,
        "test_full": test_full,
        "value_cols": value_cols,
        "ds_idx_fit": downsample_indices(fit_full["anomaly"]),
        "split": {
            "name": "50_50_mod2_with_nested_mod4_val",
            "top_level_train_rule": "building_id % 2 == 0",
            "test_rule": "building_id % 2 == 1",
            "fit_train_score_rule": "building_id % 4 == 0",
            "val_rule": "building_id % 4 == 2",
            "unit_type": "building_id",
            "n_fit_buildings": int(len(fit_buildings)),
            "n_val_buildings": int(len(val_buildings)),
            "n_test_buildings": int(len(test_buildings)),
            "n_fit_rows": int(fit_mask.sum()),
            "n_val_rows": int(val_mask.sum()),
            "n_test_rows": int(test_mask.sum()),
            "fit_anomaly_rate": float(df.loc[fit_mask, "anomaly"].mean()),
            "val_anomaly_rate": float(df.loc[val_mask, "anomaly"].mean()),
            "test_anomaly_rate": float(df.loc[test_mask, "anomaly"].mean()),
            "building_overlaps": {
                "fit_val": 0,
                "fit_test": 0,
                "val_test": 0,
            },
        },
    }


def feature_sets(table: dict[str, Any]) -> dict[str, list[str]]:
    value_cols = list(table["value_cols"])
    return {
        "raw_baseline_17": list(BASELINE_FEATURE_COLS),
        "baseline_plus_first_33_value_change_50": list(BASELINE_FEATURE_COLS)
        + value_cols[:33],
        "full_137": list(BASELINE_FEATURE_COLS) + value_cols,
    }


def make_indices(
    table: dict[str, Any],
    *,
    fit_rows: int,
    score_rows: int,
    seed: int,
) -> dict[str, np.ndarray]:
    return {
        "fit_set": balanced_subsample_indices(
            table["ds_idx_fit"],
            table["fit_full"]["anomaly"],
            min(fit_rows, 10_000),
            seed,
        ),
        "train": random_indices(
            table["fit_full"].index.to_numpy(),
            score_rows,
            seed + 10_000,
        ),
        "val": random_indices(
            table["val_full"].index.to_numpy(),
            score_rows,
            seed + 20_000,
        ),
        "test": random_indices(
            table["test_full"].index.to_numpy(),
            score_rows,
            seed + 30_000,
        ),
    }


def make_matrices(
    table: dict[str, Any],
    feature_cols: list[str],
    indices: dict[str, np.ndarray],
) -> dict[str, Any]:
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(
        table["fit_full"].loc[indices["fit_set"], feature_cols]
    )
    return {
        "x_fit": x_fit,
        "y_fit": table["fit_full"].loc[indices["fit_set"], "anomaly"],
        "x_fit_score": x_fit,
        "y_fit_score": table["fit_full"].loc[indices["fit_set"], "anomaly"],
        "x_train_score": scaler.transform(
            table["fit_full"].loc[indices["train"], feature_cols]
        ),
        "y_train_score": table["fit_full"].loc[indices["train"], "anomaly"],
        "x_val_score": scaler.transform(
            table["val_full"].loc[indices["val"], feature_cols]
        ),
        "y_val_score": table["val_full"].loc[indices["val"], "anomaly"],
        "x_test_score": scaler.transform(
            table["test_full"].loc[indices["test"], feature_cols]
        ),
        "y_test_score": table["test_full"].loc[indices["test"], "anomaly"],
    }


def score_predictions(
    matrices: dict[str, Any], preds: dict[str, np.ndarray]
) -> dict[str, Any]:
    return {
        "fit_set": metric_summary(matrices["y_fit_score"], preds["fit_set"]),
        "train": metric_summary(matrices["y_train_score"], preds["train"]),
        "val": metric_summary(matrices["y_val_score"], preds["val"]),
        "test": metric_summary(matrices["y_test_score"], preds["test"]),
    }


def predict_sets(
    model, matrices: dict[str, Any], *, nan_to_num: bool = False
) -> dict[str, np.ndarray]:
    def clean(x):
        return np.nan_to_num(x, nan=0) if nan_to_num else x

    return {
        "fit_set": model.predict_proba(clean(matrices["x_fit_score"]))[:, 1],
        "train": model.predict_proba(clean(matrices["x_train_score"]))[:, 1],
        "val": model.predict_proba(clean(matrices["x_val_score"]))[:, 1],
        "test": model.predict_proba(clean(matrices["x_test_score"]))[:, 1],
    }


def row_contract(
    table: dict[str, Any],
    indices: dict[str, np.ndarray],
    *,
    fit_rows_budget: int,
    score_rows_budget: int,
) -> dict[str, Any]:
    frames = {
        "fit_set": table["fit_full"],
        "train": table["fit_full"],
        "val": table["val_full"],
        "test": table["test_full"],
    }
    return {
        "fit_rows_budget": int(fit_rows_budget),
        "score_rows_budget": int(score_rows_budget),
        "row_index_records": {name: index_record(idx) for name, idx in indices.items()},
        "row_prevalence": {
            name: float(frames[name].loc[idx, "anomaly"].mean())
            for name, idx in indices.items()
        },
    }


def fit_default_trees(matrices: dict[str, Any], *, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tree_preds = {"fit_set": [], "train": [], "val": [], "test": []}
    for name, model in tree_models(seed).items():
        log(f"    default {name}")
        nan_to_num = name == "hist_gradient_boosting"
        x_fit = (
            np.nan_to_num(matrices["x_fit"], nan=0) if nan_to_num else matrices["x_fit"]
        )
        t0 = time.perf_counter()
        model.fit(x_fit, matrices["y_fit"])
        preds = predict_sets(model, matrices, nan_to_num=nan_to_num)
        elapsed = time.perf_counter() - t0
        for key, pred in preds.items():
            tree_preds[key].append(pred)
        out[name] = {
            "status": "completed",
            "fit_predict_seconds": float(elapsed),
            **score_predictions(matrices, preds),
        }
    ensemble_preds = {
        key: sum(values) / len(values) for key, values in tree_preds.items()
    }
    out["ensemble"] = {
        "status": "completed",
        "fit_predict_seconds": None,
        **score_predictions(matrices, ensemble_preds),
    }
    return out


def sampled_config(
    space: dict[str, list[Any]], rng: np.random.RandomState
) -> dict[str, Any]:
    return {
        key: values[int(rng.randint(0, len(values)))] for key, values in space.items()
    }


def tuned_model(name: str, config: dict[str, Any], seed: int):
    model = tree_models(seed)[name]
    return clone(model).set_params(**config)


def fit_tuned_trees(
    matrices: dict[str, Any],
    *,
    seed: int,
    tune_trials: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tree_preds = {"fit_set": [], "train": [], "val": [], "test": []}
    rng = np.random.RandomState(seed + 40_000)
    for name in ("lightgbm", "xgboost", "catboost", "hist_gradient_boosting"):
        log(f"    tuning {name}")
        nan_to_num = name == "hist_gradient_boosting"
        x_fit = (
            np.nan_to_num(matrices["x_fit"], nan=0) if nan_to_num else matrices["x_fit"]
        )
        x_val = (
            np.nan_to_num(matrices["x_val_score"], nan=0)
            if nan_to_num
            else matrices["x_val_score"]
        )
        best: dict[str, Any] | None = None
        trial_records = []
        t_tune = time.perf_counter()
        for trial in range(max(1, tune_trials)):
            config = sampled_config(TUNE_SPACE[name], rng)
            model = tuned_model(name, config, seed + trial)
            t0 = time.perf_counter()
            model.fit(x_fit, matrices["y_fit"])
            pred_val = model.predict_proba(x_val)[:, 1]
            trial_seconds = time.perf_counter() - t0
            val_pr_auc = float(
                average_precision_score(matrices["y_val_score"], pred_val)
            )
            record = {
                "trial": int(trial),
                "config": config,
                "val_pr_auc": val_pr_auc,
                "fit_val_seconds": float(trial_seconds),
            }
            trial_records.append(record)
            if best is None or val_pr_auc > best["val_pr_auc"]:
                best = {"val_pr_auc": val_pr_auc, "config": config, "model": model}
        if best is None:
            raise AssertionError("tuning loop produced no trials")
        tuning_seconds = time.perf_counter() - t_tune
        preds = predict_sets(best["model"], matrices, nan_to_num=nan_to_num)
        for key, pred in preds.items():
            tree_preds[key].append(pred)
        out[name] = {
            "status": "completed",
            "n_trials": int(max(1, tune_trials)),
            "search_space": TUNE_SPACE[name],
            "best_config": best["config"],
            "selected_by": "val_pr_auc",
            "best_val_pr_auc": float(best["val_pr_auc"]),
            "tuning_seconds": float(tuning_seconds),
            "trials": trial_records,
            **score_predictions(matrices, preds),
        }
    ensemble_preds = {
        key: sum(values) / len(values) for key, values in tree_preds.items()
    }
    out["ensemble"] = {
        "status": "completed",
        "fit_predict_seconds": None,
        "selected_by": "mean tuned tree probability after per-model val PR-AUC selection",
        **score_predictions(matrices, ensemble_preds),
    }
    return out


def fit_tabpfn_all(runner: Runner, matrices: dict[str, Any]) -> dict[str, Any]:
    log("    fitting tabpfn")
    if not runner.tabpfn_ok:
        return {
            "status": "skipped",
            "reason": "tabpfn unavailable, --skip-tabpfn, or no local checkpoint",
        }
    try:
        t0 = time.perf_counter()
        model = runner.fit_tabpfn(matrices["x_fit"], matrices["y_fit"])
        preds = predict_sets(model, matrices)
        return {
            "status": "completed",
            "fit_predict_seconds": float(time.perf_counter() - t0),
            **score_predictions(matrices, preds),
        }
    except Exception as exc:  # pragma: no cover - runtime/GPU dependent.
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=6),
        }


def run_standard_cell(
    runner: Runner,
    table: dict[str, Any],
    feature_cols: list[str],
    *,
    fit_rows: int,
    score_rows: int,
    seed: int,
) -> dict[str, Any]:
    indices = make_indices(table, fit_rows=fit_rows, score_rows=score_rows, seed=seed)
    matrices = make_matrices(table, feature_cols, indices)
    models = fit_default_trees(matrices, seed=seed)
    models["tabpfn"] = fit_tabpfn_all(runner, matrices)
    return {
        **row_contract(
            table,
            indices,
            fit_rows_budget=fit_rows,
            score_rows_budget=score_rows,
        ),
        "models": models,
    }


def axis_default_vs_tuned(
    runner: Runner,
    table: dict[str, Any],
    feature_cols: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    indices = make_indices(
        table,
        fit_rows=args.fit_rows,
        score_rows=args.score_rows,
        seed=args.seed,
    )
    matrices = make_matrices(table, feature_cols, indices)
    return {
        "description": (
            "Default trees, manually tuned trees, and TabPFN on paired full-feature "
            "rows; tuned tree configs are selected only by validation PR-AUC."
        ),
        "n_features": int(len(feature_cols)),
        **row_contract(
            table,
            indices,
            fit_rows_budget=args.fit_rows,
            score_rows_budget=args.score_rows,
        ),
        "families": {
            "default_trees": fit_default_trees(matrices, seed=args.seed),
            "tuned_trees": fit_tuned_trees(
                matrices,
                seed=args.seed,
                tune_trials=args.tune_trials,
            ),
            "tabpfn": fit_tabpfn_all(runner, matrices),
        },
    }


def best_tree_test_pr(models: dict[str, Any]) -> float | None:
    values = [
        model["test"]["pr_auc"]
        for name, model in models.items()
        if name != "tabpfn" and model.get("status", "completed") == "completed"
    ]
    return float(max(values)) if values else None


def axis_sample_efficiency(
    runner: Runner,
    table: dict[str, Any],
    feature_cols: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sizes = []
    crossover = None
    for support in args.scarcity_sizes:
        log(f"  support={support}")
        cell = run_standard_cell(
            runner,
            table,
            feature_cols,
            fit_rows=support,
            score_rows=args.score_rows,
            seed=args.seed,
        )
        tabpfn = cell["models"].get("tabpfn", {})
        tabpfn_pr = (
            tabpfn.get("test", {}).get("pr_auc")
            if tabpfn.get("status") == "completed"
            else None
        )
        tree_pr = best_tree_test_pr(cell["models"])
        if crossover is None and tabpfn_pr is not None and tree_pr is not None:
            if tree_pr >= tabpfn_pr:
                crossover = {
                    "support_size": int(support),
                    "best_tree_test_pr_auc": float(tree_pr),
                    "tabpfn_test_pr_auc": float(tabpfn_pr),
                }
        sizes.append({"support_size": int(support), **cell})
    return {
        "description": "Full 137-feature sample efficiency with finer support budgets.",
        "n_features": int(len(feature_cols)),
        "sizes": sizes,
        "crossover_support": crossover,
    }


def axis_dimensionality(
    runner: Runner,
    table: dict[str, Any],
    features: dict[str, list[str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    dims = []
    for key in (
        "raw_baseline_17",
        "baseline_plus_first_33_value_change_50",
        "full_137",
    ):
        cols = features[key]
        log(f"  feature_set={key}")
        dims.append(
            {
                "feature_set": key,
                "n_features": int(len(cols)),
                "selection_rule": (
                    "BASELINE_FEATURE_COLS plus the first 33 lag_value_* columns "
                    "from row_offset_meter_aware feature generation"
                    if key == "baseline_plus_first_33_value_change_50"
                    else "BASELINE_FEATURE_COLS"
                    if key == "raw_baseline_17"
                    else "BASELINE_FEATURE_COLS plus all 120 lag_value_* columns"
                ),
                **run_standard_cell(
                    runner,
                    table,
                    cols,
                    fit_rows=500 if not args.smoke else min(args.fit_rows, 200),
                    score_rows=args.score_rows,
                    seed=args.seed,
                ),
            }
        )
    return {
        "description": "Dimensionality sweep at small n.",
        "fixed_fit_rows": 500 if not args.smoke else min(args.fit_rows, 200),
        "feature_dimensions": dims,
    }


def aggregate_metric_sets(cells: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        out[split] = {}
        for metric in ("roc_auc", "pr_auc"):
            values = [
                float(cell[split][metric])
                for cell in cells
                if cell.get("status", "completed") == "completed" and split in cell
            ]
            out[split][metric] = {
                "mean": float(mean(values)) if values else None,
                "std": float(pstdev(values))
                if len(values) > 1
                else 0.0
                if values
                else None,
                "min": float(min(values)) if values else None,
                "max": float(max(values)) if values else None,
                "n": int(len(values)),
            }
    return out


def axis_stability(
    runner: Runner,
    table: dict[str, Any],
    feature_cols: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    seeds_out = []
    per_model: dict[str, list[dict[str, Any]]] = {name: [] for name in MODEL_ORDER}
    dseeds = list(DOWNSAMPLE_SEEDS)[:1] if args.smoke else list(DOWNSAMPLE_SEEDS)
    mseeds = list(MODEL_SEEDS)[:2] if args.smoke else list(MODEL_SEEDS)
    for downsample_seed in dseeds:
        for model_seed in mseeds:
            seed = int(downsample_seed + model_seed)
            log(f"  downsample_seed={downsample_seed} model_seed={model_seed}")
            cell = run_standard_cell(
                runner,
                table,
                feature_cols,
                fit_rows=args.fit_rows,
                score_rows=args.score_rows,
                seed=seed,
            )
            seeds_out.append(
                {
                    "downsample_seed": int(downsample_seed),
                    "model_seed": int(model_seed),
                    **cell,
                }
            )
            for name, result in cell["models"].items():
                per_model.setdefault(name, []).append(result)

    same_input = []
    if runner.tabpfn_ok:
        indices = make_indices(
            table,
            fit_rows=args.fit_rows,
            score_rows=args.score_rows,
            seed=args.seed,
        )
        matrices = make_matrices(table, feature_cols, indices)
        for rerun in range(3):
            log(f"  tabpfn same-input rerun={rerun}")
            same_input.append({"rerun": int(rerun), **fit_tabpfn_all(runner, matrices)})
    else:
        same_input.append(
            {
                "status": "skipped",
                "reason": "tabpfn unavailable, --skip-tabpfn, or no local checkpoint",
            }
        )
    return {
        "description": (
            "Full 137-feature stability over DOWNSAMPLE_SEEDS x MODEL_SEEDS, plus "
            "three same-input TabPFN reruns when available."
        ),
        "n_features": int(len(feature_cols)),
        "fit_rows": int(args.fit_rows),
        "seed_grid": {
            "downsample_seeds": [int(s) for s in dseeds],
            "model_seeds": [int(s) for s in mseeds],
        },
        "summary": {
            name: aggregate_metric_sets(cells) for name, cells in per_model.items()
        },
        "tabpfn_same_input_reruns": {
            "runs": same_input,
            "std": aggregate_metric_sets(same_input).get("test", {}),
        },
        "runs": seeds_out,
    }


def headline_rows(
    axis: dict[str, Any], selector: str
) -> list[tuple[str, str, Any, Any, Any]]:
    rows = []
    models = axis
    if selector:
        for part in selector.split("."):
            models = models.get(part, {})
    for name, result in models.items():
        if not isinstance(result, dict) or "train" not in result:
            continue
        rows.append(
            (
                name,
                result.get("status", "completed"),
                result["train"]["pr_auc"],
                result["val"]["pr_auc"],
                result["test"]["pr_auc"],
            )
        )
    return rows


def table_md(rows: list[tuple[str, str, Any, Any, Any]]) -> str:
    if not rows:
        return "_No completed row-level scores available._\n"
    lines = [
        "| model | status | train PR-AUC | val PR-AUC | test PR-AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, status, train, val, test in rows:
        lines.append(f"| {model} | {status} | {train:.6f} | {val:.6f} | {test:.6f} |")
    return "\n".join(lines) + "\n"


def write_handoff(
    args: argparse.Namespace, results: dict[str, Any], command: str
) -> None:
    axes = results["axes"]
    full_command = (
        "uv run python scripts/run_m5_phaseD_deep_comparison.py "
        "--out data/processed/m5_phaseD_deep_comparison.json "
        "--handoff docs/handoffs/m5-phaseD-deep-comparison.md "
        "--fit-rows 10000 --score-rows 4000 "
        "--scarcity-sizes 20 50 100 150 300 500 1000 2000 "
        "--tune-trials 12 --seed 42 "
        "--axes default_vs_tuned sample_efficiency_fine dimensionality_at_small_n stability_multiseed"
    )
    lines = [
        "# M5 Phase D deep comparison handoff",
        "",
        "**Non-report handoff notes.** These numbers are for follow-up consumption; no `docs/reports/` narrative was updated.",
        "",
        "## Run",
        "",
        f"- Smoke mode: `{args.smoke}`",
        f"- Requested axes this run: `{' '.join(args.axes)}`",
        f"- JSON merge mode: `{results.get('merge', {}).get('mode', 'replace')}`",
        f"- Executed command: `{command}`",
        f"- JSON: `{args.out.as_posix()}`",
        f"- Full command for later real run: `{full_command}`",
        f"- Value-change regime: `{VALUE_CHANGE_REGIME}`",
        "",
        "## Operating-Point Note",
        "",
        (
            "The test `threshold_0_5` and `fixed_recall_0_90` entries are post-hoc "
            "operating points. In particular, `fixed_recall_0_90` derives its "
            "threshold from the same split's labels, including test labels for the "
            "test summary. These entries are descriptive only and do not represent "
            "deployable performance. Model comparison and TabPFN-vs-tree claims "
            "should use threshold-free ROC-AUC / PR-AUC. For deployable operating "
            "points, choose thresholds on val and apply them once to test."
        ),
        "",
        "## Axis Headlines",
        "",
    ]
    if "default_vs_tuned" in axes:
        tabpfn = axes["default_vs_tuned"]["families"]["tabpfn"]
        lines.extend(
            [
                "### Axis 1: default_vs_tuned",
                "",
                "Default trees:",
                table_md(
                    headline_rows(axes["default_vs_tuned"], "families.default_trees")
                ),
                "Tuned trees:",
                table_md(
                    headline_rows(axes["default_vs_tuned"], "families.tuned_trees")
                ),
                "TabPFN:",
                table_md(
                    [
                        (
                            "tabpfn",
                            tabpfn.get("status"),
                            tabpfn.get("train", {}).get("pr_auc", 0.0),
                            tabpfn.get("val", {}).get("pr_auc", 0.0),
                            tabpfn.get("test", {}).get("pr_auc", 0.0),
                        )
                    ]
                    if "train" in tabpfn
                    else []
                ),
            ]
        )
    if "sample_efficiency_fine" in axes:
        lines.extend(["### Axis 2: sample_efficiency_fine", ""])
        for size_cell in axes["sample_efficiency_fine"]["sizes"]:
            lines.append(f"Support {size_cell['support_size']}:")
            lines.append(table_md(headline_rows(size_cell, "models")))
    if "dimensionality_at_small_n" in axes:
        lines.extend(
            [
                "### Axis 3: dimensionality_at_small_n",
                "",
            ]
        )
        for dim_cell in axes["dimensionality_at_small_n"]["feature_dimensions"]:
            lines.append(
                f"{dim_cell['feature_set']} ({dim_cell['n_features']} features):"
            )
            lines.append(table_md(headline_rows(dim_cell, "models")))
    if "stability_multiseed" in axes:
        lines.extend(
            [
                "### Axis 4: stability_multiseed",
                "",
                "| model | train PR-AUC mean/std | val PR-AUC mean/std | test PR-AUC mean/std |",
                "|---|---:|---:|---:|",
            ]
        )
        for model, summary in axes["stability_multiseed"]["summary"].items():
            train = summary["train"]["pr_auc"]
            val = summary["val"]["pr_auc"]
            test = summary["test"]["pr_auc"]
            if train["mean"] is None:
                continue
            lines.append(
                f"| {model} | {train['mean']:.6f}/{train['std']:.6f} | "
                f"{val['mean']:.6f}/{val['std']:.6f} | "
                f"{test['mean']:.6f}/{test['std']:.6f} |"
            )
    crossover = axes.get("sample_efficiency_fine", {}).get("crossover_support")
    lines.extend(
        [
            "",
            "## Observations",
            "",
            "- Axis 1 tuning used validation PR-AUC only; the held-out test half was scored after selection.",
            f"- Axis 2 crossover support: `{crossover}`.",
            "- Axis 3 records the 50-feature rule as baseline plus the first 33 value-change columns.",
            "- Axis 4 reports seed-grid variation and a separate same-input TabPFN rerun band when TabPFN is available.",
            "",
            "## Open Questions",
            "",
            "- Should tuned-tree search space be widened before any report-facing interpretation?",
            "- Should later full runs raise `--tune-trials` beyond the current handoff budget?",
            "- If TabPFN is skipped locally, rerun on the known local checkpoint/GPU path before comparing claims.",
            "",
        ]
    )
    args.handoff.parent.mkdir(parents=True, exist_ok=True)
    args.handoff.write_text("\n".join(lines), encoding="utf-8")


def run_all(args: argparse.Namespace, runner: Runner, df) -> dict[str, Any]:
    table = build_split_table(df)
    features = feature_sets(table)
    full_features = features["full_137"]
    axes: dict[str, Any] = {}
    if "default_vs_tuned" in args.axes:
        log("Axis 1: default_vs_tuned")
        axes["default_vs_tuned"] = axis_default_vs_tuned(
            runner,
            table,
            full_features,
            args,
        )
    if "sample_efficiency_fine" in args.axes:
        log("Axis 2: sample_efficiency_fine")
        axes["sample_efficiency_fine"] = axis_sample_efficiency(
            runner,
            table,
            full_features,
            args,
        )
    if "dimensionality_at_small_n" in args.axes:
        log("Axis 3: dimensionality_at_small_n")
        axes["dimensionality_at_small_n"] = axis_dimensionality(
            runner,
            table,
            features,
            args,
        )
    if "stability_multiseed" in args.axes:
        log("Axis 4: stability_multiseed")
        axes["stability_multiseed"] = axis_stability(
            runner,
            table,
            full_features,
            args,
        )
    return {
        "split": table["split"],
        "feature_sets": {
            "raw_baseline_17": {
                "n_features": 17,
                "columns": features["raw_baseline_17"],
            },
            "baseline_plus_first_33_value_change_50": {
                "n_features": 50,
                "selection_rule": (
                    "BASELINE_FEATURE_COLS plus value_cols[:33], where value_cols "
                    "are lag_value_* columns in generated DataFrame order."
                ),
                "columns": features["baseline_plus_first_33_value_change_50"],
            },
            "full_137": {
                "n_features": 137,
                "columns": features["full_137"],
            },
        },
        "axes": axes,
    }


def merge_existing_results(
    out_path: Path,
    results: dict[str, Any],
    *,
    axes_requested: list[str],
) -> dict[str, Any]:
    if not out_path.exists():
        results["merge"] = {
            "mode": "new",
            "axes_requested": list(axes_requested),
            "axes_preserved": [],
            "axes_replaced": sorted(results["axes"]),
        }
        return results
    try:
        with out_path.open(encoding="utf-8") as f:
            existing = json.load(f)
    except json.JSONDecodeError:
        results["merge"] = {
            "mode": "replace_unreadable_existing",
            "axes_requested": list(axes_requested),
            "axes_preserved": [],
            "axes_replaced": sorted(results["axes"]),
        }
        return results
    preserved = {
        name: value
        for name, value in existing.get("axes", {}).items()
        if name not in axes_requested
    }
    merged_axes = {**preserved, **results["axes"]}
    results["axes"] = {
        name: merged_axes[name] for name in AXIS_ORDER if name in merged_axes
    }
    results["merge"] = {
        "mode": "merge_preserve_unrequested_axes",
        "source_path": str(out_path),
        "axes_requested": list(axes_requested),
        "axes_preserved": sorted(preserved),
        "axes_replaced": sorted(
            name for name in results["axes"] if name in axes_requested
        ),
    }
    return results


def main() -> None:
    args = parse_args()
    smoke_adjust(args)
    args.fit_rows = min(args.fit_rows, 10_000)
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
        f"Device={runner.device} tabpfn_ok={runner.tabpfn_ok} smoke={args.smoke} "
        f"fit_rows={args.fit_rows} score_rows={args.score_rows}"
    )
    df = load_m3_frame(verbose=True)
    run = run_all(args, runner, df)
    command = (
        "uv run python scripts/run_m5_phaseD_deep_comparison.py "
        f"--out {args.out} --handoff {args.handoff} "
        f"--fit-rows {args.fit_rows} --score-rows {args.score_rows} "
        f"--scarcity-sizes {' '.join(map(str, args.scarcity_sizes))} "
        f"--tune-trials {args.tune_trials} --seed {args.seed}"
        + (" --skip-tabpfn" if args.skip_tabpfn else "")
        + (" --smoke" if args.smoke else "")
    )
    results = {
        "experiment": "m5_phaseD_deep_comparison",
        "scope": (
            "Additive M5/M6 follow-up comparison on existing GEPIII/M3 data; "
            "no report-facing artifact updated."
        ),
        "value_change_regime": VALUE_CHANGE_REGIME,
        "split": run["split"],
        "budgets": {
            "fit_rows": int(args.fit_rows),
            "score_rows": int(args.score_rows),
            "scarcity_sizes": [int(s) for s in args.scarcity_sizes],
            "tune_trials": int(args.tune_trials),
            "seed": int(args.seed),
            "downsampling_seeds": [int(s) for s in DOWNSAMPLE_SEEDS],
            "model_seeds": [int(s) for s in MODEL_SEEDS],
            "tabpfn_fit_rows_limit": 10_000,
            "smoke": bool(args.smoke),
            "axes_requested": list(args.axes),
        },
        "models": list(MODEL_ORDER),
        "feature_sets": run["feature_sets"],
        "environment": env,
        "axes": run["axes"],
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    results = merge_existing_results(
        args.out,
        results,
        axes_requested=list(args.axes),
    )
    write_json_with_provenance(
        args.out,
        results,
        root=ROOT,
        provenance={"command": command},
    )
    write_handoff(args, results, command)
    if args.smoke:
        for name, axis in results["axes"].items():
            log(f"Smoke shape {name}: keys={sorted(axis.keys())}")
    log(f"Saved {args.out} and {args.handoff} in {results['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
