#!/usr/bin/env python3
"""唯讀計算 K=50 building seed 0 的 Tree/TabPFN 指標（PR-AUC + ROC-AUC）。

只讀 COMPLETE.json / cell.json / predictions.npz，不寫入任何實驗輸出，
也不寫入既有的 metric cache。所有 identity gate 直接沿用正式繪圖腳本的函式。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PLOT = Path(
    "/mnt/c/Users/User/Documents/Codex/2026-09-01/wsl-ubuntu-m5-building-count-v5"
    "/outputs/scripts/plot_m5_v5_fixed50k_building_scarcity_roc.py"
)
ROOT = Path(
    "/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k/data/processed"
    "/m5_building_curve/v5_fixed_50k/model_runs"
)

spec = importlib.util.spec_from_file_location("m5plot", PLOT)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)

BUDGET = 50
BSEED = 0
ROW_SEEDS = (0, 1)


def cell_metrics(directory: Path, model: str) -> dict:
    with np.load(directory / "predictions.npz") as payload:
        missing = [k for k in (*m5.HOLDOUT_ARRAY_KEYS, model) if k not in payload]
        if missing:
            raise RuntimeError(f"missing arrays {missing} in {directory}")
        arrays = {k: np.asarray(payload[k]) for k in m5.HOLDOUT_ARRAY_KEYS}
        scores = np.asarray(payload[model], dtype=np.float64)
    lengths = {k: len(v) for k, v in arrays.items()}
    lengths[model] = len(scores)
    if set(lengths.values()) != {m5.HOLDOUT_ROWS}:
        raise RuntimeError(f"prediction row mismatch in {directory}: {lengths}")
    if not np.isfinite(scores).all():
        raise RuntimeError(f"non-finite scores in {directory}")

    labels = arrays["anomaly"].astype(np.int8, copy=False)
    meter = arrays["meter"].astype(np.int8, copy=False)
    pr, roc = {}, {}
    for meter_id, meter_name, _ in m5.METERS:
        keep = meter == meter_id
        y = labels[keep]
        if not 0 < int(y.sum()) < len(y):
            raise RuntimeError(f"invalid label support meter {meter_id} in {directory}")
        pr[meter_name] = float(average_precision_score(y, scores[keep]))
        roc[meter_name] = float(roc_auc_score(y, scores[keep]))
    return {
        "holdout_array_sha256": {k: m5.sha256_array(v) for k, v in arrays.items()},
        "meter_pr_auc": pr,
        "meter_roc_auc": roc,
    }


per_seed: dict[str, dict[int, dict]] = {"ensemble": {}, "tabpfn": {}}
for row_seed in ROW_SEEDS:
    label = f"k{BUDGET}_b{BSEED}_r{row_seed}"
    dirs = {
        model: m5.cell_dir(ROOT, BSEED, row_seed, BUDGET, model)
        for model in ("ensemble", "tabpfn")
    }
    meta = {
        model: m5.validate_cell(d, BUDGET, BSEED, row_seed, model)
        for model, d in dirs.items()
    }
    m5.validate_pair(meta["ensemble"], meta["tabpfn"], label)
    res = {model: cell_metrics(d, model) for model, d in dirs.items()}
    m5.require_same_holdout(res["ensemble"], res["tabpfn"], label)
    for model in ("ensemble", "tabpfn"):
        per_seed[model][row_seed] = res[model]
    print(f"[gate] {label}: cell identity OK, pair identity OK, holdout arrays identical")

print()
for metric_key, metric_name in (("meter_pr_auc", "PR-AUC"), ("meter_roc_auc", "ROC-AUC")):
    print(f"=== {metric_name} ===")
    print(f"{'meter':<15}{'model':<10}{'r0':>10}{'r1':>10}{'b0 mean':>12}{'delta':>10}")
    for _, meter_name, meter_label in m5.METERS:
        means = {}
        for model in ("ensemble", "tabpfn"):
            v0 = per_seed[model][0][metric_key][meter_name]
            v1 = per_seed[model][1][metric_key][meter_name]
            means[model] = (v0 + v1) / 2.0
            tag = "Tree" if model == "ensemble" else "TabPFN"
            print(f"{meter_label:<15}{tag:<10}{v0:>10.4f}{v1:>10.4f}{means[model]:>12.4f}", end="")
            print(f"{means['tabpfn'] - means['ensemble']:>+10.4f}" if model == "tabpfn" else "")
    print()

out = {
    "budget": BUDGET,
    "building_seed": BSEED,
    "row_seeds": list(ROW_SEEDS),
    "per_row_seed": {
        model: {
            str(rs): {
                "meter_pr_auc": per_seed[model][rs]["meter_pr_auc"],
                "meter_roc_auc": per_seed[model][rs]["meter_roc_auc"],
            }
            for rs in ROW_SEEDS
        }
        for model in ("ensemble", "tabpfn")
    },
}
print(json.dumps(out, indent=2, sort_keys=True))
