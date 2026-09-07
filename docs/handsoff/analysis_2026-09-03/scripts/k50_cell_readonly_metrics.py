#!/usr/bin/env python3
"""唯讀計算指定 cell 的 Tree/TabPFN 指標（PR-AUC + ROC-AUC）。

用法: python k50_cell_readonly_metrics.py <budget> <building_seed> <row_seeds,逗號分隔>
只讀 COMPLETE.json / cell.json / predictions.npz，不寫入任何實驗輸出或 metric cache。
identity gate 直接沿用正式繪圖腳本的函式。
"""
from __future__ import annotations

import importlib.util
import json
import sys
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

BUDGET = int(sys.argv[1])
BSEED = int(sys.argv[2])
ROW_SEEDS = tuple(int(x) for x in sys.argv[3].split(","))


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
    print(f"=== {metric_name} (K={BUDGET}, building seed {BSEED}) ===")
    header = "".join(f"{'r' + str(rs):>10}" for rs in ROW_SEEDS)
    print(f"{'meter':<15}{'model':<10}{header}{'mean':>12}{'delta':>10}")
    for _, meter_name, meter_label in m5.METERS:
        means = {}
        for model in ("ensemble", "tabpfn"):
            vals = [per_seed[model][rs][metric_key][meter_name] for rs in ROW_SEEDS]
            means[model] = sum(vals) / len(vals)
            tag = "Tree" if model == "ensemble" else "TabPFN"
            cols = "".join(f"{v:>10.4f}" for v in vals)
            print(f"{meter_label:<15}{tag:<10}{cols}{means[model]:>12.4f}", end="")
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
