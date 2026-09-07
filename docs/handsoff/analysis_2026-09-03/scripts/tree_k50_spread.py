#!/usr/bin/env python3
"""唯讀：K=50 Tree 在五個 building seed 的表現與離散程度。

只讀 cell.json / COMPLETE.json / predictions.npz，不寫入任何檔案。
identity gate 沿用正式繪圖腳本的 validate_cell。
"""
from __future__ import annotations

import importlib.util
import math
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
MODEL = "ensemble"


def cell_metrics(directory: Path) -> dict:
    with np.load(directory / "predictions.npz") as payload:
        arrays = {k: np.asarray(payload[k]) for k in m5.HOLDOUT_ARRAY_KEYS}
        scores = np.asarray(payload[MODEL], dtype=np.float64)
    lengths = {k: len(v) for k, v in arrays.items()}
    lengths[MODEL] = len(scores)
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
    return {"meter_pr_auc": pr, "meter_roc_auc": roc}


cells: dict[int, dict[int, dict]] = {}
holdout_digests = set()
for bseed in (0, 1, 2, 3, 4):
    cells[bseed] = {}
    for rseed in (0, 1):
        d = m5.cell_dir(ROOT, bseed, rseed, BUDGET, MODEL)
        meta = m5.validate_cell(d, BUDGET, bseed, rseed, MODEL)
        holdout_digests.add(meta["holdout_row_sha256"])
        cells[bseed][rseed] = cell_metrics(d)
    print(f"[gate] k50_b{bseed}: r0/r1 cell identity OK")
print(f"[gate] holdout digest 一致: {sorted(holdout_digests)}")
print()

for metric_key, metric_name in (("meter_pr_auc", "PR-AUC"), ("meter_roc_auc", "ROC-AUC")):
    print(f"=== Tree K=50 {metric_name} ===")
    print(f"{'meter':<15}" + "".join(f"{'b' + str(b):>9}" for b in range(5))
          + f"{'mean':>10}{'SE':>9}{'SD':>9}{'min':>9}{'max':>9}{'range':>9}")
    for _, meter_name, meter_label in m5.METERS:
        seed_means = [
            (cells[b][0][metric_key][meter_name] + cells[b][1][metric_key][meter_name]) / 2.0
            for b in range(5)
        ]
        arr = np.asarray(seed_means, dtype=np.float64)
        sd = float(arr.std(ddof=1))
        se = sd / math.sqrt(len(arr))
        print(f"{meter_label:<15}" + "".join(f"{v:>9.4f}" for v in seed_means)
              + f"{arr.mean():>10.4f}{se:>9.4f}{sd:>9.4f}"
              + f"{arr.min():>9.4f}{arr.max():>9.4f}{arr.max() - arr.min():>9.4f}")
    print()

print("=== 每個 building seed 內 row seed 的差距 (|r0 - r1|) ===")
for metric_key, metric_name in (("meter_pr_auc", "PR-AUC"), ("meter_roc_auc", "ROC-AUC")):
    print(f"-- {metric_name} --")
    print(f"{'meter':<15}" + "".join(f"{'b' + str(b):>9}" for b in range(5)) + f"{'mean':>10}")
    for _, meter_name, meter_label in m5.METERS:
        gaps = [
            abs(cells[b][0][metric_key][meter_name] - cells[b][1][metric_key][meter_name])
            for b in range(5)
        ]
        print(f"{meter_label:<15}" + "".join(f"{g:>9.4f}" for g in gaps)
              + f"{float(np.mean(gaps)):>10.4f}")
    print()

print("=== 逐格原始值 ===")
for _, meter_name, meter_label in m5.METERS:
    print(f"-- {meter_label} --")
    for b in range(5):
        for r in (0, 1):
            pr = cells[b][r]["meter_pr_auc"][meter_name]
            roc = cells[b][r]["meter_roc_auc"][meter_name]
            print(f"  b{b}r{r}  PR-AUC={pr:.6f}  ROC-AUC={roc:.6f}")
