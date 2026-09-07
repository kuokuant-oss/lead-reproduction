#!/usr/bin/env python3
"""預算每個 cell 的「逐正樣本 AP 貢獻差」，之後任何分組都能即時重組。

AP = (1/P) * Σ_positives precision@(該正樣本 tie-group 門檻)，故貢獻可加，
逐正樣本的 (TabPFN − Tree) 差值加總 = 全域 AP 落差（程式內以 1e-9 檢核）。
唯讀讀取 predictions.npz；輸出只寫 scratchpad。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

PLOT = Path(
    "/mnt/c/Users/User/Documents/Codex/2026-09-01/wsl-ubuntu-m5-building-count-v5"
    "/outputs/scripts/plot_m5_v5_fixed50k_building_scarcity_roc.py"
)
ROOT = Path(
    "/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k/data/processed"
    "/m5_building_curve/v5_fixed_50k/model_runs"
)
SCRATCH = Path(
    "/mnt/c/Users/User/AppData/Local/Temp/claude/C--Users-User-projects-lead-reproduction-temp"
    "/4888b221-2e78-4bf8-bfd6-7f21071cc05b/scratchpad"
)
HOT_WATER = 3

spec = importlib.util.spec_from_file_location("m5plot", PLOT)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y = side["y"].astype(np.int8)
n_pos = int(y.sum())
pos_idx = np.nonzero(y == 1)[0]

BUDGETS = [int(a) for a in sys.argv[1:]] or [400]
CELLS = []
for k in BUDGETS:
    if k == 725:
        continue
    CELLS += [(k, b, r) for b in range(5) for r in (0, 1)]


def per_positive_precision(scores):
    order = np.argsort(-scores, kind="mergesort")
    s, yy = scores[order], y[order]
    tp = np.cumsum(yy)
    cnt = np.arange(1, len(yy) + 1)
    bnd = np.nonzero(np.r_[s[:-1] != s[1:], True])[0]
    last = bnd[np.searchsorted(bnd, np.arange(len(s)))]
    out = np.zeros(len(scores))
    out[order] = np.where(yy == 1, tp[last] / cnt[last] / n_pos, 0.0)
    return out


rows, contrib = [], []
for budget, bseed, rseed in CELLS:
    label = f"k{budget}_b{bseed}_r{rseed}"
    dirs = {m: m5.cell_dir(ROOT, bseed, rseed, budget, m) for m in ("ensemble", "tabpfn")}
    if not all(m5.is_complete(d) for d in dirs.values()):
        print(f"skip {label}: 尚未完成")
        continue
    meta = {m: m5.validate_cell(d, budget, bseed, rseed, m) for m, d in dirs.items()}
    m5.validate_pair(meta["ensemble"], meta["tabpfn"], label)
    c = {}
    for model, d in dirs.items():
        with np.load(d / "predictions.npz") as payload:
            keep = np.asarray(payload["meter"]).astype(np.int8) == HOT_WATER
            s = np.asarray(payload[model], dtype=np.float64)[keep]
        cc = per_positive_precision(s)
        ref = average_precision_score(y, s)
        if abs(cc.sum() - ref) > 1e-9:
            raise RuntimeError(f"AP 拆解檢核失敗 {label} {model}")
        c[model] = cc
    delta = c["tabpfn"] - c["ensemble"]
    rows.append((budget, bseed, rseed))
    contrib.append(delta[pos_idx].astype(np.float32))
    print(f"  {label}: gap={delta.sum():+.5f}  (檢核通過)")

out = SCRATCH / "positive_contrib.npz"
np.savez_compressed(out, cells=np.array(rows, dtype=np.int32),
                    contrib=np.vstack(contrib), pos_idx=pos_idx.astype(np.int64))
print(f"\n[saved] {out.name}  shape={np.vstack(contrib).shape}")
