#!/usr/bin/env python3
"""Hot-water PR-AUC by month at K = 725.

圖要回答的問題（作圖規範 v0.3 §1）：
    在完整來源池（K=725）下，Tree Ensemble 與 TabPFN 的 hot water PR-AUC
    如何隨月份變化，何時各自領先？

依 v0.3：§2 骨架（名詞片語主標題、單行 subtitle、圖底置中無框圖例）、
§3 字級 16/10.5、§4 真實數值 x 位置與 8% headroom、1/2/5 刻度、
§5 mean 搭配 SE、§6 只留水平 hairline 格線、線寬/marker 依對角線縮放、
§7.2 模型登記色（Tree Ensemble #0b0b0b pentagon、TabPFN #d1498b X）、
§9.3 手動 subplots_adjust + facecolor=surface。
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PLOT = Path(
    "/mnt/c/Users/User/Documents/Codex/2026-09-01/wsl-ubuntu-m5-building-count-v5"
    "/outputs/scripts/plot_m5_v5_fixed50k_building_scarcity_roc.py"
)
REPO = Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k")
SCRATCH = Path(
    "/mnt/c/Users/User/AppData/Local/Temp/claude/C--Users-User-projects-lead-reproduction-temp"
    "/4888b221-2e78-4bf8-bfd6-7f21071cc05b/scratchpad"
)
OUTDIR = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff/analysis_2026-09-03")
HOT_WATER = 3

# §7.1 / §7.2 tokens
PRIMARY_INK, SECONDARY_INK, MUTED_INK = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
TREE_C, TABPFN_C = "#0b0b0b", "#d1498b"

FIGSIZE = (9.4, 5.6)
SCALE = math.hypot(*FIGSIZE) / math.hypot(9.4, 7.6)
LW, MK, GW = 1.0 * SCALE, 5.6 * SCALE, 0.7 * SCALE

spec = importlib.util.spec_from_file_location("m5plot", PLOT)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)
formal_root = REPO / "data/processed/m5_building_curve/v5_fixed_50k/model_runs"
ext_root = REPO / "data/processed/m5_building_curve/v5_fixed_50k_k725_row_seed_extension/model_runs"
colab = REPO / "data/processed/m5_building_curve/v5_fixed_50k_k725_colab"
_, tree_dirs = m5.validate_k725(REPO, formal_root, ext_root,
                                colab / "model_results/k725_five_seed_summary.json")
print("[gate] validate_k725 通過")

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y = side["y"].astype(np.int8)
ts = pd.to_datetime(np.load(SCRATCH / "hotwater_time.npz")["ts"])
month = ts.month.to_numpy()

per_seed_tree, per_seed_tab = [], []
for rs in (0, 1, 2, 3, 4):
    with np.load(tree_dirs[rs] / "predictions.npz") as z:
        keep = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        st = np.asarray(z["ensemble"], dtype=np.float64)[keep]
        if not np.array_equal(np.asarray(z["anomaly"])[keep].astype(np.int8), y):
            raise RuntimeError("holdout 不符")
    tab = (m5.cell_dir(formal_root, 725, rs, 725, "tabpfn") / "predictions.npz"
           if rs < 2 else colab / f"model_results/row_seed{rs}/predictions.npz")
    with np.load(tab) as z:
        k2 = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        sp = np.asarray(z["tabpfn"], dtype=np.float64)[k2]
        if not np.array_equal(np.asarray(z["anomaly"])[k2].astype(np.int8), y):
            raise RuntimeError("holdout 不符")
    per_seed_tree.append([average_precision_score(y[month == m], st[month == m])
                          for m in range(1, 13)])
    per_seed_tab.append([average_precision_score(y[month == m], sp[month == m])
                         for m in range(1, 13)])

tree = np.array(per_seed_tree)
tab = np.array(per_seed_tab)
tree_m, tree_se = tree.mean(0), tree.std(0, ddof=1) / math.sqrt(5)
tab_m, tab_se = tab.mean(0), tab.std(0, ddof=1) / math.sqrt(5)
d = tab - tree
d_m, d_se = d.mean(0), d.std(0, ddof=1) / math.sqrt(5)

x = np.arange(1, 13)
LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax = plt.subplots(figsize=FIGSIZE)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(BASELINE)
    ax.spines[s].set_linewidth(GW)
ax.grid(axis="y", color=GRID, linewidth=GW)
ax.set_axisbelow(True)
ax.tick_params(colors=SECONDARY_INK, labelsize=9.5, length=3.0, width=GW)

# 兩線之間的淡色填充，讓「誰在上面」一眼可辨（非資料系列，不進圖例）
ax.fill_between(x, tree_m, tab_m, color=MUTED_INK, alpha=0.13, linewidth=0, zorder=1)

ax.errorbar(x, tree_m, yerr=tree_se, color=TREE_C, marker="p", linewidth=LW * 1.35,
            markersize=MK, markeredgewidth=0.8, elinewidth=LW * 0.9, capsize=2.6,
            capthick=LW * 0.9, zorder=3, label="Tree Ensemble")
ax.errorbar(x, tab_m, yerr=tab_se, color=TABPFN_C, marker="X", linewidth=LW,
            markersize=MK * 0.95, markeredgewidth=0.8, elinewidth=LW * 0.9, capsize=2.6,
            capthick=LW * 0.9, zorder=4, label="TabPFN")

lo = float(min((tab_m - tab_se).min(), (tree_m - tree_se).min()))
hi = float(max((tab_m + tab_se).max(), (tree_m + tree_se).max()))
pad = 0.08 * (hi - lo)
ax.set_ylim(lo - pad, min(1.0, hi + pad))
ax.set_yticks(np.arange(0.5, 1.001, 0.1))
ax.set_xlim(0.4, 12.6)
ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_ylabel("PR-AUC", fontsize=10.5, color=SECONDARY_INK, labelpad=8)
ax.set_xlabel("Month of 2016", fontsize=10.5, color=SECONDARY_INK, labelpad=8)

fig.suptitle("Hot-water PR-AUC by month at K = 725",
             x=0.070, y=0.955, ha="left", fontsize=16, fontweight="bold",
             color=PRIMARY_INK)
fig.text(0.070, 0.888,
         "Hot-water holdout split by calendar month · 2016, hourly · "
         "mean ± standard error over 5 row seeds",
         ha="left", fontsize=10.5, color=SECONDARY_INK)
leg = fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=2, frameon=False,
                 fontsize=10.5, labelcolor=SECONDARY_INK, handlelength=2.2)

fig.subplots_adjust(left=0.088, right=0.985, top=0.800, bottom=0.185)

OUTDIR.mkdir(parents=True, exist_ok=True)
out = OUTDIR / "m5_hot_water_monthly_pr_auc_k725.png"
tmp = out.with_name(f".{out.name}.tmp")
fig.savefig(tmp, format="png", dpi=180, facecolor=SURFACE, edgecolor="none")
plt.close(fig)
os.replace(tmp, out)
print(f"[saved] {out}")

print(f"\n{'月':<5}{'Tree':>18}{'TabPFN':>18}{'差':>18}")
for i, lab in enumerate(LABELS):
    print(f"{lab:<5}{tree_m[i]:>10.4f} ± {tree_se[i]:.4f}"
          f"{tab_m[i]:>10.4f} ± {tab_se[i]:.4f}"
          f"{d_m[i]:>+10.4f} ± {d_se[i]:.4f}")
wins = int((d_m > 0).sum())
print(f"\nTabPFN 領先的月份數: {wins}/12   Tree 領先: {12 - wins}/12")
