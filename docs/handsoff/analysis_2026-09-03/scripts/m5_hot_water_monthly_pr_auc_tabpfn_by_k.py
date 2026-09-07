#!/usr/bin/env python3
"""TabPFN hot-water PR-AUC by month across building budgets (K = 100/200/400/725).

圖要回答的問題（作圖規範 v0.3 §1）：
    增加 context 來源建築數，對 TabPFN 的 hot water 表現在哪些月份有幫助？

四條線都是 TabPFN，K 是有序的比較維度，故使用 TabPFN 登記色相（#d1498b）的
單一色相由淺至深，全部實線、不加 marker；±SE 以同色淡帶呈現（§5）。
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
BUDGETS = [100, 200, 400, 725]

PRIMARY_INK, SECONDARY_INK = "#0b0b0b", "#52514e"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
# K 是有序的比較維度：TabPFN 單一色相由淺至深，全部實線、不加 marker
STYLE = {100: "#fbc0d6", 200: "#ec6fa4", 400: "#b02a6e", 725: "#4a1038"}

FIGSIZE = (9.4, 5.6)
SCALE = math.hypot(*FIGSIZE) / math.hypot(9.4, 7.6)
LW, MK, GW = 1.0 * SCALE, 5.6 * SCALE, 0.7 * SCALE

spec = importlib.util.spec_from_file_location("m5plot", PLOT)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)
formal_root = REPO / "data/processed/m5_building_curve/v5_fixed_50k/model_runs"
ext_root = REPO / "data/processed/m5_building_curve/v5_fixed_50k_k725_row_seed_extension/model_runs"
colab = REPO / "data/processed/m5_building_curve/v5_fixed_50k_k725_colab"
m5.validate_k725(REPO, formal_root, ext_root,
                 colab / "model_results/k725_five_seed_summary.json")
print("[gate] validate_k725 通過")

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y = side["y"].astype(np.int8)
ts = pd.to_datetime(np.load(SCRATCH / "hotwater_time.npz")["ts"])
month = ts.month.to_numpy()
masks = [month == m for m in range(1, 13)]

curves = {}
for budget in BUDGETS:
    per_rep = []
    if budget == 725:
        for rs in range(5):
            p = (m5.cell_dir(formal_root, 725, rs, 725, "tabpfn") / "predictions.npz"
                 if rs < 2 else colab / f"model_results/row_seed{rs}/predictions.npz")
            with np.load(p) as z:
                keep = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
                s = np.asarray(z["tabpfn"], dtype=np.float64)[keep]
                if not np.array_equal(np.asarray(z["anomaly"])[keep].astype(np.int8), y):
                    raise RuntimeError("holdout 不符")
            per_rep.append([average_precision_score(y[mk], s[mk]) for mk in masks])
    else:
        for b in range(5):
            rows = []
            for r in (0, 1):
                d = m5.cell_dir(formal_root, b, r, budget, "tabpfn")
                m5.validate_cell(d, budget, b, r, "tabpfn")
                with np.load(d / "predictions.npz") as z:
                    keep = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
                    s = np.asarray(z["tabpfn"], dtype=np.float64)[keep]
                    if not np.array_equal(np.asarray(z["anomaly"])[keep].astype(np.int8), y):
                        raise RuntimeError("holdout 不符")
                rows.append([average_precision_score(y[mk], s[mk]) for mk in masks])
            per_rep.append(np.mean(rows, axis=0))
    a = np.array(per_rep)
    curves[budget] = (a.mean(0), a.std(0, ddof=1) / math.sqrt(5))
    print(f"  K={budget} 完成", flush=True)

x = np.arange(1, 13)
LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax = plt.subplots(figsize=FIGSIZE)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
for s_ in ("left", "bottom"):
    ax.spines[s_].set_color(BASELINE)
    ax.spines[s_].set_linewidth(GW)
ax.grid(axis="y", color=GRID, linewidth=GW)
ax.set_axisbelow(True)
ax.tick_params(colors=SECONDARY_INK, labelsize=9.5, length=3.0, width=GW)

for budget in BUDGETS:
    m_, se_ = curves[budget]
    c = STYLE[budget]
    z = 3 + BUDGETS.index(budget)
    ax.fill_between(x, m_ - se_, m_ + se_, color=c, alpha=0.16, linewidth=0, zorder=z)
    ax.plot(x, m_, color=c, linestyle="-", linewidth=LW * 1.6,
            solid_capstyle="round", zorder=z + 10, label=f"K = {budget}")

lo = min((curves[b][0] - curves[b][1]).min() for b in BUDGETS)
hi = max((curves[b][0] + curves[b][1]).max() for b in BUDGETS)
pad = 0.08 * (hi - lo)
ax.set_ylim(lo - pad, min(1.0, hi + pad))
ax.set_yticks(np.arange(0.4, 1.001, 0.1))
ax.set_xlim(0.4, 12.6)
ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_ylabel("PR-AUC", fontsize=10.5, color=SECONDARY_INK, labelpad=8)
ax.set_xlabel("Month of 2016", fontsize=10.5, color=SECONDARY_INK, labelpad=8)

fig.suptitle("TabPFN hot-water PR-AUC by month across building budgets",
             x=0.070, y=0.955, ha="left", fontsize=16, fontweight="bold",
             color=PRIMARY_INK)
fig.text(0.070, 0.888,
         "Hot-water holdout split by calendar month · 2016, hourly · mean ± standard error "
         "over 5 seeds",
         ha="left", fontsize=10.5, color=SECONDARY_INK)
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4, frameon=False,
           fontsize=10.5, labelcolor=SECONDARY_INK, handlelength=2.6)
fig.subplots_adjust(left=0.088, right=0.985, top=0.800, bottom=0.185)

OUTDIR.mkdir(parents=True, exist_ok=True)
out = OUTDIR / "m5_hot_water_monthly_pr_auc_tabpfn_by_k.png"
tmp = out.with_name(f".{out.name}.tmp")
fig.savefig(tmp, format="png", dpi=180, facecolor=SURFACE, edgecolor="none")
plt.close(fig)
os.replace(tmp, out)
print(f"[saved] {out}")

print(f"\n{'月':<5}" + "".join(f"{'K=' + str(b):>18}" for b in BUDGETS) + f"{'100→725 增益':>14}")
for i, lab in enumerate(LABELS):
    line = f"{lab:<5}"
    for b in BUDGETS:
        line += f"{curves[b][0][i]:>10.4f} ± {curves[b][1][i]:.4f}"
    line += f"{curves[725][0][i] - curves[100][0][i]:>+14.4f}"
    print(line)
