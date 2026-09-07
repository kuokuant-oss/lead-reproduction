#!/usr/bin/env python3
"""hot water 兩張圖，套用 plot_m5_v5_fixed50k_building_scarcity_roc.py 的視覺規格。"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRATCH = Path(
    "/mnt/c/Users/User/AppData/Local/Temp/claude/C--Users-User-projects-lead-reproduction-temp"
    "/4888b221-2e78-4bf8-bfd6-7f21071cc05b/scratchpad"
)

# ---- 與正式繪圖腳本一致的視覺常數 ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
RUST = "#c2543c"
TEAL = "#3d6b7d"
AMBER = "#d9a441"

BUDGETS = [100, 200, 400, 725]
attrib = json.loads((SCRATCH / "hotwater_ap_attribution.json").read_text())
gapdata = json.loads((SCRATCH / "hotwater_gap_by_budget.json").read_text())
k725 = json.loads((SCRATCH / "k725_five_seed_hotwater.json").read_text())
side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y, bid, reading = side["y"], side["bid"], side["reading"]


def agg(pairs):
    by_b = {}
    for (b, _), v in pairs:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        by_b.setdefault(b, []).append(v)
    means = [float(np.mean(v)) for v in by_b.values()]
    if not means:
        return float("nan"), float("nan")
    if len(means) == 1:
        return means[0], float("nan")
    return float(np.mean(means)), float(np.std(means, ddof=1) / math.sqrt(len(means)))


def ms(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a)))


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def budget_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(88, 850)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(v) for v in BUDGETS])
    ax.minorticks_off()
    ax.set_xlabel("Source buildings (K)", fontsize=8.9, color=SECONDARY, labelpad=7)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    fig.savefig(tmp, format="png", dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    os.replace(tmp, path)


SERIES_STYLE = {
    "total": {"color": INK, "marker": "p", "linewidth": 1.70, "markersize": 3.6},
    "zero": {"color": RUST, "marker": "X", "linewidth": 1.45, "markersize": 3.9},
    "nonzero": {"color": TEAL, "marker": "o", "linewidth": 1.28, "markersize": 3.2},
}


def draw(ax, x, mean, err, st, label):
    ax.errorbar(x, mean, yerr=err, color=st["color"], marker=st["marker"],
                linewidth=st["linewidth"], markersize=st["markersize"],
                markeredgewidth=0.8, elinewidth=0.9, capsize=3.0, capthick=0.9,
                zorder=3, label=label)


# ============================ 圖 1 ============================
zero_m, zero_s, non_m, non_s, tot_m, tot_s = [], [], [], [], [], []
for k in BUDGETS[:3]:
    sub = [r for r in attrib if r["budget"] == k]
    sk = lambda r: (r["building_seed"], r["row_seed"])
    m, s = agg([(sk(r), r["reading_bin"].get("zero", {"delta": 0})["delta"]) for r in sub])
    zero_m.append(m)
    zero_s.append(s)
    m, s = agg([(sk(r), sum(v["delta"] for g, v in r["reading_bin"].items() if g != "zero"))
                for r in sub])
    non_m.append(m)
    non_s.append(s)
    m, s = agg([(sk(r), r["ap_tab"] - r["ap_tree"]) for r in sub])
    tot_m.append(m)
    tot_s.append(s)
for arr_m, arr_s, key in ((zero_m, zero_s, "contrib_zero"), (non_m, non_s, "contrib_nonzero")):
    m, s = ms([r[key] for r in k725])
    arr_m.append(m)
    arr_s.append(s)
m, s = ms([r["ap_tab"] - r["ap_tree"] for r in k725])
tot_m.append(m)
tot_s.append(s)

fig, ax = plt.subplots(figsize=(7.6, 5.05))
fig.patch.set_facecolor(SURFACE)
style_axis(ax)
budget_axis(ax)
ax.axhline(0, color=AXIS, linewidth=0.9, zorder=1)
draw(ax, BUDGETS, tot_m, tot_s, SERIES_STYLE["total"], "Total gap")
draw(ax, BUDGETS, zero_m, zero_s, SERIES_STYLE["zero"],
     "Contribution of meter reading = 0")
draw(ax, BUDGETS, non_m, non_s, SERIES_STYLE["nonzero"],
     "Contribution of meter reading > 0")
ax.set_ylim(-0.125, 0.028)
ax.set_yticks(np.arange(-0.125, 0.026, 0.025))
ax.set_ylabel("PR-AUC gap  (TabPFN − Tree Ensemble)", fontsize=9.2, color=SECONDARY, labelpad=7)
fig.suptitle("Hot water — the PR-AUC gap is carried entirely by zero-reading anomalies",
             x=0.075, y=0.972, ha="left", fontsize=15, fontweight="bold", color=INK)
fig.text(0.075, 0.890,
         "Additive decomposition of the gap · mean ± standard error · "
         "50,000 training rows (25k anomaly / 25k normal) · 137 features.",
         ha="left", fontsize=9.15, color=SECONDARY)
fig.text(0.075, 0.840,
         "Hot water holdout 636,121 rows · 90,691 anomalies · 89.1% of anomalies have "
         "meter reading = 0 · K=100–400 n=5 building seeds · K=725 n=5 row seeds",
         ha="left", fontsize=8.45, color=SECONDARY)
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=3, frameon=False,
           fontsize=9.2, labelcolor=SECONDARY)
fig.subplots_adjust(left=0.105, right=0.985, top=0.735, bottom=0.20)
save(fig, SCRATCH / "fig1_house.png")
print("fig1_house saved")

# ============================ 圖 2 ============================
pz = {}
for b in np.unique(bid):
    sel = bid == b
    z = reading[sel] == 0
    if int((y[sel] == 1).sum()) == 0:
        continue
    pz[int(b)] = float((y[sel][z] == 1).mean()) if z.any() else float("nan")


def grp(b):
    v = pz.get(int(b), float("nan"))
    if math.isnan(v):
        return None
    return "A" if v < 0.10 else ("B" if v < 0.60 else "C")


GROUPS = {
    "C": (TEAL, "o", "P(anomaly | reading = 0) > 0.60   ·  31 buildings, 83.4% of anomalies"),
    "B": (AMBER, "D", "0.10 – 0.60   ·  13 buildings, 12.9%"),
    "A": (RUST, "X", "< 0.10   ·  19 buildings, 3.7%"),
}
series = {g: ([], []) for g in GROUPS}
for k in BUDGETS[:3]:
    sub = [r for r in gapdata if r["budget"] == k]
    for g in GROUPS:
        pairs = []
        for r in sub:
            vals = [gd["ap_tab"] - gd["ap_tree"] for b, gd in r["groups"]["building"].items()
                    if grp(b) == g and gd["pos"] > 0 and gd["ap_tree"] is not None
                    and gd["ap_tab"] is not None and not math.isnan(gd["ap_tree"])
                    and not math.isnan(gd["ap_tab"])]
            if vals:
                pairs.append(((r["building_seed"], r["row_seed"]), float(np.mean(vals))))
        m, s = agg(pairs)
        series[g][0].append(m)
        series[g][1].append(s)
for g in GROUPS:
    per_seed = [float(np.mean([v for b, v in r["per_building"].items() if grp(b) == g]))
                for r in k725]
    m, s = ms(per_seed)
    series[g][0].append(m)
    series[g][1].append(s)

fig2, ax = plt.subplots(figsize=(7.6, 5.05))
fig2.patch.set_facecolor(SURFACE)
style_axis(ax)
budget_axis(ax)
ax.axhline(0, color=AXIS, linewidth=0.9, zorder=1)
for g in ("C", "B", "A"):
    color, marker, label = GROUPS[g]
    ax.errorbar(BUDGETS, series[g][0], yerr=series[g][1], color=color, marker=marker,
                linewidth=1.55, markersize=3.8, markeredgewidth=0.8, elinewidth=0.9,
                capsize=3.0, capthick=0.9, zorder=3, label=label)
ax.set_ylim(-0.168, 0.030)
ax.set_yticks(np.arange(-0.150, 0.026, 0.025))
ax.set_ylabel("Mean per-building PR-AUC gap  (TabPFN − Tree Ensemble)",
              fontsize=9.2, color=SECONDARY, labelpad=7)
fig2.suptitle("Only buildings where a zero reading is normal retain a deficit",
              x=0.075, y=0.972, ha="left", fontsize=15, fontweight="bold", color=INK)
fig2.text(0.075, 0.890,
          "Buildings grouped by the within-building probability that a zero reading is an "
          "anomaly · mean ± standard error.",
          ha="left", fontsize=9.15, color=SECONDARY)
fig2.text(0.075, 0.840,
          "65 buildings with at least one hot-water anomaly · "
          "K=100–400 n=5 building seeds · K=725 n=5 row seeds",
          ha="left", fontsize=8.45, color=SECONDARY)
fig2.legend(loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=1, frameon=False,
            fontsize=9.2, labelcolor=SECONDARY, handlelength=2.2, labelspacing=0.45)
fig2.subplots_adjust(left=0.105, right=0.985, top=0.735, bottom=0.255)
save(fig2, SCRATCH / "fig2_house.png")
print("fig2_house saved")

print("\n圖 1")
for k, a, b, c in zip(BUDGETS, tot_m, zero_m, non_m):
    print(f"  K={k}: total={a:+.4f} zero={b:+.4f} nonzero={c:+.4f}")
print("圖 2")
for g in ("C", "B", "A"):
    print(f"  {g}: " + "  ".join(f"K{k}={m:+.4f}±{s:.4f}"
                                 for k, m, s in zip(BUDGETS, series[g][0], series[g][1])))
