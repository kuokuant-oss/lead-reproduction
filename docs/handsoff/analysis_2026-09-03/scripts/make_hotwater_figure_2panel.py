#!/usr/bin/env python3
"""hot water 誤差分析：單一雙欄圖，套用正式繪圖腳本的視覺規格。"""
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
SURFACE, INK, SECONDARY, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
RUST, TEAL, AMBER, BAND = "#c2543c", "#3d6b7d", "#d9a441", "#d8d7d0"
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
    ax.set_xscale("log")
    ax.set_xlim(88, 850)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(v) for v in BUDGETS])
    ax.minorticks_off()
    ax.set_xlabel("Source buildings (K)", fontsize=8.9, color=SECONDARY, labelpad=7)


def eb(ax, mean, err, color, marker, label, lw=1.55, msz=3.8):
    ax.errorbar(BUDGETS, mean, yerr=err, color=color, marker=marker, linewidth=lw,
                markersize=msz, markeredgewidth=0.8, elinewidth=0.9, capsize=3.0,
                capthick=0.9, zorder=3, label=label)


# ---------- (a) 資料 ----------
zero_m, zero_s, non_m, non_s, tot_m = [], [], [], [], []
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
    tot_m.append(agg([(sk(r), r["ap_tab"] - r["ap_tree"]) for r in sub])[0])
m, s = ms([r["contrib_zero"] for r in k725]); zero_m.append(m); zero_s.append(s)
m, s = ms([r["contrib_nonzero"] for r in k725]); non_m.append(m); non_s.append(s)
tot_m.append(ms([r["ap_tab"] - r["ap_tree"] for r in k725])[0])

# ---------- (b) 資料 ----------
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


GROUPS = {"C": (TEAL, "o", "> 0.60   ·  31 buildings, 83.4% of anomalies"),
          "B": (AMBER, "D", "0.10 – 0.60   ·  13 buildings, 12.9%"),
          "A": (RUST, "X", "< 0.10   ·  19 buildings, 3.7%")}
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
    m, s = ms([float(np.mean([v for b, v in r["per_building"].items() if grp(b) == g]))
               for r in k725])
    series[g][0].append(m)
    series[g][1].append(s)

# ---------- 繪圖 ----------
fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.05))
fig.patch.set_facecolor(SURFACE)

ax = axes[0]
style_axis(ax)
ax.axhline(0, color=AXIS, linewidth=0.9, zorder=1)
ax.plot(BUDGETS, tot_m, color=BAND, linewidth=5.0, solid_capstyle="round", zorder=2,
        label="Total gap")
eb(ax, zero_m, zero_s, RUST, "X", "Rows with meter reading = 0")
eb(ax, non_m, non_s, TEAL, "o", "Rows with meter reading > 0", lw=1.28, msz=3.2)
ax.set_ylim(-0.125, 0.028)
ax.set_yticks(np.arange(-0.125, 0.026, 0.025))
ax.set_ylabel("Contribution to PR-AUC gap  (TabPFN − Tree Ensemble)",
              fontsize=9.2, color=SECONDARY, labelpad=7)
ax.set_title("a   Additive decomposition of the gap", loc="left", fontsize=11.3,
             fontweight="bold", color=INK, pad=9)
leg = ax.legend(loc="lower right", bbox_to_anchor=(0.995, 0.005), frameon=False,
                fontsize=8.6, labelcolor=SECONDARY, handlelength=2.0, labelspacing=0.42)

ax = axes[1]
style_axis(ax)
ax.axhline(0, color=AXIS, linewidth=0.9, zorder=1)
for g in ("C", "B", "A"):
    color, marker, label = GROUPS[g]
    eb(ax, series[g][0], series[g][1], color, marker, label)
ax.set_ylim(-0.168, 0.030)
ax.set_yticks(np.arange(-0.150, 0.026, 0.025))
ax.set_ylabel("Mean per-building PR-AUC gap", fontsize=9.2, color=SECONDARY, labelpad=7)
ax.set_title("b   By P(anomaly | meter reading = 0) within building", loc="left",
             fontsize=11.3, fontweight="bold", color=INK, pad=9)
leg = ax.legend(loc="lower right", bbox_to_anchor=(0.995, 0.005), frameon=False,
                fontsize=8.6, labelcolor=SECONDARY, handlelength=2.0, labelspacing=0.42)

fig.suptitle("Hot water — the TabPFN deficit is confined to zero-reading anomalies",
             x=0.055, y=0.972, ha="left", fontsize=15, fontweight="bold", color=INK)
fig.text(0.055, 0.893,
         "Mean ± standard error · 50,000 training rows (25k anomaly / 25k normal) · 137 features.",
         ha="left", fontsize=9.15, color=SECONDARY)
fig.text(0.055, 0.845,
         "Hot water holdout 636,121 rows · 90,691 anomalies · 89.1% of anomalies occur at a zero "
         "reading · K=100–400 n=5 building seeds · K=725 n=5 row seeds",
         ha="left", fontsize=8.45, color=SECONDARY)
fig.subplots_adjust(left=0.055, right=0.995, top=0.735, bottom=0.135, wspace=0.19)

out = SCRATCH / "fig_hotwater_2panel.png"
tmp = out.with_name(f".{out.name}.tmp")
fig.savefig(tmp, format="png", dpi=180, facecolor=SURFACE, edgecolor="none")
plt.close(fig)
os.replace(tmp, out)
print("saved", out.name)
