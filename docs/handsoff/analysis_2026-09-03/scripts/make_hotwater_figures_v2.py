#!/usr/bin/env python3
"""修正版圖 1（分組長條）與替代版圖 2（分群收斂曲線）。"""
from __future__ import annotations

import json
import math
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
PINK, BLUE, RED, GOLD = "#d1498b", "#8fb6c4", "#c2543c", "#c9a227"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.6,
                     "axes.linewidth": 0.6, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

BUDGETS = [100, 200, 400, 725]
attrib = json.loads((SCRATCH / "hotwater_ap_attribution.json").read_text())
gapdata = json.loads((SCRATCH / "hotwater_gap_by_budget.json").read_text())
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


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, length=2.2, width=0.6, labelsize=6.2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)


# ---------------- 圖 1（修正版）----------------
zero, nonzero, total, tse = [], [], [], []
for k in BUDGETS:
    sub = [r for r in attrib if r["budget"] == k]
    sk = lambda r: (r["building_seed"], r["row_seed"])
    zero.append(agg([(sk(r), r["reading_bin"].get("zero", {"delta": 0})["delta"]) for r in sub])[0])
    nonzero.append(agg([(sk(r), sum(v["delta"] for g, v in r["reading_bin"].items() if g != "zero"))
                        for r in sub])[0])
    m, s = agg([(sk(r), r["ap_tab"] - r["ap_tree"]) for r in sub])
    total.append(m)
    tse.append(s)

fig, ax = plt.subplots(figsize=(3.45, 2.35), dpi=320)
style(ax)
x = np.arange(4)
w = 0.30
ax.bar(x - w / 2 - 0.02, zero, width=w, color=PINK, edgecolor="none",
       label="reading = 0  (89.1% of anomalies)")
ax.bar(x + w / 2 + 0.02, nonzero, width=w, color=BLUE, edgecolor="none",
       label="reading > 0  (10.9%)")
ax.errorbar(x, total, yerr=tse, fmt="D", color=INK, markersize=2.6, elinewidth=0.75,
            capsize=1.6, capthick=0.75, zorder=6, label="total gap ± SE")
ax.axhline(0, color=INK, linewidth=0.7, zorder=4)
for i, (z, t) in enumerate(zip(zero, total)):
    ax.annotate(f"{z / t * 100:.0f}%", (i - w / 2 - 0.02, z), textcoords="offset points",
                xytext=(0, -8), ha="center", fontsize=5.9, color=SECONDARY)
ax.set_ylim(-0.118, 0.052)
ax.set_xticks(x)
ax.set_xticklabels([f"K={k}" for k in BUDGETS], fontsize=6.6)
ax.set_ylabel("PR-AUC gap  (TabPFN − Tree)", fontsize=6.8, color=INK)
leg = ax.legend(frameon=False, fontsize=5.9, loc="upper left", handlelength=1.1,
                borderpad=0.1, labelspacing=0.22, bbox_to_anchor=(-0.01, 1.04))
for t in leg.get_texts():
    t.set_color(SECONDARY)
ax.set_title("Hot water deficit is confined to zero-reading anomalies",
             fontsize=7.2, color=INK, pad=4, loc="left")
fig.text(0.0, -0.055, "Labels: zero-reading share of the total gap. n = 5 building seeds × 2 row "
                      "seeds; K=725 has 2 row seeds only.", fontsize=5.4, color=SECONDARY)
fig.tight_layout()
fig.savefig(SCRATCH / "fig1_v2.png", bbox_inches="tight")
print("fig1_v2 saved")

# ---------------- 圖 2 替代版：分群收斂 ----------------
pz = {}
for b in np.unique(bid):
    sel = bid == b
    z = reading[sel] == 0
    if int((y[sel] == 1).sum()) == 0:
        continue
    pz[int(b)] = float((y[sel][z] == 1).mean()) if z.any() else np.nan


def group_of(b):
    v = pz.get(int(b), np.nan)
    if math.isnan(v):
        return None
    return "A" if v < 0.10 else ("B" if v < 0.60 else "C")


GROUPS = {
    "A": (RED, "P(anom | reading=0) < 0.10\n19 buildings · 3.7% of anomalies"),
    "B": (GOLD, "0.10 – 0.60\n13 buildings · 12.9%"),
    "C": (PINK, "> 0.60\n31 buildings · 83.4%"),
}
series = {g: ([], []) for g in GROUPS}
for k in BUDGETS:
    sub = [r for r in gapdata if r["budget"] == k]
    for g in GROUPS:
        pairs = []
        for r in sub:
            vals = []
            for b, gd in r["groups"]["building"].items():
                if group_of(b) != g or gd["pos"] == 0:
                    continue
                a, t = gd["ap_tree"], gd["ap_tab"]
                if a is None or t is None or math.isnan(a) or math.isnan(t):
                    continue
                vals.append(t - a)
            if vals:
                pairs.append(((r["building_seed"], r["row_seed"]), float(np.mean(vals))))
        m, s = agg(pairs)
        series[g][0].append(m)
        series[g][1].append(s)

fig2, ax = plt.subplots(figsize=(3.45, 2.35), dpi=320)
style(ax)
ax.axhline(0, color=INK, linewidth=0.7, zorder=2)
for g, (color, label) in GROUPS.items():
    m = np.array(series[g][0])
    s = np.array(series[g][1])
    ax.errorbar(x, m, yerr=s, color=color, marker="o", markersize=3.2, linewidth=1.25,
                elinewidth=0.7, capsize=1.6, capthick=0.7, label=label, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([f"K={k}" for k in BUDGETS], fontsize=6.6)
ax.set_ylabel("mean per-building PR-AUC gap\n(TabPFN − Tree)", fontsize=6.8, color=INK)
leg = ax.legend(frameon=False, fontsize=5.6, loc="lower right", handlelength=1.1,
                borderpad=0.1, labelspacing=0.5)
for t in leg.get_texts():
    t.set_color(SECONDARY)
ax.set_title("Only buildings where a zero reading is normal fail to converge",
             fontsize=7.2, color=INK, pad=4, loc="left")
fig2.text(0.0, -0.055, "Mean over buildings within each group; error bars = SE across building "
                       "seeds (K=725: 2 row seeds, no SE).", fontsize=5.4, color=SECONDARY)
fig2.tight_layout()
fig2.savefig(SCRATCH / "fig2_v2_convergence.png", bbox_inches="tight")
print("fig2_v2 saved")

print("\n--- 圖 2 替代版數值（mean ± SE）---")
for g in GROUPS:
    print(f"  {g}: " + "  ".join(
        f"K{k}={m:+.4f}" + (f"±{s:.4f}" if not math.isnan(s) else "")
        for k, m, s in zip(BUDGETS, series[g][0], series[g][1])))
