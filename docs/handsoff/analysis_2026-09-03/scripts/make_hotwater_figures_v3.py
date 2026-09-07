#!/usr/bin/env python3
"""最終兩張圖：K=725 使用五個 row seed。"""
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
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.6, "axes.linewidth": 0.6,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})

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


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, length=2.2, width=0.6, labelsize=6.2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)


# ---------------- 圖 1 ----------------
zero, nonzero, total, tse = [], [], [], []
for k in BUDGETS[:3]:
    sub = [r for r in attrib if r["budget"] == k]
    sk = lambda r: (r["building_seed"], r["row_seed"])
    zero.append(agg([(sk(r), r["reading_bin"].get("zero", {"delta": 0})["delta"]) for r in sub])[0])
    nonzero.append(agg([(sk(r), sum(v["delta"] for g, v in r["reading_bin"].items() if g != "zero"))
                        for r in sub])[0])
    m, s = agg([(sk(r), r["ap_tab"] - r["ap_tree"]) for r in sub])
    total.append(m)
    tse.append(s)
zero.append(ms([r["contrib_zero"] for r in k725])[0])
nonzero.append(ms([r["contrib_nonzero"] for r in k725])[0])
m725, s725 = ms([r["ap_tab"] - r["ap_tree"] for r in k725])
total.append(m725)
tse.append(s725)

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
for i in range(3):
    ax.annotate(f"{zero[i] / total[i] * 100:.0f}%", (i - w / 2 - 0.02, zero[i]),
                textcoords="offset points", xytext=(0, -8), ha="center",
                fontsize=5.9, color=SECONDARY)
ax.annotate("n.s.", (3, total[3]), textcoords="offset points", xytext=(11, -1),
            ha="left", va="center", fontsize=5.9, color=SECONDARY)
ax.set_ylim(-0.118, 0.052)
ax.set_xticks(x)
ax.set_xticklabels([f"K={k}" for k in BUDGETS], fontsize=6.6)
ax.set_ylabel("PR-AUC gap  (TabPFN − Tree)", fontsize=6.8, color=INK)
leg = ax.legend(frameon=False, fontsize=5.9, loc="upper left", handlelength=1.1,
                borderpad=0.1, labelspacing=0.22, bbox_to_anchor=(-0.01, 1.02))
for t in leg.get_texts():
    t.set_color(SECONDARY)
ax.set_title("Hot water deficit is confined to zero-reading anomalies",
             fontsize=7.2, color=INK, pad=5, loc="left")
fig.text(0.0, -0.055, "Labels: zero-reading share of the total gap. K ≤ 400: n = 5 building "
                      "seeds × 2 row seeds; K = 725: n = 5 row seeds.", fontsize=5.4,
         color=SECONDARY)
fig.tight_layout()
fig.savefig(SCRATCH / "fig1_final.png", bbox_inches="tight")
print("fig1_final saved")

# ---------------- 圖 2 ----------------
pz = {}
for b in np.unique(bid):
    sel = bid == b
    z = reading[sel] == 0
    if int((y[sel] == 1).sum()) == 0:
        continue
    pz[int(b)] = float((y[sel][z] == 1).mean()) if z.any() else np.nan


def grp(b):
    v = pz.get(int(b), float("nan"))
    if math.isnan(v):
        return None
    return "A" if v < 0.10 else ("B" if v < 0.60 else "C")


GROUPS = {"A": (RED, "P(anomaly | reading = 0) < 0.10\n19 buildings · 3.7% of anomalies"),
          "B": (GOLD, "0.10 – 0.60\n13 buildings · 12.9%"),
          "C": (PINK, "> 0.60\n31 buildings · 83.4%")}
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
    per_seed = []
    for r in k725:
        vals = [v for b, v in r["per_building"].items() if grp(b) == g]
        if vals:
            per_seed.append(float(np.mean(vals)))
    m, s = ms(per_seed)
    series[g][0].append(m)
    series[g][1].append(s)

fig2, ax = plt.subplots(figsize=(3.45, 2.35), dpi=320)
style(ax)
ax.axhline(0, color=INK, linewidth=0.7, zorder=2)
for g, (color, label) in GROUPS.items():
    ax.errorbar(x, series[g][0], yerr=series[g][1], color=color, marker="o", markersize=3.2,
                linewidth=1.25, elinewidth=0.7, capsize=1.6, capthick=0.7, label=label, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([f"K={k}" for k in BUDGETS], fontsize=6.6)
ax.set_xlim(-0.25, 3.55)
ax.set_ylabel("mean per-building PR-AUC gap\n(TabPFN − Tree)", fontsize=6.8, color=INK)
leg = ax.legend(frameon=False, fontsize=5.6, loc="lower right", handlelength=1.1,
                borderpad=0.1, labelspacing=0.45, bbox_to_anchor=(1.02, -0.02))
for t in leg.get_texts():
    t.set_color(SECONDARY)
ax.set_title("At full source only buildings where zero is normal keep a deficit",
             fontsize=7.2, color=INK, pad=5, loc="left")
fig2.text(0.0, -0.055, "Mean over buildings within a group; error bars = SE across replicates "
                       "(K ≤ 400: 5 building seeds; K = 725: 5 row seeds).",
          fontsize=5.4, color=SECONDARY)
fig2.tight_layout()
fig2.savefig(SCRATCH / "fig2_final.png", bbox_inches="tight")
print("fig2_final saved")

print("\n圖 1 數值")
for k, z_, nz, t, s in zip(BUDGETS, zero, nonzero, total, tse):
    share = f"{z_ / t * 100:.0f}%" if abs(t) > 1e-9 else "-"
    print(f"  K={k}: zero={z_:+.4f} nonzero={nz:+.4f} total={t:+.4f}±{s:.4f} share={share}")
print("\n圖 2 數值")
for g in GROUPS:
    print(f"  {g}: " + "  ".join(f"K{k}={m:+.4f}±{s:.4f}"
                                 for k, m, s in zip(BUDGETS, series[g][0], series[g][1])))
