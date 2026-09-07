#!/usr/bin/env python3
"""hot water 誤差分析的兩張圖（K=100–725）。輸出到 scratchpad，不動任何實驗檔案。"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCRATCH = Path(
    "/mnt/c/Users/User/AppData/Local/Temp/claude/C--Users-User-projects-lead-reproduction-temp"
    "/4888b221-2e78-4bf8-bfd6-7f21071cc05b/scratchpad"
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TABPFN = "#d1498b"
ZERO_C = "#d1498b"
NONZERO_C = "#8fb6c4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.4,
    "axes.linewidth": 0.7,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

BUDGETS = [100, 200, 400, 725]
attrib = json.loads((SCRATCH / "hotwater_ap_attribution.json").read_text())
gapdata = json.loads((SCRATCH / "hotwater_gap_by_budget.json").read_text())
side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y, bid, reading = side["y"], side["bid"], side["reading"]


def bseed_mean(pairs):
    by_b = {}
    for (b, _), v in pairs:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        by_b.setdefault(b, []).append(v)
    return float(np.mean([np.mean(v) for v in by_b.values()])) if by_b else float("nan")


def bseed_se(pairs):
    by_b = {}
    for (b, _), v in pairs:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        by_b.setdefault(b, []).append(v)
    means = [float(np.mean(v)) for v in by_b.values()]
    if len(means) < 2:
        return float("nan")
    return float(np.std(means, ddof=1) / math.sqrt(len(means)))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side_ in ("top", "right"):
        ax.spines[side_].set_visible(False)
    for side_ in ("left", "bottom"):
        ax.spines[side_].set_color(AXIS)
        ax.spines[side_].set_linewidth(0.7)
    ax.tick_params(colors=SECONDARY, length=2.4, width=0.7, labelsize=6.8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


# ============================ 圖 1 ============================
zero_c, nonzero_c, total_g, total_se = [], [], [], []
for k in BUDGETS:
    sub = [r for r in attrib if r["budget"] == k]
    sk = lambda r: (r["building_seed"], r["row_seed"])
    zero_c.append(bseed_mean([(sk(r), r["reading_bin"].get("zero", {"delta": 0})["delta"])
                              for r in sub]))
    nonzero_c.append(bseed_mean(
        [(sk(r), sum(v["delta"] for g, v in r["reading_bin"].items() if g != "zero"))
         for r in sub]))
    total_g.append(bseed_mean([(sk(r), r["ap_tab"] - r["ap_tree"]) for r in sub]))
    total_se.append(bseed_se([(sk(r), r["ap_tab"] - r["ap_tree"]) for r in sub]))

fig1, ax = plt.subplots(figsize=(3.35, 2.45), dpi=300)
style(ax)
x = np.arange(len(BUDGETS))
zero_c = np.array(zero_c)
nonzero_c = np.array(nonzero_c)
ax.bar(x, zero_c, width=0.52, color=ZERO_C, edgecolor="none",
       label="meter reading = 0  (89.1% of anomalies)")
ax.bar(x, nonzero_c, width=0.52, bottom=np.where(zero_c < 0, zero_c, 0),
       color=NONZERO_C, edgecolor="none", label="meter reading > 0")
ax.errorbar(x, total_g, yerr=total_se, fmt="o", color=INK, markersize=3.0,
            elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=5, label="total gap ± SE")
ax.axhline(0, color=INK, linewidth=0.7)
for i, (z, t) in enumerate(zip(zero_c, total_g)):
    ax.annotate(f"{z / t * 100:.0f}%", (i, z), textcoords="offset points",
                xytext=(0, -9), ha="center", fontsize=6.2, color=SECONDARY)
ax.set_xticks(x)
ax.set_xticklabels([f"K={k}" for k in BUDGETS])
ax.set_ylabel("PR-AUC gap  (TabPFN − Tree)", fontsize=7.2, color=INK)
ax.set_title("Hot water: the deficit is confined to zero-reading anomalies",
             fontsize=7.6, color=INK, pad=6, loc="left")
leg = ax.legend(frameon=False, fontsize=6.3, loc="lower right", handlelength=1.2,
                borderpad=0.2, labelspacing=0.3)
for t in leg.get_texts():
    t.set_color(SECONDARY)
ax.text(0.0, -0.30, "n = 5 building seeds × 2 row seeds per K; K=725 has 2 row seeds only.\n"
                    "Percentages give the zero-reading share of the total gap.",
        transform=ax.transAxes, fontsize=5.8, color=SECONDARY, va="top")
fig1.tight_layout()
fig1.savefig(SCRATCH / "fig1_hotwater_gap_source.png", bbox_inches="tight")
print("fig1 saved")

# ============================ 圖 2 ============================
pz, npos = {}, {}
for b in np.unique(bid):
    sel = bid == b
    z = reading[sel] == 0
    p = int((y[sel] == 1).sum())
    if p == 0:
        continue
    pz[int(b)] = float((y[sel][z] == 1).mean()) if z.any() else np.nan
    npos[int(b)] = p

rows = []
for k in BUDGETS:
    sub = [r for r in gapdata if r["budget"] == k]
    keys = sorted({b for r in sub for b in r["groups"]["building"]}, key=int)
    for b in keys:
        vals = []
        for r in sub:
            g = r["groups"]["building"].get(b)
            if not g or g["pos"] == 0:
                continue
            a, t = g["ap_tree"], g["ap_tab"]
            if a is None or t is None or math.isnan(a) or math.isnan(t):
                continue
            vals.append(((r["building_seed"], r["row_seed"]), t - a))
        d = bseed_mean(vals)
        if not math.isnan(d) and int(b) in pz and not math.isnan(pz[int(b)]):
            rows.append({"K": k, "b": int(b), "pz": pz[int(b)],
                         "pos": npos[int(b)], "delta": d})
df = pd.DataFrame(rows)

panels = [400, 725]
fig2, axes = plt.subplots(1, len(panels), figsize=(6.6, 2.55), dpi=300, sharey=True)
for ax, k in zip(axes, panels):
    style(ax)
    d = df[df.K == k]
    sizes = 6 + 62 * (d["pos"] / d["pos"].max()) ** 0.55
    colors = np.where(d["pz"] < 0.10, "#c2543c", np.where(d["pz"] < 0.60, "#c9a227", TABPFN))
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.axvspan(-0.02, 0.10, color="#c2543c", alpha=0.06, lw=0)
    ax.scatter(d["pz"], d["delta"], s=sizes, c=colors, alpha=0.72,
               edgecolors="white", linewidths=0.4, zorder=3)
    rho = d[["pz", "delta"]].corr(method="spearman").iloc[0, 1]
    ax.set_title(f"K = {k}", fontsize=7.6, color=INK, loc="left", pad=5)
    ax.text(0.97, 0.05, f"Spearman ρ = {rho:+.2f}", transform=ax.transAxes,
            ha="right", fontsize=6.4, color=SECONDARY)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("P(anomaly | meter reading = 0) within building", fontsize=7.0, color=INK)
axes[0].set_ylabel("PR-AUC gap per building\n(TabPFN − Tree)", fontsize=7.0, color=INK)
axes[0].text(0.02, 0.06, "zero readings are\nnormal operation", transform=axes[0].transAxes,
             fontsize=6.0, color="#c2543c", va="bottom")
handles = [Line2D([], [], marker="o", linestyle="", color="#c2543c", markersize=4,
                  label="P < 0.10  (19 buildings, 3.7% of anomalies)"),
           Line2D([], [], marker="o", linestyle="", color="#c9a227", markersize=4,
                  label="0.10 – 0.60  (13 buildings)"),
           Line2D([], [], marker="o", linestyle="", color=TABPFN, markersize=4,
                  label="P > 0.60  (31 buildings, 83.4% of anomalies)")]
leg = fig2.legend(handles=handles, frameon=False, fontsize=6.3, ncol=3,
                  loc="lower center", bbox_to_anchor=(0.5, -0.10), handlelength=1.0)
for t in leg.get_texts():
    t.set_color(SECONDARY)
fig2.suptitle("Residual deficit sits in buildings where a zero reading is normal",
              fontsize=7.8, color=INK, x=0.012, ha="left", y=1.02)
fig2.text(0.012, -0.20, "Marker area ∝ number of anomalies in the building. "
                        "One point per building with at least one anomaly (n = 65).",
          fontsize=5.8, color=SECONDARY)
fig2.tight_layout()
fig2.savefig(SCRATCH / "fig2_hotwater_mechanism.png", bbox_inches="tight")
print("fig2 saved")

print("\n--- 圖 1 數值 ---")
for k, z, nz, t in zip(BUDGETS, zero_c, nonzero_c, total_g):
    print(f"  K={k}: zero={z:+.4f}  nonzero={nz:+.4f}  total={t:+.4f}  zero share={z / t * 100:.1f}%")
print("\n--- 圖 2 相關 ---")
for k in BUDGETS:
    d = df[df.K == k]
    print(f"  K={k}: n={len(d)}  Spearman rho={d[['pz','delta']].corr(method='spearman').iloc[0,1]:+.3f}")
