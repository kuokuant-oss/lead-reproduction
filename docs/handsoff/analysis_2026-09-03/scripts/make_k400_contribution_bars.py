#!/usr/bin/env python3
"""K=400：hot water PR-AUC 落差的類別貢獻（水平長條，共用 x 軸）。"""
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
RUST, TEAL = "#c2543c", "#3d6b7d"
K = 400

attrib = json.loads((SCRATCH / "hotwater_ap_attribution.json").read_text())
side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y, bid, sid, reading, use = (side["y"], side["bid"], side["sid"],
                             side["reading"], side["use"].astype(str))
n_pos = int(y.sum())

READING_LABEL = {
    "zero": "= 0",
    "d0": "0.002 – 5.8", "d1": "5.8 – 20.5", "d2": "20.5 – 44", "d3": "44 – 76",
    "d4": "76 – 139", "d5": "139 – 214", "d6": "214 – 366", "d7": "366 – 615",
    "d8": "615 – 1 132", "d9": "> 1 132",
}


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


bins = np.full(len(y), "zero", dtype=object)
posr = reading > 0
import pandas as pd
q = pd.qcut(reading[posr], 10, labels=False, duplicates="drop")
bins[posr] = [f"d{int(v)}" for v in q]

POS_SHARE = {
    "reading_bin": {k: float((y[(bins == k)] == 1).sum()) / n_pos for k in set(bins)},
    "use": {k: float((y[use == k] == 1).sum()) / n_pos for k in set(use)},
    "site": {str(k): float((y[sid == k] == 1).sum()) / n_pos for k in set(sid.tolist())},
}

sub = [r for r in attrib if r["budget"] == K]
total = agg([((r["building_seed"], r["row_seed"]), r["ap_tab"] - r["ap_tree"]) for r in sub])[0]


def rows_for(dim):
    keys = sorted({k for r in sub for k in r[dim]})
    out = []
    for key in keys:
        m, s = agg([((r["building_seed"], r["row_seed"]), r[dim].get(key, {"delta": 0.0})["delta"])
                    for r in sub])
        w = POS_SHARE[dim].get(key, 0.0)
        if w == 0 and abs(m) < 1e-6:
            continue
        out.append({"key": key, "delta": m, "se": s, "w": w})
    return sorted(out, key=lambda d: d["delta"])


PANELS = [
    ("reading_bin", "a   By meter reading", lambda k: READING_LABEL.get(k, k)),
    ("use", "b   By primary use", lambda k: k),
    ("site", "c   By site", lambda k: f"Site {k}"),
]
data = {dim: rows_for(dim) for dim, _, _ in PANELS}
heights = [len(data[dim]) for dim, _, _ in PANELS]

fig, axes = plt.subplots(3, 1, figsize=(9.6, 9.8),
                         gridspec_kw={"height_ratios": heights}, sharex=True)
fig.patch.set_facecolor(SURFACE)
XLO, XHI = -0.098, 0.050

for ax, (dim, title, fmt) in zip(axes, PANELS):
    rows = data[dim]
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=8.5, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ypos = np.arange(len(rows))[::-1]
    colors = [RUST if r["delta"] < 0 else TEAL for r in rows]
    ax.barh(ypos, [r["delta"] for r in rows], height=0.62, color=colors, edgecolor="none",
            zorder=3)
    ax.errorbar([r["delta"] for r in rows], ypos,
                xerr=[0 if math.isnan(r["se"]) else r["se"] for r in rows],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.4, capthick=0.8, zorder=4)
    ax.axvline(0, color=AXIS, linewidth=0.9, zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{fmt(r['key'])}   ·  {r['w'] * 100:.1f}%" for r in rows], fontsize=8.2)
    ax.set_xlim(XLO, XHI)
    ax.set_xticks([-0.08, -0.06, -0.04, -0.02, 0.0])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    for r, yy in zip(rows, ypos):
        share = r["delta"] / total * 100
        txt = f"{r['delta']:+.4f}"
        if abs(share) >= 2.0:
            txt += f"   ({share:.0f}% of gap)"
        ax.annotate(txt, (0.0095, yy), va="center", ha="left",
                    fontsize=7.8, color=SECONDARY, zorder=5)
    ax.set_title(title, loc="left", fontsize=11.3, fontweight="bold", color=INK, pad=8)

axes[-1].set_xlabel("Contribution to the hot-water PR-AUC gap  (TabPFN − Tree Ensemble)",
                    fontsize=8.9, color=SECONDARY, labelpad=7)

fig.suptitle("K = 400 — what the hot-water PR-AUC gap is made of",
             x=0.026, y=0.976, ha="left", fontsize=15, fontweight="bold", color=INK)
fig.text(0.026, 0.947,
         "Additive decomposition · contributions sum to the total gap of "
         f"{total:+.4f} · mean ± standard error over 5 building seeds · shared x-scale.",
         ha="left", fontsize=9.15, color=SECONDARY)
fig.text(0.026, 0.926,
         "Row labels give each category's share of the 90,691 hot-water anomalies; "
         "that share does not predict its contribution.",
         ha="left", fontsize=8.45, color=SECONDARY)
fig.subplots_adjust(left=0.275, right=0.995, top=0.885, bottom=0.058, hspace=0.32)

out = SCRATCH / "fig_k400_contributions.png"
tmp = out.with_name(f".{out.name}.tmp")
fig.savefig(tmp, format="png", dpi=180, facecolor=SURFACE, edgecolor="none")
plt.close(fig)
os.replace(tmp, out)
print("saved", out.name, f"total={total:+.4f}")
for dim, _, _ in PANELS:
    print(f"\n{dim}")
    for r in data[dim]:
        print(f"  {r['key']:<28} w={r['w']:.4f}  delta={r['delta']:+.5f} "
              f"se={r['se']:.5f}  share={r['delta'] / total * 100:+.1f}%")
