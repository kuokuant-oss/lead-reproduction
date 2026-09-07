#!/usr/bin/env python3
"""Composition of the hot-water PR-AUC gap at K = 400.

圖要回答的問題（作圖規範 v0.3 §1）：
    在相同的 hot-water 評估列上，TabPFN 相對 Tree Ensemble 的 PR-AUC 落差
    由 holdout 的哪些部分承擔？

依 v0.3：§2 骨架（名詞片語主標題、單行 subtitle、左對齊 panel 標題、無外框、
圖底不放方法備註）、§3 字級 16/10.5/11.5 並保留標題行距、§4 共用 x 尺度、
8% headroom、1/2/5 刻度、只有最底 panel 顯示刻度與單一共用軸名、§5 mean 搭配 SE、
§6 只標少數需精確讀取的值、線寬/marker/格線依對角線縮放、§6.1 留白用於分組、
§7.1 只用 ink token（不自建與模型色衝突的類別 palette）、§8 排序 dot plot、
§9.3 手動 subplots_adjust + facecolor=surface。

有序量（meter reading）保持數值升冪且使用等距（十進位）分箱；名目類別
（primary use、site）依貢獻排序。數值標籤放在座標軸外的固定欄，資料軸因此
維持 8% headroom 且不會與資料或格線互撞。
"""
from __future__ import annotations

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

PRIMARY_INK, SECONDARY_INK, MUTED_INK = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

FIGSIZE = (9.6, 8.7)
GRIDWIDTH = 0.7
BARHEIGHT = 0.60  # §4-7：bar 自 0 起算
LABEL_SHARE_CUTOFF = 2.0

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y, sid, reading, use = side["y"], side["sid"], side["reading"], side["use"].astype(str)
store = np.load(SCRATCH / "positive_contrib.npz")
cells, contrib, pos_idx = store["cells"], store["contrib"], store["pos_idx"]

READING_EDGES = [0.0, 1.0, 10.0, 100.0, 1e3, 1e4, np.inf]
READING_KEYS = ["= 0", "> 0 – 1", "1 – 10", "10 – 100", "100 – 1 000",
                "1 000 – 10 000", "> 10 000"]
reading_bin = np.full(len(y), READING_KEYS[0], dtype=object)
for i in range(len(READING_EDGES) - 1):
    reading_bin[(reading > READING_EDGES[i]) & (reading <= READING_EDGES[i + 1])] = \
        READING_KEYS[i + 1]

USE_SHORT = {"Entertainment/public assembly": "Entertainment / assembly",
             "Food sales and service": "Food sales and service",
             "Lodging/residential": "Lodging / residential",
             "Technology/science": "Technology / science"}

DIMENSIONS = {
    "reading_bin": np.asarray(reading_bin, dtype=object)[pos_idx],
    "use": np.asarray([USE_SHORT.get(u, u) for u in use], dtype=object)[pos_idx],
    "site": np.asarray([f"Site {s}" for s in sid], dtype=object)[pos_idx],
}
n_pos = len(pos_idx)


def aggregate(mask):
    per_cell = contrib[:, mask].sum(axis=1)
    by_seed = {}
    for (_, bseed, _), v in zip(cells, per_cell):
        by_seed.setdefault(int(bseed), []).append(float(v))
    means = [float(np.mean(v)) for v in by_seed.values()]
    return float(np.mean(means)), float(np.std(means, ddof=1) / math.sqrt(len(means)))


total, total_se = aggregate(np.ones(n_pos, dtype=bool))


def rows_for(dim, order=None):
    values = DIMENSIONS[dim]
    out = []
    for key in (order or sorted(set(values))):
        mask = values == key
        if not mask.any():
            continue
        m, s = aggregate(mask)
        out.append({"key": key, "delta": m, "se": s, "w": float(mask.sum()) / n_pos})
    return out if order else sorted(out, key=lambda d: d["delta"])


PANELS = [
    ("reading_bin", "Meter reading, log-spaced bins", READING_KEYS),
    ("use", "Primary use", None),
    ("site", "Site", None),
]
data = {dim: rows_for(dim, order) for dim, _, order in PANELS}

values = [r["delta"] for rows in data.values() for r in rows]
errs_all = [0.0 if math.isnan(r["se"]) else r["se"]
            for rows in data.values() for r in rows]
lo = min(v - e for v, e in zip(values, errs_all))
hi = max(v + e for v, e in zip(values, errs_all))
pad = 0.08 * (hi - lo)
XLO, XHI = lo - pad, hi + pad
XTICKS = [round(t, 3) for t in np.arange(-0.10, 0.021, 0.02) if XLO <= t <= XHI]

fig, axes = plt.subplots(
    len(PANELS), 1, figsize=FIGSIZE, sharex=True,
    gridspec_kw={"height_ratios": [len(data[d]) for d, _, _ in PANELS]},
)
fig.patch.set_facecolor(SURFACE)

for ax, (dim, panel_title, _) in zip(axes, PANELS):
    rows = data[dim]
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(GRIDWIDTH)
    ax.grid(axis="x", color=GRID, linewidth=GRIDWIDTH)
    ax.set_axisbelow(True)

    ypos = np.arange(len(rows))[::-1]
    deltas = np.array([r["delta"] for r in rows])
    errs = np.array([0.0 if math.isnan(r["se"]) else r["se"] for r in rows])

    ax.barh(ypos, deltas, height=BARHEIGHT, color=PRIMARY_INK, edgecolor="none",
            zorder=3)
    ax.axvline(0.0, color=BASELINE, linewidth=GRIDWIDTH * 1.3, zorder=4)

    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r['key']}   ·   {r['w'] * 100:.1f}%" for r in rows],
                       fontsize=9.5, color=SECONDARY_INK)
    ax.tick_params(axis="y", length=0, colors=SECONDARY_INK)
    ax.tick_params(axis="x", colors=SECONDARY_INK, labelsize=9.5,
                   length=3.0, width=GRIDWIDTH)
    ax.set_xlim(XLO, XHI)
    ax.set_xticks(XTICKS)
    ax.set_ylim(-0.7, len(rows) - 0.3)

    # 數值欄放在座標軸外（axes fraction x=1.03），不侵占資料區
    for row, yy in zip(rows, ypos):
        share = row["delta"] / total * 100
        value = f"{row['delta']:+.4f}".replace("-", "−")
        se = "" if math.isnan(row["se"]) else f" ± {row['se']:.4f}"
        text = (f"{value}{se}   {share:.0f}%"
                if abs(share) >= LABEL_SHARE_CUTOFF else f"{value}{se}")
        weight = "bold" if abs(share) >= 10.0 else "normal"
        ax.annotate(text, xy=(1.03, yy), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=9.5, fontweight=weight,
                    color=PRIMARY_INK if abs(share) >= LABEL_SHARE_CUTOFF else MUTED_INK,
                    annotation_clip=False)

    ax.set_title(panel_title, loc="left", fontsize=11.5, fontweight="bold",
                 color=PRIMARY_INK, pad=6)

axes[-1].set_xlabel("Contribution to PR-AUC gap, TabPFN − Tree Ensemble",
                    fontsize=10.5, color=SECONDARY_INK, labelpad=8)

fig.suptitle("Composition of the hot-water PR-AUC gap at K = 400",
             x=0.070, y=0.968, ha="left", fontsize=16, fontweight="bold",
             color=PRIMARY_INK)
fig.text(0.070, 0.918,
         "Additive contributions over 5 building seeds · total gap "
         + f"{total:+.4f}".replace("-", "−")
         + f" ± {total_se:.4f} · shared x-scale",
         ha="left", fontsize=10.5, color=SECONDARY_INK)
fig.text(0.070, 0.851, "category  ·  % of anomalies",
         ha="left", fontsize=8.8, color=MUTED_INK)
fig.text(0.795, 0.851, "contribution ± SE  ·  % of gap",
         ha="left", fontsize=8.8, color=MUTED_INK)

fig.subplots_adjust(left=0.335, right=0.782, top=0.838, bottom=0.125, hspace=0.26)

out = SCRATCH / "m5_hot_water_gap_composition_pr_auc.png"
tmp = out.with_name(f".{out.name}.tmp")
fig.savefig(tmp, format="png", dpi=180, facecolor=SURFACE, edgecolor="none")
plt.close(fig)
os.replace(tmp, out)

print(f"saved {out.name}   total={total:+.5f} ± {total_se:.5f}")
for dim, _, _ in PANELS:
    print(f"\n{dim}")
    for r in data[dim]:
        print(f"  {r['key']:<26} w={r['w']:.4f}  delta={r['delta']:+.5f}  "
              f"se={r['se']:.5f}  share={r['delta'] / total * 100:+.1f}%")
