#!/usr/bin/env python3
"""兩個趨勢檢查：
A. K=200 → K=400 的落差惡化是否為真（同 building seed 配對）。
B. 「四個 K 都為負」的組別是否指向同一批列。
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
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
META = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/building_metadata.csv")
HOT_WATER = 3

spec = importlib.util.spec_from_file_location("m5plot", PLOT)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)
formal_root = REPO / "data/processed/m5_building_curve/v5_fixed_50k/model_runs"

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y = side["y"].astype(np.int8)
bid = side["bid"].astype(np.int32)
sid = side["sid"].astype(np.int32)
N = len(y)

# ---------- A. K=200 vs K=400 的配對比較 ----------
print("=== A. 落差在 K=200 → K=400 是否惡化（同 building seed 配對）===")
gaps = {100: {}, 200: {}, 400: {}}
for budget in (100, 200, 400):
    for b in range(5):
        per_row_seed = []
        for r in (0, 1):
            td = m5.cell_dir(formal_root, b, r, budget, "ensemble")
            pd_ = m5.cell_dir(formal_root, b, r, budget, "tabpfn")
            m5.validate_pair(m5.validate_cell(td, budget, b, r, "ensemble"),
                             m5.validate_cell(pd_, budget, b, r, "tabpfn"),
                             f"k{budget}_b{b}_r{r}")
            with np.load(td / "predictions.npz") as z:
                keep = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
                st = np.asarray(z["ensemble"], dtype=np.float64)[keep]
            with np.load(pd_ / "predictions.npz") as z:
                k2 = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
                sp = np.asarray(z["tabpfn"], dtype=np.float64)[k2]
            per_row_seed.append(average_precision_score(y, sp)
                                - average_precision_score(y, st))
        gaps[budget][b] = float(np.mean(per_row_seed))
    print(f"  K={budget} 逐 building seed 落差: "
          + ", ".join(f"b{b}={gaps[budget][b]:+.4f}" for b in range(5)))

for lo, hi in ((100, 200), (200, 400)):
    d = np.array([gaps[hi][b] - gaps[lo][b] for b in range(5)])
    t, p = stats.ttest_rel([gaps[hi][b] for b in range(5)], [gaps[lo][b] for b in range(5)])
    w = stats.wilcoxon([gaps[hi][b] for b in range(5)], [gaps[lo][b] for b in range(5)])
    print(f"\n  K={lo} → K={hi}: 逐 seed 變化 " + ", ".join(f"{v:+.4f}" for v in d))
    print(f"    平均 {d.mean():+.4f} ± {d.std(ddof=1) / math.sqrt(5):.4f}  "
          f"同向 {int((np.sign(d) == np.sign(d.mean())).sum())}/5")
    print(f"    配對 t: t={t:.3f} p={p:.4f}   Wilcoxon: W={w.statistic:.0f} p={w.pvalue:.4f}")

# ---------- B. 持續為負的組別是否重疊 ----------
print("\n=== B. 四個 K 都為負的組別，是否指向同一批列 ===")
bmeta = pd.read_csv(META).set_index("building_id")
use = bmeta["primary_use"].reindex(bid).to_numpy()
sqft = bmeta["square_feet"].reindex(bid).to_numpy(dtype=float)
year = bmeta["year_built"].reindex(bid).to_numpy(dtype=float)
floors = bmeta["floor_count"].reindex(bid).to_numpy(dtype=float)

uniq = pd.DataFrame({"b": bid, "v": sqft}).drop_duplicates("b")
q = np.percentile(uniq["v"], [0, 25, 50, 75, 100])
fu = pd.DataFrame({"b": bid, "v": floors}).drop_duplicates("b")
fq = np.percentile(fu[np.isfinite(fu["v"])]["v"], [0, 100 / 3, 200 / 3, 100])

PERSIST = {
    "Site 10": sid == 10,
    "Site 15": sid == 15,
    "Technology/science": use == "Technology/science",
    "Food sales and service": use == "Food sales and service",
    "sqft Q1 (9.7k–49.7k)": (sqft >= q[0]) & (sqft < q[1]),
    "year 1900–1949": (year >= 1900) & (year < 1950),
    "floors 2–4": (floors >= fq[0]) & (floors < fq[1]),
    "Education": use == "Education",
    "Site 2": sid == 2,
    "year 缺值": ~np.isfinite(year),
}
labels = list(PERSIST)
print(f"{'組別':<26}{'列':>9}{'正樣本':>9}   建物")
for k, m in PERSIST.items():
    print(f"{k:<26}{int(m.sum()):>9,}{int(y[m].sum()):>9,}   "
          + ",".join(f"b{b}" for b in np.unique(bid[m])[:8])
          + ("…" if len(np.unique(bid[m])) > 8 else ""))

print("\n  兩兩重疊（Jaccard，以列為單位）")
print("       " + "".join(f"{i:>7}" for i in range(len(labels))))
for i, a in enumerate(labels):
    row = f"  [{i}] "
    for j, b in enumerate(labels):
        ma, mb = PERSIST[a], PERSIST[b]
        inter = int((ma & mb).sum())
        union = int((ma | mb).sum())
        row += f"{inter / union:>7.2f}"
    print(row + f"  {a}")

small = ["Site 10", "Site 15", "Technology/science", "Food sales and service",
         "year 1900–1949", "floors 2–4"]
union = np.zeros(N, dtype=bool)
for k in small:
    union |= PERSIST[k]
print(f"\n  六個小型持續負向組的聯集: {int(union.sum()):,} 列 "
      f"({union.sum() / N * 100:.1f}%)、{int(y[union].sum()):,} 正樣本 "
      f"({y[union].sum() / y.sum() * 100:.1f}%)、"
      f"{len(np.unique(bid[union]))} 棟建物")
print(f"  其中的建物: " + ",".join(f"b{b}" for b in np.unique(bid[union])))
