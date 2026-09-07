#!/usr/bin/env python3
"""K=725 hot water 分組 PR-AUC，正確口徑：
- 每個分組涵蓋該組全部列（含沒有異常的建物，它們是負樣本）
- 缺值自成一個類別，不丟棄
- 建物數同時報「全部」與「其中有異常」
- 另做「固定 site 2」的年份對照，去除 site 混淆
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

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
META = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/building_metadata.csv")
HOT_WATER = 3

spec = importlib.util.spec_from_file_location("m5plot", PLOT)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)
formal_root = REPO / "data/processed/m5_building_curve/v5_fixed_50k/model_runs"
ext_root = REPO / "data/processed/m5_building_curve/v5_fixed_50k_k725_row_seed_extension/model_runs"
colab = REPO / "data/processed/m5_building_curve/v5_fixed_50k_k725_colab"
_, tree_dirs = m5.validate_k725(REPO, formal_root, ext_root,
                                colab / "model_results/k725_five_seed_summary.json")

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y, bid, sid = side["y"].astype(np.int8), side["bid"].astype(np.int32), side["sid"].astype(np.int32)
tree_s, tab_s = [], []
for rs in (0, 1, 2, 3, 4):
    with np.load(tree_dirs[rs] / "predictions.npz") as z:
        keep = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        tree_s.append(np.asarray(z["ensemble"], dtype=np.float64)[keep])
    p = (m5.cell_dir(formal_root, 725, rs, 725, "tabpfn") / "predictions.npz" if rs < 2
         else colab / f"model_results/row_seed{rs}/predictions.npz")
    with np.load(p) as z:
        k2 = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        tab_s.append(np.asarray(z["tabpfn"], dtype=np.float64)[k2])

bmeta = pd.read_csv(META).set_index("building_id")
year = bmeta["year_built"].reindex(bid).to_numpy(dtype=float)
sqft = bmeta["square_feet"].reindex(bid).to_numpy(dtype=float)
use = bmeta["primary_use"].reindex(bid).to_numpy()


def row(mask, label):
    if mask.sum() == 0:
        return None
    yy = y[mask]
    p = int(yy.sum())
    b_all = np.unique(bid[mask])
    b_pos = np.unique(bid[mask & (y == 1)]) if p else np.array([])
    if p == 0 or p == len(yy):
        return {"group": label, "buildings": len(b_all), "b_with_anom": len(b_pos),
                "rows": int(mask.sum()), "anomalies": p, "prev": p / int(mask.sum()),
                "tree": np.nan, "tabpfn": np.nan, "delta": np.nan, "delta_se": np.nan,
                "same_dir": 0}
    tr = [average_precision_score(yy, s[mask]) for s in tree_s]
    tb = [average_precision_score(yy, s[mask]) for s in tab_s]
    d = np.array(tb) - np.array(tr)
    return {"group": label, "buildings": len(b_all), "b_with_anom": len(b_pos),
            "rows": int(mask.sum()), "anomalies": p, "prev": p / int(mask.sum()),
            "tree": float(np.mean(tr)), "tabpfn": float(np.mean(tb)),
            "delta": float(d.mean()), "delta_se": float(d.std(ddof=1) / math.sqrt(5)),
            "same_dir": int((np.sign(d) == np.sign(d.mean())).sum())}


COLS = ["group", "buildings", "b_with_anom", "rows", "anomalies", "prev",
        "tree", "tabpfn", "delta", "delta_se", "same_dir"]
fmt = lambda v: f"{v:.4f}"

YBINS = [(1900, 1950, "1900–1949"), (1950, 1970, "1950–1969"),
         (1970, 1990, "1970–1989"), (1990, 2020, "1990–2019")]

print("=== A. year_built（缺值自成類別，全部 73 棟）===")
rows = [row((year >= lo) & (year < hi), lab) for lo, hi, lab in YBINS]
rows.append(row(~np.isfinite(year), "缺值 (missing)"))
dfa = pd.DataFrame([r for r in rows if r])
print(f"  檢核：建物 {dfa.buildings.sum()} / 列 {dfa.rows.sum():,} / 異常 {dfa.anomalies.sum():,}")
with pd.option_context("display.width", 260, "display.float_format", fmt):
    print(dfa[COLS].to_string(index=False))

print("\n=== B. 固定 site 2 後的 year_built（去除 site 混淆）===")
s2 = sid == 2
rows = [row(s2 & (year >= lo) & (year < hi), lab) for lo, hi, lab in YBINS]
rows.append(row(s2 & ~np.isfinite(year), "缺值 (missing)"))
dfb = pd.DataFrame([r for r in rows if r])
with pd.option_context("display.width", 260, "display.float_format", fmt):
    print(dfb[COLS].to_string(index=False))

print("\n=== C. square feet 四分位（無缺值，全部 73 棟）===")
uniq = pd.DataFrame({"b": bid, "v": sqft}).drop_duplicates("b")
edges = np.percentile(uniq["v"], [0, 25, 50, 75, 100])
rows = []
for i in range(4):
    lo, hi = edges[i], edges[i + 1]
    m = (sqft >= lo) & (sqft <= hi) if i == 3 else (sqft >= lo) & (sqft < hi)
    rows.append(row(m, f"{lo:,.0f}–{hi:,.0f} sq ft"))
dfc = pd.DataFrame([r for r in rows if r])
print(f"  檢核：建物 {dfc.buildings.sum()} / 列 {dfc.rows.sum():,} / 異常 {dfc.anomalies.sum():,}")
with pd.option_context("display.width", 260, "display.float_format", fmt):
    print(dfc[COLS].to_string(index=False))

print("\n=== D. site（全部 73 棟）===")
dfd = pd.DataFrame([r for r in (row(sid == s, f"site {s}") for s in np.unique(sid)) if r])
print(f"  檢核：建物 {dfd.buildings.sum()} / 列 {dfd.rows.sum():,} / 異常 {dfd.anomalies.sum():,}")
with pd.option_context("display.width", 260, "display.float_format", fmt):
    print(dfd[COLS].to_string(index=False))

print("\n=== E. primary use（全部 73 棟）===")
dfe = pd.DataFrame([r for r in (row(use == u, u) for u in pd.unique(use)) if r])
print(f"  檢核：建物 {dfe.buildings.sum()} / 列 {dfe.rows.sum():,} / 異常 {dfe.anomalies.sum():,}")
with pd.option_context("display.width", 260, "display.float_format", fmt):
    print(dfe.sort_values("delta")[COLS].to_string(index=False))

print("\n=== F. 沒有異常的建物（只提供負樣本）===")
b_pos = set(np.unique(bid[y == 1]).tolist())
no_anom = sorted(set(np.unique(bid).tolist()) - b_pos)
for b in no_anom:
    m = bid == b
    print(f"  b{b}: site {sid[m][0]}, {use[m][0]}, {int(m.sum()):,} 列, "
          f"sqft {sqft[m][0]:,.0f}, year {year[m][0] if np.isfinite(year[m][0]) else '缺值'}")
