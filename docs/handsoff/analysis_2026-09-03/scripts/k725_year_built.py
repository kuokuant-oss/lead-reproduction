#!/usr/bin/env python3
"""K=725 hot water：依 year_built 的 PR-AUC 比較，含穩健性檢查（移除最大建物）。"""
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
y, bid = side["y"].astype(np.int8), side["bid"].astype(np.int32)
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


def compare(mask, label):
    yy = y[mask]
    p = int(yy.sum())
    if p < 50 or p == len(yy):
        return None
    tr = [average_precision_score(yy, s[mask]) for s in tree_s]
    tb = [average_precision_score(yy, s[mask]) for s in tab_s]
    d = np.array(tb) - np.array(tr)
    pos_b = bid[mask & (y == 1)]
    vc = pd.Series(pos_b).value_counts()
    return {"group": label, "buildings": int(len(vc)), "anomalies": p,
            "top_b": int(vc.index[0]), "top_share": float(vc.iloc[0] / p),
            "tree": float(np.mean(tr)), "tabpfn": float(np.mean(tb)),
            "delta": float(d.mean()), "delta_se": float(d.std(ddof=1) / math.sqrt(5)),
            "same_dir": int((np.sign(d) == np.sign(d.mean())).sum())}


BINS = [(1900, 1950), (1950, 1970), (1970, 1990), (1990, 2020)]
rows = []
for lo, hi in BINS:
    m = (year >= lo) & (year < hi)
    r = compare(m, f"{lo}–{hi - 1}")
    if r:
        rows.append(r)
rows.append(compare(~np.isfinite(year), "year_built 缺值"))
df = pd.DataFrame([r for r in rows if r])
with pd.option_context("display.width", 260, "display.float_format", lambda v: f"{v:.4f}"):
    print("=== 依 year_built ===")
    print(df.to_string(index=False))

print("\n=== 穩健性：移除該箱最大貢獻建物後 ===")
for lo, hi in BINS:
    m = (year >= lo) & (year < hi)
    base = compare(m, "")
    if base is None:
        continue
    m2 = m & (bid != base["top_b"])
    r2 = compare(m2, "")
    if r2 is None:
        continue
    print(f"  {lo}–{hi - 1}: 原 delta={base['delta']:+.4f}（{base['buildings']} 棟）"
          f" → 移除 b{base['top_b']} 後 delta={r2['delta']:+.4f}"
          f"（{r2['buildings']} 棟, {r2['anomalies']:,} 異常, "
          f"Tree {r2['tree']:.4f} / TabPFN {r2['tabpfn']:.4f}, 同向 {r2['same_dir']}/5）")

print("\n=== 各箱的建物清單（異常數）===")
for lo, hi in BINS:
    m = (year >= lo) & (year < hi)
    vc = pd.Series(bid[m & (y == 1)]).value_counts()
    if len(vc) == 0:
        continue
    print(f"  {lo}–{hi - 1}: " + ", ".join(f"b{b}({n:,})" for b, n in vc.items()))
