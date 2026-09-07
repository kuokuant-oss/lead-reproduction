#!/usr/bin/env python3
"""K=725 hot water：掃描所有分組維度，找出 Tree 大勝且非單一建物主導的群組。

分組維度：site、primary use、square feet 四分位、year built 分箱、floor count 分箱、
site × use 交叉。篩選條件：≥4 棟有異常的建物、最大建物佔異常 ≤ 40%、異常數 ≥ 500。
唯讀。
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
summary, tree_dirs = m5.validate_k725(REPO, formal_root, ext_root,
                                      colab / "model_results/k725_five_seed_summary.json")
print(f"[gate] validate_k725 通過")

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y = side["y"].astype(np.int8)
bid = side["bid"].astype(np.int32)
sid = side["sid"].astype(np.int32)

tree_scores, tab_scores = [], []
for rs in (0, 1, 2, 3, 4):
    with np.load(tree_dirs[rs] / "predictions.npz") as z:
        keep = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        tree_scores.append(np.asarray(z["ensemble"], dtype=np.float64)[keep])
    p = (m5.cell_dir(formal_root, 725, rs, 725, "tabpfn") / "predictions.npz" if rs < 2
         else colab / f"model_results/row_seed{rs}/predictions.npz")
    with np.load(p) as z:
        k2 = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        tab_scores.append(np.asarray(z["tabpfn"], dtype=np.float64)[k2])

bmeta = pd.read_csv(META).set_index("building_id")
use_arr = bmeta["primary_use"].reindex(bid).to_numpy()
sqft = bmeta["square_feet"].reindex(bid).to_numpy(dtype=float)
year = bmeta["year_built"].reindex(bid).to_numpy(dtype=float)
floors = bmeta["floor_count"].reindex(bid).to_numpy(dtype=float)


def qbin(values, label, n=4):
    """依建物層級的分位切箱，回傳 (mask, label) 清單。"""
    uniq = pd.DataFrame({"b": bid, "v": values}).drop_duplicates("b")
    uniq = uniq[np.isfinite(uniq["v"])]
    if len(uniq) < n * 2:
        return []
    edges = np.percentile(uniq["v"], np.linspace(0, 100, n + 1))
    out = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        m = (values >= lo) & (values <= hi) if i == n - 1 else (values >= lo) & (values < hi)
        out.append((m, f"{label} {lo:,.0f}–{hi:,.0f}"))
    return out


groups = [(sid == s, f"site {s}") for s in np.unique(sid)]
groups += [(use_arr == u, f"use {u}") for u in pd.unique(use_arr)]
groups += qbin(sqft, "sqft")
groups += qbin(year, "year built")
groups += qbin(floors, "floors", n=3)
for s in np.unique(sid):
    for u in pd.unique(use_arr[sid == s]):
        groups.append(((sid == s) & (use_arr == u), f"site {s} × {u}"))

rows = []
for mask, label in groups:
    if mask.sum() == 0:
        continue
    yy = y[mask]
    p = int(yy.sum())
    if p < 500 or p == len(yy):
        continue
    pos_b = bid[mask & (y == 1)]
    vc = pd.Series(pos_b).value_counts()
    tree = [average_precision_score(yy, s[mask]) for s in tree_scores]
    tab = [average_precision_score(yy, s[mask]) for s in tab_scores]
    d = np.array(tab) - np.array(tree)
    rows.append({
        "group": label, "buildings": int(len(vc)), "anomalies": p,
        "top_share": float(vc.iloc[0] / p),
        "tree": float(np.mean(tree)), "tabpfn": float(np.mean(tab)),
        "delta": float(d.mean()), "delta_se": float(d.std(ddof=1) / math.sqrt(5)),
        "same_dir": int((np.sign(d) == np.sign(d.mean())).sum()),
    })

df = pd.DataFrame(rows)
clean = df[(df.buildings >= 4) & (df.top_share <= 0.40)].copy()

cols = ["group", "buildings", "anomalies", "top_share", "tree", "tabpfn",
        "delta", "delta_se", "same_dir"]
with pd.option_context("display.width", 260, "display.max_columns", 20,
                       "display.float_format", lambda v: f"{v:.4f}"):
    print(f"\n=== 通過篩選（≥4 棟、最大建物 ≤40%、異常 ≥500）：{len(clean)} 組 ===")
    print("\n-- Tree 領先最多 --")
    print(clean.sort_values("delta").head(12)[cols].to_string(index=False))
    print("\n-- TabPFN 領先最多 --")
    print(clean.sort_values("delta", ascending=False).head(8)[cols].to_string(index=False))
    print(f"\n=== 未通過篩選但差距大的（單一建物主導）===")
    dirty = df[(df.buildings < 4) | (df.top_share > 0.40)]
    print(dirty.reindex(dirty.delta.abs().sort_values(ascending=False).index)
          .head(10)[cols].to_string(index=False))
