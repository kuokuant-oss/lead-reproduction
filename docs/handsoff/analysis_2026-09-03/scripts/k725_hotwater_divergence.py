#!/usr/bin/env python3
"""K=725 hot water：總 PR-AUC 接近，但逐建物／site／use／面積的表現差異。

沿用正式繪圖腳本的 validate_k725 閘門（r0/r1 formal root、r2-r4 extension + colab）。
唯讀；輸出寫 scratchpad。
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

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
print(f"[gate] validate_k725 通過，row_seeds={summary['row_seeds']}")

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y = side["y"].astype(np.int8)
bid = side["bid"].astype(np.int32)
sid = side["sid"].astype(np.int32)
reading = side["reading"]
use = side["use"].astype(str)

scores = {"ensemble": [], "tabpfn": []}
for row_seed in (0, 1, 2, 3, 4):
    with np.load(tree_dirs[row_seed] / "predictions.npz") as z:
        keep = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        s_tree = np.asarray(z["ensemble"], dtype=np.float64)[keep]
        if not (np.array_equal(np.asarray(z["building_id"])[keep].astype(np.int32), bid)
                and np.array_equal(np.asarray(z["anomaly"])[keep].astype(np.int8), y)):
            raise RuntimeError(f"r{row_seed} tree holdout 不符")
    tab_path = (m5.cell_dir(formal_root, 725, row_seed, 725, "tabpfn") / "predictions.npz"
                if row_seed < 2 else
                colab / f"model_results/row_seed{row_seed}/predictions.npz")
    with np.load(tab_path) as z:
        keep2 = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        s_tab = np.asarray(z["tabpfn"], dtype=np.float64)[keep2]
        if not (np.array_equal(np.asarray(z["building_id"])[keep2].astype(np.int32), bid)
                and np.array_equal(np.asarray(z["anomaly"])[keep2].astype(np.int8), y)):
            raise RuntimeError(f"r{row_seed} tabpfn holdout 不符")
    scores["ensemble"].append(s_tree)
    scores["tabpfn"].append(s_tab)
print("[gate] 5 個 row seed 的 holdout 逐元素相符")

bmeta = pd.read_csv(META).set_index("building_id")
sqft = bmeta["square_feet"].reindex(bid).to_numpy()
primary = bmeta["primary_use"].reindex(bid).to_numpy()


def group_stats(mask, label):
    yy = y[mask]
    p = int(yy.sum())
    if p == 0 or p == len(yy):
        return None
    tree = [average_precision_score(yy, s[mask]) for s in scores["ensemble"]]
    tab = [average_precision_score(yy, s[mask]) for s in scores["tabpfn"]]
    rt = [roc_auc_score(yy, s[mask]) for s in scores["ensemble"]]
    rp = [roc_auc_score(yy, s[mask]) for s in scores["tabpfn"]]
    d = np.array(tab) - np.array(tree)
    return {
        "label": label, "rows": int(mask.sum()), "pos": p, "prev": p / int(mask.sum()),
        "ap_tree": float(np.mean(tree)), "ap_tree_se": float(np.std(tree, ddof=1) / math.sqrt(5)),
        "ap_tab": float(np.mean(tab)), "ap_tab_se": float(np.std(tab, ddof=1) / math.sqrt(5)),
        "delta": float(d.mean()), "delta_se": float(d.std(ddof=1) / math.sqrt(5)),
        "delta_min": float(d.min()), "delta_max": float(d.max()),
        "roc_tree": float(np.mean(rt)), "roc_tab": float(np.mean(rp)),
    }


overall = group_stats(np.ones(len(y), dtype=bool), "ALL")
print(f"\n=== K=725 hot water 全域 ===")
print(f"  PR-AUC  Tree {overall['ap_tree']:.4f} ± {overall['ap_tree_se']:.4f}   "
      f"TabPFN {overall['ap_tab']:.4f} ± {overall['ap_tab_se']:.4f}   "
      f"delta {overall['delta']:+.4f} ± {overall['delta_se']:.4f}")
print(f"  ROC-AUC Tree {overall['roc_tree']:.4f}   TabPFN {overall['roc_tab']:.4f}")

rows = []
for b in np.unique(bid):
    st = group_stats(bid == b, f"building {b}")
    if st is None:
        continue
    st["building"] = int(b)
    st["site"] = int(sid[bid == b][0])
    st["use"] = str(primary[bid == b][0])
    st["sqft"] = float(sqft[bid == b][0])
    z = reading[bid == b] == 0
    st["zero_rate"] = float(z.mean())
    st["p_anom_given_zero"] = float((y[bid == b][z] == 1).mean()) if z.any() else float("nan")
    rows.append(st)
df = pd.DataFrame(rows)
df.to_json(SCRATCH / "k725_hotwater_per_building.json", orient="records", indent=2)

win_tree = int((df["delta"] < 0).sum())
win_tab = int((df["delta"] > 0).sum())
print(f"\n=== 逐建物（{len(df)} 棟有異常）===")
print(f"  Tree 較佳 {win_tree} 棟 · TabPFN 較佳 {win_tab} 棟")
print(f"  兩模型逐建物 AP 的 Spearman 相關 = "
      f"{df[['ap_tree', 'ap_tab']].corr(method='spearman').iloc[0, 1]:.3f}")
print(f"  |delta| 中位數 = {df['delta'].abs().median():.4f}，"
      f"最大 = {df['delta'].abs().max():.4f}")

cols = ["building", "site", "use", "pos", "prev", "sqft", "zero_rate",
        "p_anom_given_zero", "ap_tree", "ap_tab", "delta", "delta_se",
        "delta_min", "delta_max"]
with pd.option_context("display.width", 260, "display.max_columns", 30,
                       "display.float_format", lambda v: f"{v:.4f}"):
    print("\n-- Tree 大勝的 8 棟 --")
    print(df.sort_values("delta").head(8)[cols].to_string(index=False))
    print("\n-- TabPFN 大勝的 8 棟 --")
    print(df.sort_values("delta", ascending=False).head(8)[cols].to_string(index=False))

print("\n=== 依 site ===")
srows = [group_stats(sid == s, f"site {s}") for s in np.unique(sid)]
sdf = pd.DataFrame([r for r in srows if r]).sort_values("delta")
with pd.option_context("display.width", 240, "display.float_format", lambda v: f"{v:.4f}"):
    print(sdf[["label", "rows", "pos", "prev", "ap_tree", "ap_tab", "delta", "delta_se"]]
          .to_string(index=False))

print("\n=== 依 primary use ===")
urows = [group_stats(primary == u, f"{u}") for u in pd.unique(primary)]
udf = pd.DataFrame([r for r in urows if r]).sort_values("delta")
with pd.option_context("display.width", 240, "display.float_format", lambda v: f"{v:.4f}"):
    print(udf[["label", "rows", "pos", "prev", "ap_tree", "ap_tab", "delta", "delta_se"]]
          .to_string(index=False))

print("\n=== 依 square feet 四分位 ===")
edges = np.nanpercentile(np.unique(np.c_[bid, sqft], axis=0)[:, 1], [0, 25, 50, 75, 100])
for i in range(4):
    lo, hi = edges[i], edges[i + 1]
    mask = (sqft >= lo) & (sqft <= hi) if i == 3 else (sqft >= lo) & (sqft < hi)
    st = group_stats(mask, f"{lo:,.0f} – {hi:,.0f} sqft")
    if st:
        print(f"  {st['label']:<28} rows={st['rows']:>7,} pos={st['pos']:>6,} "
              f"Tree={st['ap_tree']:.4f} TabPFN={st['ap_tab']:.4f} "
              f"delta={st['delta']:+.4f} ± {st['delta_se']:.4f}")

print(f"\n[saved] k725_hotwater_per_building.json")
