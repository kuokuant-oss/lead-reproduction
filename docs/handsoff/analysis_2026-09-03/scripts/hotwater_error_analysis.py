#!/usr/bin/env python3
"""唯讀：hot water (meter=3) 上 TabPFN vs Tree 的 PR-AUC 誤差結構分析。

用法: python hotwater_error_analysis.py <budget> <building_seed> <row_seed>
只讀 predictions.npz / cell.json / building_metadata.csv，不寫入任何檔案。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PLOT = Path(
    "/mnt/c/Users/User/Documents/Codex/2026-09-01/wsl-ubuntu-m5-building-count-v5"
    "/outputs/scripts/plot_m5_v5_fixed50k_building_scarcity_roc.py"
)
ROOT = Path(
    "/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k/data/processed"
    "/m5_building_curve/v5_fixed_50k/model_runs"
)
META = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/building_metadata.csv")
HOT_WATER = 3

spec = importlib.util.spec_from_file_location("m5plot", PLOT)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)

BUDGET, BSEED, RSEED = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
LABEL = f"k{BUDGET}_b{BSEED}_r{RSEED}"

dirs = {m: m5.cell_dir(ROOT, BSEED, RSEED, BUDGET, m) for m in ("ensemble", "tabpfn")}
meta = {m: m5.validate_cell(d, BUDGET, BSEED, RSEED, m) for m, d in dirs.items()}
m5.validate_pair(meta["ensemble"], meta["tabpfn"], LABEL)
print(f"[gate] {LABEL}: cell identity OK, pair identity OK")

arrays = {}
scores = {}
for model, d in dirs.items():
    with np.load(d / "predictions.npz") as payload:
        arr = {k: np.asarray(payload[k]) for k in m5.HOLDOUT_ARRAY_KEYS}
        scores[model] = np.asarray(payload[model], dtype=np.float64)
    if arrays:
        for k in m5.HOLDOUT_ARRAY_KEYS:
            if not np.array_equal(arrays[k], arr[k]):
                raise RuntimeError(f"holdout array {k} differs between models")
    else:
        arrays = arr
print("[gate] holdout arrays element-wise identical between Tree and TabPFN")

keep = arrays["meter"].astype(np.int8, copy=False) == HOT_WATER
y = arrays["anomaly"].astype(np.int8, copy=False)[keep]
bid = arrays["building_id"][keep]
sid = arrays["site_id"][keep]
s_tree = scores["ensemble"][keep]
s_tab = scores["tabpfn"][keep]
del scores, arrays

n, n_pos = len(y), int(y.sum())
print(f"\n=== hot water 子集組成 ({LABEL}) ===")
print(f"rows={n:,}  positives={n_pos:,} ({n_pos / n:.4%})  buildings={len(np.unique(bid))}  sites={len(np.unique(sid))}")

ap_tree = average_precision_score(y, s_tree)
ap_tab = average_precision_score(y, s_tab)
print(f"PR-AUC  Tree={ap_tree:.4f}  TabPFN={ap_tab:.4f}  delta={ap_tab - ap_tree:+.4f}")
print(f"ROC-AUC Tree={roc_auc_score(y, s_tree):.4f}  TabPFN={roc_auc_score(y, s_tab):.4f}")

bmeta = pd.read_csv(META).set_index("building_id")
use = bmeta["primary_use"].reindex(bid).to_numpy()
sqft = bmeta["square_feet"].reindex(bid).to_numpy()

df = pd.DataFrame(
    {"y": y, "bid": bid, "sid": sid, "use": use, "sqft": sqft,
     "tree": s_tree, "tab": s_tab}
)


def group_report(df: pd.DataFrame, key: str, title: str, top: int | None = None) -> None:
    rows = []
    for value, sub in df.groupby(key, observed=True):
        pos = int(sub["y"].sum())
        if pos == 0 or pos == len(sub):
            rows.append({key: value, "rows": len(sub), "pos": pos, "prev": pos / len(sub),
                         "ap_tree": np.nan, "ap_tab": np.nan, "delta": np.nan})
            continue
        a_t = average_precision_score(sub["y"], sub["tree"])
        a_p = average_precision_score(sub["y"], sub["tab"])
        rows.append({key: value, "rows": len(sub), "pos": pos, "prev": pos / len(sub),
                     "ap_tree": a_t, "ap_tab": a_p, "delta": a_p - a_t})
    out = pd.DataFrame(rows).sort_values("delta", na_position="last")
    if top:
        out = pd.concat([out.head(top), out.tail(top)])
    print(f"\n=== {title} (組內 PR-AUC) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(out.to_string(index=False))


group_report(df, "sid", "依 site")
group_report(df, "use", "依 primary_use")

print("\n=== 依 building：PR-AUC 落差最大的 10 棟 / 最有利的 10 棟 ===")
group_report(df, "bid", "依 building", top=10)

# ---- 全域 AP 的 leave-one-building-out：gap 集中度 ----
print("\n=== leave-one-building-out：移除該棟後的全域 gap 變化 ===")
base_gap = ap_tab - ap_tree
loo = []
for b in np.unique(bid):
    m = bid != b
    if not (0 < int(y[m].sum()) < int(m.sum())):
        continue
    g = average_precision_score(y[m], s_tab[m]) - average_precision_score(y[m], s_tree[m])
    loo.append({"bid": int(b), "rows_removed": int((~m).sum()),
                "pos_removed": int(y[~m].sum()), "gap_without": g,
                "gap_recovered": g - base_gap})
loo_df = pd.DataFrame(loo).sort_values("gap_recovered", ascending=False)
print(f"base gap = {base_gap:+.4f}")
with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.4f}"):
    print(loo_df.head(10).to_string(index=False))

# ---- top-K 誤報結構 ----
print("\n=== R-precision 與高分區誤報結構 ===")
for name, s in (("Tree", s_tree), ("TabPFN", s_tab)):
    order = np.argsort(-s, kind="stable")[:n_pos]
    hit = int(y[order].sum())
    print(f"{name}: R-precision@{n_pos} = {hit / n_pos:.4f}  ({hit:,} 真陽 / {n_pos:,})")
    fp = order[y[order] == 0]
    if len(fp):
        vc = pd.Series(bid[fp]).value_counts().head(5)
        vcu = pd.Series(use[fp]).value_counts().head(5)
        vcs = pd.Series(sid[fp]).value_counts().head(5)
        print(f"  誤報 {len(fp):,} 筆，前 5 棟建物: {dict(vc)}")
        print(f"  誤報 primary_use 分布(前5): {dict(vcu)}")
        print(f"  誤報 site 分布(前5): {dict(vcs)}")

# ---- 正樣本排名退化 ----
print("\n=== 正樣本的排名百分位（1=最高分）===")
pct_tree = pd.Series(s_tree).rank(pct=True, method="average").to_numpy()
pct_tab = pd.Series(s_tab).rank(pct=True, method="average").to_numpy()
pos_mask = y == 1
d_pct = pct_tab[pos_mask] - pct_tree[pos_mask]
print(f"正樣本平均百分位  Tree={pct_tree[pos_mask].mean():.4f}  TabPFN={pct_tab[pos_mask].mean():.4f}")
print(f"TabPFN 排名變差(百分位下降>0.1)的正樣本比例: {(d_pct < -0.1).mean():.4%}")
worst = pd.DataFrame({"bid": bid[pos_mask], "use": use[pos_mask], "sid": sid[pos_mask], "d": d_pct})
agg = worst.groupby("bid").agg(pos=("d", "size"), mean_drop=("d", "mean")).sort_values("mean_drop")
print("\n排名退化最嚴重的 10 棟（正樣本平均百分位變化）:")
with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
    print(agg.head(10).to_string())
print("\n排名改善最多的 10 棟:")
with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
    print(agg.tail(10).to_string())
