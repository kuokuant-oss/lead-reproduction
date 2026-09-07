#!/usr/bin/env python3
"""K=725 五個 row seed 的 hot water 落差結構（沿用正式繪圖腳本的 validate_k725 閘門）。

r0/r1：formal root 的 tree + tabpfn cell。
r2-r4：tree 在 k725_row_seed_extension/model_runs；tabpfn 預測在
       k725_colab/model_results/row_seed{r}/predictions.npz，配對以 fit_result.json 的
       context_feature_matrix_sha256 與 tree 對齊（與 validate_k725 相同）。
唯讀；輸出寫 scratchpad。
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
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
HOT_WATER = 3

spec = importlib.util.spec_from_file_location("m5plot", PLOT)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)

formal_root = REPO / "data/processed/m5_building_curve/v5_fixed_50k/model_runs"
ext_root = REPO / "data/processed/m5_building_curve/v5_fixed_50k_k725_row_seed_extension/model_runs"
colab = REPO / "data/processed/m5_building_curve/v5_fixed_50k_k725_colab"
summary_path = colab / "model_results/k725_five_seed_summary.json"

summary, tree_dirs = m5.validate_k725(REPO, formal_root, ext_root, summary_path)
print(f"[gate] validate_k725 通過；row_seeds={summary['row_seeds']}")

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y_hw = side["y"].astype(np.int8)
bid_hw = side["bid"].astype(np.int32)
reading = side["reading"]
n_pos = int(y_hw.sum())

results = []
for row_seed in (0, 1, 2, 3, 4):
    tree_dir = tree_dirs[row_seed]
    with np.load(tree_dir / "predictions.npz") as z:
        meter = np.asarray(z["meter"]).astype(np.int8)
        keep = meter == HOT_WATER
        s_tree = np.asarray(z["ensemble"], dtype=np.float64)[keep]
        bid_chk = np.asarray(z["building_id"])[keep].astype(np.int32)
        y_chk = np.asarray(z["anomaly"])[keep].astype(np.int8)
    if not (np.array_equal(bid_chk, bid_hw) and np.array_equal(y_chk, y_hw)):
        raise RuntimeError(f"r{row_seed} tree holdout 與快取不符")

    if row_seed < 2:
        tab_dir = m5.cell_dir(formal_root, 725, row_seed, 725, "tabpfn")
        tab_path = tab_dir / "predictions.npz"
    else:
        tab_path = colab / f"model_results/row_seed{row_seed}/predictions.npz"
    with np.load(tab_path) as z:
        keep2 = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        s_tab = np.asarray(z["tabpfn"], dtype=np.float64)[keep2]
        bid2 = np.asarray(z["building_id"])[keep2].astype(np.int32)
        y2 = np.asarray(z["anomaly"])[keep2].astype(np.int8)
        rawi2 = np.asarray(z["validation_raw_index"])
    if not (np.array_equal(bid2, bid_hw) and np.array_equal(y2, y_hw)):
        raise RuntimeError(f"r{row_seed} tabpfn holdout 與快取不符")
    with np.load(tree_dir / "predictions.npz") as z:
        if not np.array_equal(np.asarray(z["validation_raw_index"]), rawi2):
            raise RuntimeError(f"r{row_seed} validation_raw_index 不符")
    if not np.isfinite(s_tab).all() or not np.isfinite(s_tree).all():
        raise RuntimeError(f"r{row_seed} 非有限分數")
    print(f"[gate] r{row_seed}: holdout 逐元素相符（building_id / anomaly / raw_index）")

    zero = reading == 0
    rec = {"row_seed": row_seed}
    rec["ap_tree"] = float(average_precision_score(y_hw, s_tree))
    rec["ap_tab"] = float(average_precision_score(y_hw, s_tab))
    rec["roc_tree"] = float(roc_auc_score(y_hw, s_tree))
    rec["roc_tab"] = float(roc_auc_score(y_hw, s_tab))

    # 零讀數 / 非零讀數 對全域 AP 落差的可加貢獻
    def contrib(scores):
        order = np.argsort(-scores, kind="mergesort")
        s, yy = scores[order], y_hw[order]
        tp = np.cumsum(yy)
        cnt = np.arange(1, len(yy) + 1)
        bnd = np.nonzero(np.r_[s[:-1] != s[1:], True])[0]
        last = bnd[np.searchsorted(bnd, np.arange(len(s)))]
        c = np.where(yy == 1, tp[last] / cnt[last] / n_pos, 0.0)
        out = np.empty(len(scores))
        out[order] = c
        return out

    d = contrib(s_tab) - contrib(s_tree)
    if abs(d.sum() - (rec["ap_tab"] - rec["ap_tree"])) > 1e-9:
        raise RuntimeError("拆解檢核失敗")
    rec["contrib_zero"] = float(d[zero].sum())
    rec["contrib_nonzero"] = float(d[~zero].sum())

    # 逐建物 AP 落差
    pb = {}
    for b in np.unique(bid_hw):
        sel = bid_hw == b
        yy = y_hw[sel]
        p = int(yy.sum())
        if p == 0 or p == len(yy):
            continue
        pb[int(b)] = float(average_precision_score(yy, s_tab[sel])
                           - average_precision_score(yy, s_tree[sel]))
    rec["per_building"] = pb
    results.append(rec)
    print(f"  r{row_seed}: PR-AUC tree={rec['ap_tree']:.4f} tab={rec['ap_tab']:.4f} "
          f"gap={rec['ap_tab'] - rec['ap_tree']:+.4f}  zero={rec['contrib_zero']:+.4f} "
          f"nonzero={rec['contrib_nonzero']:+.4f}")

(SCRATCH / "k725_five_seed_hotwater.json").write_text(json.dumps(results, indent=2))

gaps = [r["ap_tab"] - r["ap_tree"] for r in results]
print(f"\nK=725 (n=5 row seeds): hot water PR-AUC gap = {np.mean(gaps):+.4f} "
      f"± {np.std(gaps, ddof=1) / math.sqrt(5):.4f}")
print(f"  Tree PR-AUC  = {np.mean([r['ap_tree'] for r in results]):.4f}")
print(f"  TabPFN PR-AUC= {np.mean([r['ap_tab'] for r in results]):.4f}")
print(f"  Tree ROC-AUC = {np.mean([r['roc_tree'] for r in results]):.4f}")
print(f"  TabPFN ROC   = {np.mean([r['roc_tab'] for r in results]):.4f}")
z = [r["contrib_zero"] for r in results]
nz = [r["contrib_nonzero"] for r in results]
print(f"  zero-reading 貢獻 = {np.mean(z):+.4f} ± {np.std(z, ddof=1) / math.sqrt(5):.4f} "
      f"({np.mean(z) / np.mean(gaps) * 100:.0f}% of gap)")
print(f"  non-zero 貢獻     = {np.mean(nz):+.4f} ± {np.std(nz, ddof=1) / math.sqrt(5):.4f}")

# 建物分群
pz = {}
for b in np.unique(bid_hw):
    sel = bid_hw == b
    zz = reading[sel] == 0
    if int((y_hw[sel] == 1).sum()) == 0:
        continue
    pz[int(b)] = float((y_hw[sel][zz] == 1).mean()) if zz.any() else np.nan
print("\n依建物分群（每個 row seed 先算組內建物平均，再跨 5 個 seed）：")
for name, lo, hi in (("A <0.10", -1, 0.10), ("B 0.10-0.60", 0.10, 0.60), ("C >0.60", 0.60, 2)):
    per_seed = []
    for r in results:
        vals = [v for b, v in r["per_building"].items()
                if not math.isnan(pz.get(int(b), float("nan"))) and lo <= pz[int(b)] < hi]
        if vals:
            per_seed.append(float(np.mean(vals)))
    if per_seed:
        se = np.std(per_seed, ddof=1) / math.sqrt(len(per_seed)) if len(per_seed) > 1 else float("nan")
        n_b = sum(1 for b in pz if not math.isnan(pz[b]) and lo <= pz[b] < hi)
        print(f"  {name:<12} n_buildings={n_b:>2}  gap={np.mean(per_seed):+.4f} ± {se:.4f}")
