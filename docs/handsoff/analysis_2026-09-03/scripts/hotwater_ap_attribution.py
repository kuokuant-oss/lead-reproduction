#!/usr/bin/env python3
"""hot water PR-AUC 落差的可加性拆解（誰貢獻最多）。

AP = (1/P) * sum_over_positives  precision@(該正樣本所屬 tie-group 的門檻)
每個正樣本的貢獻可加，故把貢獻依類別加總即為該類別對 AP 的占比。
delta_contribution = TabPFN 貢獻 − Tree 貢獻，加總等於全域 AP 落差（精確）。

只讀 predictions.npz / cell.json，不寫任何實驗輸出。
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PLOT = Path(
    "/mnt/c/Users/User/Documents/Codex/2026-09-01/wsl-ubuntu-m5-building-count-v5"
    "/outputs/scripts/plot_m5_v5_fixed50k_building_scarcity_roc.py"
)
ROOT = Path(
    "/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k/data/processed"
    "/m5_building_curve/v5_fixed_50k/model_runs"
)
SCRATCH = Path(
    "/mnt/c/Users/User/AppData/Local/Temp/claude/C--Users-User-projects-lead-reproduction-temp"
    "/4888b221-2e78-4bf8-bfd6-7f21071cc05b/scratchpad"
)
META = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/building_metadata.csv")
HOT_WATER = 3

spec = importlib.util.spec_from_file_location("m5plot", PLOT)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y = side["y"].astype(np.int8)
bid = side["bid"].astype(np.int32)
sid = side["sid"].astype(np.int32)
reading = side["reading"]
use = side["use"].astype(str)
n_rows = len(y)
n_pos = int(y.sum())

# 每棟的「零讀數真的是異常」的機率 → 建物型態分群
p_anom_zero = {}
for b in np.unique(bid):
    sel = bid == b
    z = reading[sel] == 0
    p_anom_zero[int(b)] = float((y[sel][z] == 1).mean()) if z.any() else np.nan
pz = np.array([p_anom_zero[int(b)] for b in bid])
btype = np.where(np.isnan(pz), "no_zero",
                 np.where(pz < 0.10, "A_零多半正常",
                          np.where(pz < 0.60, "B_混合", "C_零幾乎都異常")))

is_zero = reading == 0
segment = np.char.add(np.where(is_zero, "reading=0 / ", "reading>0 / "), btype)

# reading 分箱
bins = np.full(n_rows, "zero", dtype=object)
posr = reading > 0
q = pd.qcut(reading[posr], 10, labels=False, duplicates="drop")
bins[posr] = [f"d{int(v)}" for v in q]
BIN_ORDER = ["zero"] + [f"d{i}" for i in range(10)]

GROUPS = {
    "site": sid.astype(str),
    "use": use,
    "reading_bin": np.asarray(bins, dtype=str),
    "building": bid.astype(str),
    "segment": segment,
    "btype": btype,
}


def ap_contributions(scores: np.ndarray) -> np.ndarray:
    """回傳長度 n_rows 的陣列：正樣本填其對 AP 的貢獻，其餘 0。總和 = sklearn AP。"""
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    yy = y[order]
    tp = np.cumsum(yy)
    cnt = np.arange(1, len(yy) + 1)
    # tie-group 結尾位置
    last = np.empty(len(s), dtype=np.int64)
    boundary = np.nonzero(np.r_[s[:-1] != s[1:], True])[0]
    idx = np.searchsorted(boundary, np.arange(len(s)))
    last = boundary[idx]
    prec_at_group_end = tp[last] / cnt[last]
    contrib_sorted = np.where(yy == 1, prec_at_group_end / n_pos, 0.0)
    out = np.empty(n_rows, dtype=np.float64)
    out[order] = contrib_sorted
    return out


CELLS = [(k, b, r) for k in (100, 200, 400) for b in range(5) for r in (0, 1)]
CELLS += [(725, 725, 0), (725, 725, 1)]

per_cell = []
for budget, bseed, rseed in CELLS:
    label = f"k{budget}_b{bseed}_r{rseed}"
    dirs = {m: m5.cell_dir(ROOT, bseed, rseed, budget, m) for m in ("ensemble", "tabpfn")}
    if not all(m5.is_complete(d) for d in dirs.values()):
        continue
    meta = {m: m5.validate_cell(d, budget, bseed, rseed, m) for m, d in dirs.items()}
    m5.validate_pair(meta["ensemble"], meta["tabpfn"], label)
    contrib = {}
    for model, d in dirs.items():
        with np.load(d / "predictions.npz") as payload:
            keep = np.asarray(payload["meter"]).astype(np.int8) == HOT_WATER
            s = np.asarray(payload[model], dtype=np.float64)[keep]
        c = ap_contributions(s)
        ref = average_precision_score(y, s)
        if abs(c.sum() - ref) > 1e-9:
            raise RuntimeError(f"AP 拆解不符 {label} {model}: {c.sum()} vs {ref}")
        contrib[model] = c
    delta = contrib["tabpfn"] - contrib["ensemble"]
    row = {"budget": budget, "building_seed": bseed, "row_seed": rseed,
           "ap_tree": float(contrib["ensemble"].sum()), "ap_tab": float(contrib["tabpfn"].sum())}
    for gname, gvals in GROUPS.items():
        agg = pd.Series(delta).groupby(pd.Series(gvals)).sum()
        agg_tree = pd.Series(contrib["ensemble"]).groupby(pd.Series(gvals)).sum()
        agg_tab = pd.Series(contrib["tabpfn"]).groupby(pd.Series(gvals)).sum()
        row[gname] = {str(k): {"delta": float(v), "tree": float(agg_tree[k]),
                               "tab": float(agg_tab[k])} for k, v in agg.items()}
    per_cell.append(row)
    print(f"  {label}: AP tree={row['ap_tree']:.4f} tab={row['ap_tab']:.4f} "
          f"delta={row['ap_tab'] - row['ap_tree']:+.4f} (拆解檢核通過)")

(SCRATCH / "hotwater_ap_attribution.json").write_text(
    json.dumps(per_cell, indent=2), encoding="utf-8")
print(f"\n[saved] hotwater_ap_attribution.json  cells={len(per_cell)}")

# ---------------- 彙總 ----------------
BUDGETS = [100, 200, 400, 725]


def bseed_mean(vals: dict[tuple[int, int], float]) -> float:
    by_b: dict[int, list[float]] = {}
    for (b, _), v in vals.items():
        by_b.setdefault(b, []).append(v)
    return float(np.mean([np.mean(v) for v in by_b.values()])) if by_b else float("nan")


pos_share = {}
for gname, gvals in GROUPS.items():
    s = pd.Series(y).groupby(pd.Series(gvals)).sum()
    pos_share[gname] = (s / n_pos).to_dict()


def report(gname: str, title: str, order=None, top=None) -> None:
    keys = sorted({k for r in per_cell for k in r[gname]})
    table = {}
    for key in keys:
        col = {}
        for k in BUDGETS:
            sub = [r for r in per_cell if r["budget"] == k]
            col[f"delta_K{k}"] = bseed_mean(
                {(r["building_seed"], r["row_seed"]): r[gname].get(key, {"delta": 0.0})["delta"]
                 for r in sub})
        col["正樣本占比"] = pos_share[gname].get(key, 0.0)
        table[key] = col
    df = pd.DataFrame(table).T
    for k in BUDGETS:
        total = df[f"delta_K{k}"].sum()
        df[f"占落差%_K{k}"] = df[f"delta_K{k}"] / total * 100 if total else np.nan
    df = df.reindex([o for o in order if o in df.index]) if order else df.sort_values("delta_K400")
    if top:
        df = df.head(top)
    cols = ["正樣本占比"] + [c for k in BUDGETS for c in (f"delta_K{k}", f"占落差%_K{k}")]
    print(f"\n=== {title} ===")
    with pd.option_context("display.width", 260, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:+.4f}"):
        print(df[cols].to_string())


print("\n各 K 的全域落差（貢獻加總，與前面一致）：")
for k in BUDGETS:
    sub = [r for r in per_cell if r["budget"] == k]
    g = bseed_mean({(r["building_seed"], r["row_seed"]): r["ap_tab"] - r["ap_tree"] for r in sub})
    print(f"  K={k}: {g:+.4f}")

report("segment", "落差來源占比：reading 是否為零 × 建物型態")
report("btype", "落差來源占比：建物型態")
report("reading_bin", "落差來源占比：依 meter_reading 分箱", order=BIN_ORDER)
report("site", "落差來源占比：依 site")
report("use", "落差來源占比：依 primary_use")
report("building", "落差來源占比：貢獻最負的 12 棟建物", top=12)
