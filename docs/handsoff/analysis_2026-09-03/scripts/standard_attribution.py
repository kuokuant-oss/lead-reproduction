#!/usr/bin/env python3
"""hot water PR-AUC 落差的標準歸因分析（不自創指標）。

1. shift-share 恆等式        gap = sum_g w_g * mean_within_g   (Oaxaca-Blinder 的 structure 項)
2. Shapley value 歸因        v(S) = AP_tab(S) - AP_tree(S)，等同 general dominance
3. complete dominance        逐對比較邊際貢獻是否在所有 coalition 上都占優
4. slice 報表                Slice Finder / SliceLine 式：size、組內效果、size-weighted score
5. cluster bootstrap         以 building 為 cluster，給落差的信賴區間
6. 一階隨機優越 + Wilcoxon    逐建物 AP 分布：普遍性 vs 尾端

唯讀：只讀 predictions.npz / cell.json，輸出寫在 scratchpad。
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
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
HOT_WATER = 3
RNG = np.random.default_rng(20260902)
N_BOOT = 300

spec = importlib.util.spec_from_file_location("m5plot", PLOT)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y = side["y"].astype(np.int8)
bid = side["bid"].astype(np.int32)
sid = side["sid"].astype(np.int32)
reading = side["reading"]
use = side["use"].astype(str)
n_rows, n_pos = len(y), int(y.sum())

p_anom_zero = {}
for b in np.unique(bid):
    sel = bid == b
    z = reading[sel] == 0
    p_anom_zero[int(b)] = float((y[sel][z] == 1).mean()) if z.any() else np.nan
pz = np.array([p_anom_zero[int(b)] for b in bid])
btype = np.where(np.isnan(pz), "D_no_zero",
                 np.where(pz < 0.10, "A_zero_mostly_normal",
                          np.where(pz < 0.60, "B_mixed", "C_zero_is_anomaly")))
segment = np.where(reading == 0, "zero/", "pos/")
segment = np.char.add(segment.astype(str), btype)

DIMENSIONS = {
    "segment": segment,
    "site": np.char.add("site", sid.astype(str)),
    "use": use,
}
CELLS = [(k, b, r) for k in (100, 200, 400) for b in range(5) for r in (0, 1)]
CELLS += [(725, 725, 0), (725, 725, 1)]
BUDGETS = [100, 200, 400, 725]
SHAPLEY_DIMS = {"segment": BUDGETS, "site": [400, 725], "use": [400, 725]}

buildings = np.unique(bid)
b_index = {int(b): np.nonzero(bid == b)[0] for b in buildings}


def ap(mask, s):
    yy = y[mask]
    p = int(yy.sum())
    if p == 0 or p == len(yy):
        return float("nan")
    return float(average_precision_score(yy, s[mask]))


def per_positive_precision(scores):
    """每個正樣本在其 tie-group 門檻處的 precision（AP 的標準定義：這些值的平均即 AP）。"""
    order = np.argsort(-scores, kind="mergesort")
    s, yy = scores[order], y[order]
    tp = np.cumsum(yy)
    cnt = np.arange(1, len(yy) + 1)
    boundary = np.nonzero(np.r_[s[:-1] != s[1:], True])[0]
    last = boundary[np.searchsorted(boundary, np.arange(len(s)))]
    prec = np.where(yy == 1, tp[last] / cnt[last], np.nan)
    out = np.full(n_rows, np.nan)
    out[order] = prec
    return out


def shapley(values: dict[frozenset, float], members: list[str]) -> dict[str, float]:
    n = len(members)
    phi = {m: 0.0 for m in members}
    for m in members:
        others = [o for o in members if o != m]
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for combo in itertools.combinations(others, size):
                S = frozenset(combo)
                phi[m] += weight * (values[S | {m}] - values[S])
    return phi


def marginals(values: dict[frozenset, float], members: list[str]) -> dict[str, dict[int, float]]:
    """各成員在每個 coalition size 上的平均邊際貢獻（Azen & Budescu 的 conditional dominance）。"""
    out = {}
    n = len(members)
    for m in members:
        others = [o for o in members if o != m]
        by_size = {}
        for size in range(n):
            vals = [values[frozenset(c) | {m}] - values[frozenset(c)]
                    for c in itertools.combinations(others, size)]
            by_size[size] = float(np.mean(vals))
        out[m] = by_size
    return out


results = {"cells": [], "shapley": {}, "shift_share": {}, "bootstrap": {}, "dominance": {}}
per_building_ap = {}

for budget, bseed, rseed in CELLS:
    label = f"k{budget}_b{bseed}_r{rseed}"
    dirs = {m: m5.cell_dir(ROOT, bseed, rseed, budget, m) for m in ("ensemble", "tabpfn")}
    if not all(m5.is_complete(d) for d in dirs.values()):
        continue
    meta = {m: m5.validate_cell(d, budget, bseed, rseed, m) for m, d in dirs.items()}
    m5.validate_pair(meta["ensemble"], meta["tabpfn"], label)
    s = {}
    for model, d in dirs.items():
        with np.load(d / "predictions.npz") as payload:
            keep = np.asarray(payload["meter"]).astype(np.int8) == HOT_WATER
            s[model] = np.asarray(payload[model], dtype=np.float64)[keep]
    full = np.ones(n_rows, dtype=bool)
    gap = ap(full, s["tabpfn"]) - ap(full, s["ensemble"])

    rec = {"budget": budget, "building_seed": bseed, "row_seed": rseed, "gap": gap,
           "ap_tree": ap(full, s["ensemble"]), "ap_tab": ap(full, s["tabpfn"])}

    # 1) shift-share：AP 是「正樣本在其排名處 precision」的平均，故均值差可精確分解
    prec = {m: per_positive_precision(s[m]) for m in ("ensemble", "tabpfn")}
    dprec = prec["tabpfn"] - prec["ensemble"]
    posmask = y == 1
    rec["shift_share"] = {}
    for dim, values in DIMENSIONS.items():
        rows = []
        for g in sorted(set(values[posmask])):
            sel = posmask & (values == g)
            w = float(sel.sum() / n_pos)
            dbar = float(np.mean(dprec[sel]))
            rows.append({"group": str(g), "w": w, "mean_within": dbar, "contribution": w * dbar})
        rec["shift_share"][dim] = rows

    # 2) Shapley / dominance
    rec["coalitions"] = {}
    for dim, budgets in SHAPLEY_DIMS.items():
        if budget not in budgets:
            continue
        values = DIMENSIONS[dim]
        members = sorted({g for g in set(values[posmask])})
        vals = {}
        for size in range(len(members) + 1):
            for combo in itertools.combinations(members, size):
                S = frozenset(combo)
                if not S:
                    vals[S] = 0.0
                    continue
                mask = np.isin(values, list(S))
                a_t, a_p = ap(mask, s["ensemble"]), ap(mask, s["tabpfn"])
                vals[S] = 0.0 if (math.isnan(a_t) or math.isnan(a_p)) else a_p - a_t
        rec["coalitions"][dim] = {"members": members,
                                  "values": {"|".join(sorted(k)): v for k, v in vals.items()}}

    # 3) 逐建物 AP（隨機優越用）
    pb = {}
    for b, idx in b_index.items():
        mask = np.zeros(n_rows, dtype=bool)
        mask[idx] = True
        a_t, a_p = ap(mask, s["ensemble"]), ap(mask, s["tabpfn"])
        if not (math.isnan(a_t) or math.isnan(a_p)):
            pb[b] = (a_t, a_p)
    per_building_ap[label] = pb

    # 4) cluster bootstrap（以 building 為 cluster）
    if budget in (400, 725):
        boots = []
        for _ in range(N_BOOT):
            pick = RNG.choice(buildings, size=len(buildings), replace=True)
            idx = np.concatenate([b_index[int(b)] for b in pick])
            yy = y[idx]
            if not 0 < int(yy.sum()) < len(yy):
                continue
            g = (average_precision_score(yy, s["tabpfn"][idx])
                 - average_precision_score(yy, s["ensemble"][idx]))
            boots.append(float(g))
        rec["bootstrap"] = {"n": len(boots), "mean": float(np.mean(boots)),
                            "lo": float(np.percentile(boots, 2.5)),
                            "hi": float(np.percentile(boots, 97.5))}

    results["cells"].append(rec)
    print(f"  {label}: gap={gap:+.4f}", flush=True)
    del s, prec, dprec

json.dump(results, (SCRATCH / "standard_attribution.json").open("w"), indent=2)
json.dump({k: {str(b): v for b, v in d.items()} for k, d in per_building_ap.items()},
          (SCRATCH / "per_building_ap.json").open("w"), indent=2)
print("\n[saved] standard_attribution.json / per_building_ap.json", flush=True)
