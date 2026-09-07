#!/usr/bin/env python3
"""輸出標準歸因報表：Shapley/dominance、shift-share、SliceLine 式 slice 報表、
cluster bootstrap CI、一階隨機優越 + 配對 Wilcoxon。"""
from __future__ import annotations

import itertools
import json
import math
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRATCH = Path(
    "/mnt/c/Users/User/AppData/Local/Temp/claude/C--Users-User-projects-lead-reproduction-temp"
    "/4888b221-2e78-4bf8-bfd6-7f21071cc05b/scratchpad"
)
res = json.loads((SCRATCH / "standard_attribution.json").read_text())
pb = json.loads((SCRATCH / "per_building_ap.json").read_text())
cells = res["cells"]
BUDGETS = [100, 200, 400, 725]
ALPHA = 0.95  # SliceLine 評分中誤差重要性與 slice 大小的權衡


def bseed_mean(pairs):
    by_b = {}
    for (b, _), v in pairs:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        by_b.setdefault(b, []).append(v)
    return float(np.mean([np.mean(v) for v in by_b.values()])) if by_b else float("nan")


def bseed_se(pairs):
    by_b = {}
    for (b, _), v in pairs:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        by_b.setdefault(b, []).append(v)
    means = [float(np.mean(v)) for v in by_b.values()]
    if len(means) < 2:
        return float("nan")
    return float(np.std(means, ddof=1) / math.sqrt(len(means)))


print("=" * 100)
print("A. 全域落差與 cluster bootstrap 信賴區間（以 building 為 cluster）")
print("=" * 100)
rows = []
for k in BUDGETS:
    sub = [c for c in cells if c["budget"] == k]
    gap = bseed_mean([((c["building_seed"], c["row_seed"]), c["gap"]) for c in sub])
    se = bseed_se([((c["building_seed"], c["row_seed"]), c["gap"]) for c in sub])
    boots = [c["bootstrap"] for c in sub if "bootstrap" in c]
    row = {"K": k, "cells": len(sub), "gap": gap, "seed_SE": se}
    if boots:
        row["boot_lo"] = float(np.mean([b["lo"] for b in boots]))
        row["boot_hi"] = float(np.mean([b["hi"] for b in boots]))
    rows.append(row)
with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
    print(pd.DataFrame(rows).to_string(index=False))
print("（cluster bootstrap 為各 cell 的 95% 區間平均；同一棟建物的列不獨立，故以建物為重抽單位）")

print()
print("=" * 100)
print("B. shift-share 分解：gap = Σ w_g × Δ̄_g")
print("   w_g = 該組正樣本占比；Δ̄_g = 組內每個正樣本在其排名處 precision 的平均變化")
print("   兩模型面對同一 holdout，composition 完全相同 → Oaxaca–Blinder 的 endowment 項為 0，")
print("   全部落差皆為 structure 項。")
print("=" * 100)
for dim in ("segment", "site", "use"):
    keys = sorted({g["group"] for c in cells for g in c["shift_share"][dim]})
    table = {}
    for key in keys:
        col = {}
        for k in BUDGETS:
            sub = [c for c in cells if c["budget"] == k]
            pairs_w, pairs_d, pairs_c = [], [], []
            for c in sub:
                g = next((g for g in c["shift_share"][dim] if g["group"] == key), None)
                if g is None:
                    continue
                sk = (c["building_seed"], c["row_seed"])
                pairs_w.append((sk, g["w"]))
                pairs_d.append((sk, g["mean_within"]))
                pairs_c.append((sk, g["contribution"]))
            col["w"] = bseed_mean(pairs_w)
            col[f"dbar_K{k}"] = bseed_mean(pairs_d)
            col[f"contrib_K{k}"] = bseed_mean(pairs_c)
        table[key] = col
    df = pd.DataFrame(table).T.sort_values("contrib_K400")
    cols = ["w"] + [c for k in BUDGETS for c in (f"dbar_K{k}", f"contrib_K{k}")]
    print(f"\n--- 維度：{dim} ---")
    with pd.option_context("display.width", 260, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:+.4f}"):
        print(df[cols].to_string())
    for k in BUDGETS:
        tot = df[f"contrib_K{k}"].sum()
        print(f"  檢核 K={k}: Σ contribution = {tot:+.6f}")

print()
print("=" * 100)
print("C. Shapley value 歸因（= general dominance）")
print("   v(S) = AP_TabPFN(S) − AP_Tree(S)，S 為 slice 的聯集；v(∅)=0")
print("   效率性公理保證 Σ φ_g = v(全體) = 全域落差")
print("=" * 100)


def shapley_from(values: dict[frozenset, float], members: list[str]):
    n = len(members)
    phi = {m: 0.0 for m in members}
    cond = {m: {} for m in members}
    for m in members:
        others = [o for o in members if o != m]
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            vals = []
            for combo in itertools.combinations(others, size):
                S = frozenset(combo)
                vals.append(values[S | {m}] - values[S])
            phi[m] += weight * float(np.sum(vals))
            cond[m][size] = float(np.mean(vals))
    return phi, cond


def decode(cell, dim):
    co = cell["coalitions"][dim]
    members = co["members"]
    values = {frozenset(k.split("|")) if k else frozenset(): v
              for k, v in co["values"].items()}
    values[frozenset()] = 0.0
    return members, values


for dim in ("segment", "site", "use"):
    for k in BUDGETS:
        sub = [c for c in cells if c["budget"] == k and dim in c.get("coalitions", {})]
        if not sub:
            continue
        members = decode(sub[0], dim)[0]
        phis, conds = [], []
        for c in sub:
            _, values = decode(c, dim)
            phi, cond = shapley_from(values, members)
            phis.append(((c["building_seed"], c["row_seed"]), phi))
            conds.append(cond)
        rows = []
        for m in members:
            rows.append({
                "slice": m,
                "shapley": bseed_mean([(sk, p[m]) for sk, p in phis]),
                "SE": bseed_se([(sk, p[m]) for sk, p in phis]),
            })
        df = pd.DataFrame(rows).sort_values("shapley")
        total = df["shapley"].sum()
        df["share_%"] = df["shapley"] / total * 100
        gap = bseed_mean([((c["building_seed"], c["row_seed"]), c["gap"]) for c in sub])
        print(f"\n--- {dim} @ K={k}  (cells={len(sub)}) ---")
        with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
            print(df.to_string(index=False))
        print(f"  Σφ = {total:+.6f}   全域落差 = {gap:+.6f}   (效率性檢核)")

        # conditional dominance：各 coalition size 的平均邊際貢獻
        sizes = sorted(conds[0][members[0]].keys())
        cd = pd.DataFrame({m: {f"|S|={s}": float(np.mean([c[m][s] for c in conds]))
                               for s in sizes} for m in members}).T
        print("  conditional dominance（各 coalition 規模下的平均邊際貢獻）:")
        with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:+.4f}"):
            print("  " + cd.to_string().replace("\n", "\n  "))
        # complete dominance：A 在所有規模上都比 B 更負 → A 完全支配 B（更主要的落差來源）
        doms = []
        for a, b in itertools.permutations(members, 2):
            if all(cd.loc[a, f"|S|={s}"] < cd.loc[b, f"|S|={s}"] for s in sizes):
                doms.append(f"{a} ≻ {b}")
        print(f"  complete dominance（在所有 coalition 規模上皆為更大的落差來源）: "
              f"{len(doms)} 組")
        for d in doms[:12]:
            print(f"    {d}")

print()
print("=" * 100)
print("D. SliceLine 式 slice 報表")
print("   sc = α·((se/|S|)/(e/n) − 1) − (1−α)·(n/|S| − 1)，α = 0.95")
print("   se = slice 內的 precision 損失總和（Tree − TabPFN，正值為損失），|S| = slice 內正樣本數")
print("=" * 100)
for dim in ("segment", "site", "use"):
    keys = sorted({g["group"] for c in cells for g in c["shift_share"][dim]})
    for k in (400, 725):
        sub = [c for c in cells if c["budget"] == k]
        if not sub:
            continue
        n_all = 1.0  # 以占比表示
        e_all = -bseed_mean([((c["building_seed"], c["row_seed"]), c["gap"]) for c in sub])
        rows = []
        for key in keys:
            w = bseed_mean([((c["building_seed"], c["row_seed"]),
                             next(g["w"] for g in c["shift_share"][dim] if g["group"] == key))
                            for c in sub if any(g["group"] == key for g in c["shift_share"][dim])])
            se_ = -bseed_mean([((c["building_seed"], c["row_seed"]),
                                next(g["contribution"] for g in c["shift_share"][dim]
                                     if g["group"] == key))
                               for c in sub if any(g["group"] == key for g in c["shift_share"][dim])])
            if not w or math.isnan(w) or w == 0:
                continue
            avg_ratio = (se_ / w) / (e_all / n_all) if e_all else float("nan")
            sc = ALPHA * (avg_ratio - 1) - (1 - ALPHA) * (n_all / w - 1)
            rows.append({"slice": key, "size_w": w, "slice_loss": se_,
                         "avg_loss_ratio": avg_ratio, "sliceline_score": sc})
        df = pd.DataFrame(rows).sort_values("sliceline_score", ascending=False)
        print(f"\n--- {dim} @ K={k}  (總損失 e = {e_all:+.4f}) ---")
        with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
            print(df.to_string(index=False))

print()
print("=" * 100)
print("E. 一階隨機優越與配對檢定：逐建物 AP 分布")
print("=" * 100)
for k in BUDGETS:
    labels = [f"k{k}_b{c['building_seed']}_r{c['row_seed']}" for c in cells if c["budget"] == k]
    tree_all, tab_all = [], []
    for lab in labels:
        d = pb.get(lab, {})
        for b, (a_t, a_p) in d.items():
            tree_all.append((lab, int(b), a_t))
            tab_all.append((lab, int(b), a_p))
    dft = pd.DataFrame(tree_all, columns=["cell", "b", "tree"])
    dfp = pd.DataFrame(tab_all, columns=["cell", "b", "tab"])
    df = dft.merge(dfp, on=["cell", "b"])
    # 先在 cell 間平均，得到每棟一個值
    per_b = df.groupby("b")[["tree", "tab"]].mean()
    t, p_ = per_b["tree"].to_numpy(), per_b["tab"].to_numpy()
    grid = np.linspace(0, 1, 501)
    F_t = np.array([(t <= g).mean() for g in grid])
    F_p = np.array([(p_ <= g).mean() for g in grid])
    d_tree_dom = float(np.max(F_p - F_t))   # >0 且處處 ≥0 → Tree 一階優越
    d_tab_dom = float(np.max(F_t - F_p))
    w = stats.wilcoxon(p_, t, zero_method="wilcox", alternative="two-sided")
    print(f"\nK={k}  建物數={len(per_b)}")
    print(f"  逐建物 AP 中位數  Tree={np.median(t):.4f}  TabPFN={np.median(p_):.4f}")
    print(f"  分位數 (10/25/50/75/90):")
    for name, arr in (("Tree", t), ("TabPFN", p_)):
        qs = np.percentile(arr, [10, 25, 50, 75, 90])
        print(f"    {name:<7}" + "  ".join(f"{q:.4f}" for q in qs))
    print(f"  ECDF 最大差  sup(F_tab − F_tree) = {d_tree_dom:+.4f}（>0 表示該處 Tree 較優）")
    print(f"               sup(F_tree − F_tab) = {d_tab_dom:+.4f}")
    verdict = ("Tree 一階隨機優越 TabPFN" if d_tab_dom <= 1e-12 else
               "TabPFN 一階隨機優越 Tree" if d_tree_dom <= 1e-12 else
               "無一階隨機優越（ECDF 交叉）")
    print(f"  判定：{verdict}")
    print(f"  配對 Wilcoxon signed-rank (TabPFN − Tree): W={w.statistic:.0f}  p={w.pvalue:.3g}  "
          f"中位差={np.median(p_ - t):+.4f}")
