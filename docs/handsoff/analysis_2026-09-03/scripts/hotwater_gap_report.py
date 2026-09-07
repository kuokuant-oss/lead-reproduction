#!/usr/bin/env python3
"""彙總 hotwater_gap_by_budget.json：先在 building seed 內平均 row seed，再跨 building seed。"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = Path(
    "/mnt/c/Users/User/AppData/Local/Temp/claude/C--Users-User-projects-lead-reproduction-temp"
    "/4888b221-2e78-4bf8-bfd6-7f21071cc05b/scratchpad"
)
records = json.loads((SCRATCH / "hotwater_gap_by_budget.json").read_text(encoding="utf-8"))
BUDGETS = [100, 200, 400, 725]
BIN_ORDER = ["zero"] + [f"d{i}" for i in range(10)]


def seed_level(values: dict[tuple[int, int], float]) -> tuple[float, float, int]:
    """values: {(bseed, rseed): v} -> (mean over building seeds, SE, n_building_seeds)"""
    by_b: dict[int, list[float]] = {}
    for (b, _), v in values.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        by_b.setdefault(b, []).append(v)
    means = [float(np.mean(vs)) for vs in by_b.values()]
    if not means:
        return float("nan"), float("nan"), 0
    if len(means) == 1:
        return means[0], float("nan"), 1
    sd = float(np.std(means, ddof=1))
    return float(np.mean(means)), sd / math.sqrt(len(means)), len(means)


print("=== hot water 全域 PR-AUC / ROC-AUC（先組內平均 row seed，再跨 building seed）===")
rows = []
for k in BUDGETS:
    sub = [r for r in records if r["budget"] == k]
    if not sub:
        continue
    tree = seed_level({(r["building_seed"], r["row_seed"]): r["ap_tree"] for r in sub})
    tab = seed_level({(r["building_seed"], r["row_seed"]): r["ap_tab"] for r in sub})
    gap = seed_level({(r["building_seed"], r["row_seed"]): r["ap_tab"] - r["ap_tree"] for r in sub})
    rtree = seed_level({(r["building_seed"], r["row_seed"]): r["roc_tree"] for r in sub})
    rtab = seed_level({(r["building_seed"], r["row_seed"]): r["roc_tab"] for r in sub})
    neg = sum(1 for r in sub if r["ap_tab"] < r["ap_tree"])
    rows.append({"K": k, "cells": len(sub), "n_bseed": tree[2],
                 "PR_tree": tree[0], "PR_tab": tab[0], "PR_gap": gap[0], "PR_gap_SE": gap[1],
                 "ROC_tree": rtree[0], "ROC_tab": rtab[0],
                 "cells_tabpfn_worse": f"{neg}/{len(sub)}"})
with pd.option_context("display.width", 220, "display.float_format", lambda v: f"{v:.4f}"):
    print(pd.DataFrame(rows).to_string(index=False))


def group_table(name: str, order=None, min_pos: int = 1) -> None:
    print(f"\n=== 依 {name}：TabPFN − Tree 的組內 PR-AUC 落差（跨 building seed 平均 ± SE）===")
    keys = sorted({k for r in records for k in r["groups"][name]})
    table = {}
    info = {}
    for key in keys:
        col = {}
        for k in BUDGETS:
            sub = [r for r in records if r["budget"] == k and key in r["groups"][name]]
            vals = {}
            for r in sub:
                g = r["groups"][name][key]
                if g["pos"] < min_pos:
                    continue
                a, b = g["ap_tree"], g["ap_tab"]
                if a is None or b is None or math.isnan(a) or math.isnan(b):
                    continue
                vals[(r["building_seed"], r["row_seed"])] = b - a
                info[key] = (g["n"], g["pos"])
            m, se, nb = seed_level(vals)
            col[f"K={k}"] = m
            col[f"SE{k}"] = se
        table[key] = col
    df = pd.DataFrame(table).T
    df.insert(0, "pos", [info.get(k, (0, 0))[1] for k in df.index])
    df.insert(0, "rows", [info.get(k, (0, 0))[0] for k in df.index])
    if order:
        df = df.reindex([o for o in order if o in df.index])
    else:
        df = df.sort_values("K=400")
    cols = ["rows", "pos"] + [c for k in BUDGETS for c in (f"K={k}", f"SE{k}")]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:+.4f}"):
        print(df[cols].to_string())


group_table("site")
group_table("use")
group_table("reading_bin", order=BIN_ORDER)

print("\n=== 集中度：leave-one-building-out 對全域落差的回復量（K 內跨 cell 平均）===")
print("（正值＝移除該棟後 TabPFN 的劣勢縮小；總落差 = 上表 PR_gap）")
for k in BUDGETS:
    sub = [r for r in records if r["budget"] == k]
    if not sub:
        continue
    keys = sorted({b for r in sub for b in r["loo_recovered"]}, key=int)
    means = {b: float(np.mean([r["loo_recovered"][b] for r in sub if b in r["loo_recovered"]]))
             for b in keys}
    ser = pd.Series(means).sort_values(ascending=False)
    base = float(np.mean([r["ap_tab"] - r["ap_tree"] for r in sub]))
    top5 = ser.head(5)
    print(f"\nK={k}  平均總落差={base:+.4f}  73 棟 LOO 回復量總和={ser.sum():+.4f}")
    print(f"  最大 5 棟合計回復 {top5.sum():+.4f}（佔總落差 {abs(top5.sum() / base):.1%}）")
    print("  " + "  ".join(f"b{b}:{v:+.4f}" for b, v in top5.items()))

print("\n=== 各 K 下組內落差最大的 8 棟建物 ===")
for k in BUDGETS:
    sub = [r for r in records if r["budget"] == k]
    if not sub:
        continue
    keys = sorted({b for r in sub for b in r["groups"]["building"]}, key=int)
    rows = []
    for b in keys:
        vals = {}
        n = p = 0
        for r in sub:
            g = r["groups"]["building"].get(b)
            if not g or g["pos"] == 0:
                continue
            a, t = g["ap_tree"], g["ap_tab"]
            if a is None or t is None or math.isnan(a) or math.isnan(t):
                continue
            vals[(r["building_seed"], r["row_seed"])] = t - a
            n, p = g["n"], g["pos"]
        m, se, nb = seed_level(vals)
        if not math.isnan(m):
            rows.append({"building": int(b), "rows": n, "pos": p, "delta": m, "SE": se})
    df = pd.DataFrame(rows).sort_values("delta")
    print(f"\nK={k}")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(df.head(8).to_string(index=False))
