#!/usr/bin/env python3
"""檢驗假說：TabPFN 在「reading=0 不代表異常」的建物上持續失效，且 K=725 也修不好。"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRATCH = Path(
    "/mnt/c/Users/User/AppData/Local/Temp/claude/C--Users-User-projects-lead-reproduction-temp"
    "/4888b221-2e78-4bf8-bfd6-7f21071cc05b/scratchpad"
)
records = json.loads((SCRATCH / "hotwater_gap_by_budget.json").read_text(encoding="utf-8"))
side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
bid, y, reading = side["bid"], side["y"], side["reading"]
BUDGETS = [100, 200, 400, 725]


def bseed_mean(vals):
    by_b = {}
    for (b, _), v in vals.items():
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            by_b.setdefault(b, []).append(v)
    return float(np.mean([np.mean(v) for v in by_b.values()])) if by_b else float("nan")


rows = []
for b in sorted({b for r in records for b in r["groups"]["building"]}, key=int):
    row = {"building": int(b)}
    for k in BUDGETS:
        vals = {}
        for r in records:
            if r["budget"] != k:
                continue
            g = r["groups"]["building"].get(b)
            if not g or g["pos"] == 0:
                continue
            a, t = g["ap_tree"], g["ap_tab"]
            if a is not None and t is not None and not math.isnan(a) and not math.isnan(t):
                vals[(r["building_seed"], r["row_seed"])] = t - a
                row["pos"] = g["pos"]
        row[k] = bseed_mean(vals)
    if "pos" in row:
        rows.append(row)
df = pd.DataFrame(rows)

m = bid[:, None] == df["building"].to_numpy()[None, :]
stat = []
for i, b in enumerate(df["building"]):
    sel = bid == b
    z = reading[sel] == 0
    yy = y[sel] == 1
    stat.append({
        "zero_rate": float(z.mean()),
        "prev": float(yy.mean()),
        # 在 reading=0 的列中，真的是異常的比例（"零即異常" 這條全域規則在該棟的正確率）
        "p_anom_given_zero": float(yy[z].mean()) if z.any() else np.nan,
        # 異常中有多少落在 reading=0（該棟異常是否靠零值表現）
        "p_zero_given_anom": float(z[yy].mean()) if yy.any() else np.nan,
    })
df = pd.concat([df, pd.DataFrame(stat)], axis=1)
df["misleading_zeros"] = df["zero_rate"] * (1 - df["p_anom_given_zero"])

print("=== 相關性：建物特徵 vs TabPFN−Tree 落差 ===")
print(f"{'特徵':<28}" + "".join(f"{'K=' + str(k):>12}" for k in BUDGETS))
for feat in ("p_anom_given_zero", "p_zero_given_anom", "misleading_zeros", "prev", "zero_rate"):
    cells = []
    for k in BUDGETS:
        sub = df[[feat, k]].dropna()
        r, p = stats.spearmanr(sub[feat], sub[k])
        cells.append(f"{r:>+8.3f}{'*' if p < 0.05 else ' '}   ")
    print(f"{feat:<28}" + "".join(cells))
print("(Spearman ρ，* = p<0.05；正值代表該特徵越高、TabPFN 相對越好)")

print("\n=== 依「reading=0 時實際是異常的機率」分組 ===")
df["zgrp"] = pd.cut(df["p_anom_given_zero"], [-0.01, 0.1, 0.3, 0.6, 1.01],
                    labels=["<10% (零多半正常)", "10-30%", "30-60%", ">60% (零幾乎都異常)"])
agg = df.groupby("zgrp", observed=True).agg(
    buildings=("building", "size"), pos=("pos", "sum"),
    **{f"K{k}": (k, "mean") for k in BUDGETS})
with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:+.4f}"):
    print(agg.to_string())

print("\n=== 全體 hot water：reading=0 的列中有多少真的是異常 ===")
z = reading == 0
print(f"整體 P(anomaly | reading=0) = {float((y[z] == 1).mean()):.4f}")
print(f"整體 P(reading=0 | anomaly) = {float((reading[y == 1] == 0).mean()):.4f}")
print(f"reading=0 但為正常的列數 = {int((z & (y == 0)).sum()):,}"
      f"（佔 hot water 全部列 {float((z & (y == 0)).mean()):.2%}）")
