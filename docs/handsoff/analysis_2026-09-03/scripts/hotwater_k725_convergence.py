#!/usr/bin/env python3
"""K=725 是否修復了 hot water 的落差：逐建物收斂診斷。"""
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
META = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/building_metadata.csv")
records = json.loads((SCRATCH / "hotwater_gap_by_budget.json").read_text(encoding="utf-8"))
side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
bid, y, reading = side["bid"], side["y"], side["reading"]
BUDGETS = [100, 200, 400, 725]


def bseed_mean(vals: dict[tuple[int, int], float]) -> float:
    by_b: dict[int, list[float]] = {}
    for (b, _), v in vals.items():
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            by_b.setdefault(b, []).append(v)
    return float(np.mean([np.mean(v) for v in by_b.values()])) if by_b else float("nan")


keys = sorted({b for r in records for b in r["groups"]["building"]}, key=int)
rows = []
for b in keys:
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
            if a is None or t is None or math.isnan(a) or math.isnan(t):
                continue
            vals[(r["building_seed"], r["row_seed"])] = t - a
            row["rows"], row["pos"] = g["n"], g["pos"]
        row[k] = bseed_mean(vals)
    if "pos" in row:
        rows.append(row)
df = pd.DataFrame(rows).dropna(subset=BUDGETS, how="all")

print("=== 建物層級落差的分布（有正樣本的建物）===")
print(f"{'K':>5}{'建物數':>8}{'落差<0':>9}{'落差<-0.05':>12}{'落差<-0.15':>12}"
      f"{'中位數':>10}{'第10百分位':>12}")
for k in BUDGETS:
    v = df[k].dropna()
    print(f"{k:>5}{len(v):>8}{int((v < 0).sum()):>9}{int((v < -0.05).sum()):>12}"
          f"{int((v < -0.15).sum()):>12}{v.median():>10.4f}{v.quantile(0.10):>12.4f}")

print("\n=== K=400 → K=725 的收斂：改善最多 / 完全沒改善 ===")
df["improve"] = df[725] - df[400]
bmeta = pd.read_csv(META).set_index("building_id")
df["use"] = bmeta["primary_use"].reindex(df["building"]).to_numpy()
df["site"] = bmeta["site_id"].reindex(df["building"]).to_numpy()
prev = {int(b): float(y[bid == b].mean()) for b in df["building"]}
zrate = {int(b): float((reading[bid == b] == 0).mean()) for b in df["building"]}
df["prev"] = df["building"].map(prev)
df["zero_rate"] = df["building"].map(zrate)
cols = ["building", "site", "use", "pos", "prev", "zero_rate", 100, 200, 400, 725, "improve"]
with pd.option_context("display.width", 250, "display.max_columns", 20,
                       "display.float_format", lambda v: f"{v:+.4f}"):
    print("\n-- 改善最多的 10 棟 --")
    print(df.sort_values("improve", ascending=False).head(10)[cols].to_string(index=False))
    print("\n-- K=725 仍然最差的 10 棟 --")
    print(df.sort_values(725).head(10)[cols].to_string(index=False))
    print("\n-- K=400 已經是負、K=725 更負（惡化）的建物 --")
    worse = df[(df[400] < 0) & (df["improve"] < 0)].sort_values("improve")
    print(worse.head(10)[cols].to_string(index=False))

print(f"\nK=400 為負的建物共 {int((df[400] < 0).sum())} 棟，"
      f"其中 K=725 轉正 {int(((df[400] < 0) & (df[725] > 0)).sum())} 棟、"
      f"仍為負 {int(((df[400] < 0) & (df[725] < 0)).sum())} 棟、"
      f"更惡化 {int(((df[400] < 0) & (df['improve'] < 0)).sum())} 棟")

print("\n=== 若移除 K=725 仍為負的前 N 棟，全域落差會變成多少 ===")
k725 = [r for r in records if r["budget"] == 725]
base = float(np.mean([r["ap_tab"] - r["ap_tree"] for r in k725]))
loo = {int(b): float(np.mean([r["loo_recovered"][b] for r in k725 if b in r["loo_recovered"]]))
       for b in k725[0]["loo_recovered"]}
ser = pd.Series(loo).sort_values(ascending=False)
print(f"K=725 平均總落差 = {base:+.4f}")
for n in (1, 3, 5, 10):
    print(f"  移除 LOO 回復量最大的 {n:>2} 棟後（近似）: {base + ser.head(n).sum():+.4f}")
