#!/usr/bin/env python3
"""hot water 的月份覆蓋率檢查：列與異常在 site / primary use / 建物之間是否均勻分布。

問題：是否有地點缺少某些月份？異常是否在各類別間集中於特定月份？
唯讀。
"""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = Path(
    "/mnt/c/Users/User/AppData/Local/Temp/claude/C--Users-User-projects-lead-reproduction-temp"
    "/4888b221-2e78-4bf8-bfd6-7f21071cc05b/scratchpad"
)
META = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/building_metadata.csv")

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y = side["y"].astype(np.int8)
bid = side["bid"].astype(np.int32)
sid = side["sid"].astype(np.int32)
ts = pd.to_datetime(np.load(SCRATCH / "hotwater_time.npz")["ts"])
month = ts.month.to_numpy()
bmeta = pd.read_csv(META).set_index("building_id")
use = bmeta["primary_use"].reindex(bid).to_numpy()

MONTHS = list(range(1, 13))
LAB = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
HOURS = {m: calendar.monthrange(2016, m)[1] * 24 for m in MONTHS}
N_B = len(np.unique(bid))

df = pd.DataFrame({"b": bid, "site": sid, "use": use, "m": month, "y": y})

print("=" * 96)
print("A. 全體：每月實際列數 vs 完整覆蓋應有列數（73 棟 × 該月小時數）")
print("=" * 96)
print(f"{'月':<6}{'小時數':>8}{'應有列數':>11}{'實際列數':>11}{'缺漏':>9}{'缺漏率':>9}{'異常數':>9}")
tot_exp = tot_act = tot_pos = 0
for m, lab in zip(MONTHS, LAB):
    exp = HOURS[m] * N_B
    act = int((month == m).sum())
    pos = int(y[month == m].sum())
    tot_exp += exp
    tot_act += act
    tot_pos += pos
    print(f"{lab:<6}{HOURS[m]:>8}{exp:>11,}{act:>11,}{exp - act:>9,}"
          f"{(exp - act) / exp * 100:>8.2f}%{pos:>9,}")
print(f"{'合計':<6}{8784:>8}{tot_exp:>11,}{tot_act:>11,}{tot_exp - tot_act:>9,}"
      f"{(tot_exp - tot_act) / tot_exp * 100:>8.2f}%{tot_pos:>9,}")

print("\n" + "=" * 96)
print("B. 逐建物：有沒有整個月完全沒有資料")
print("=" * 96)
piv = df.pivot_table(index="b", columns="m", values="y", aggfunc="size", fill_value=0)
piv = piv.reindex(columns=MONTHS, fill_value=0)
missing_any = piv[(piv == 0).any(axis=1)]
print(f"有整月零列的建物: {len(missing_any)} / {N_B}")
if len(missing_any):
    for b, row in missing_any.iterrows():
        miss = [LAB[i] for i, m in enumerate(MONTHS) if row[m] == 0]
        print(f"  b{b} (site {int(df[df.b == b].site.iloc[0])}): 缺 {', '.join(miss)}")
short = piv.copy()
for m in MONTHS:
    short[m] = piv[m] / HOURS[m]
worst = short.min(axis=1).sort_values()
print(f"\n各建物「最不完整的那個月」覆蓋率，最低 8 棟：")
for b, v in worst.head(8).items():
    m = int(short.loc[b].idxmin())
    print(f"  b{b} (site {int(df[df.b == b].site.iloc[0])}): {LAB[m - 1]} 覆蓋 {v:.1%}"
          f"（{int(piv.loc[b, m]):,}/{HOURS[m]:,} 列）")
print(f"覆蓋率 100% 的建物-月組合: "
      f"{int((short >= 0.9999).sum().sum())} / {N_B * 12}")

print("\n" + "=" * 96)
print("C. site × 月：列數覆蓋率（實際 / 該 site 建物數 × 該月小時數）")
print("=" * 96)
sites = sorted(df.site.unique())
print(f"{'site':<8}{'建物':>5}" + "".join(f"{l:>8}" for l in LAB))
for s in sites:
    sub = df[df.site == s]
    nb = sub.b.nunique()
    line = f"site {s:<3}{nb:>5}"
    for m in MONTHS:
        act = int((sub.m == m).sum())
        line += f"{act / (nb * HOURS[m]) * 100:>7.1f}%"
    print(line)

print("\n" + "=" * 96)
print("D. site × 月：異常數，以及該 site 的異常在各月的分布（%）")
print("=" * 96)
for s in sites:
    sub = df[df.site == s]
    tot = int(sub.y.sum())
    if tot == 0:
        print(f"site {s}: 異常 0（僅提供負樣本）")
        continue
    counts = [int(sub[(sub.m == m)].y.sum()) for m in MONTHS]
    print(f"site {s:<3} 異常 {tot:>6,}  " +
          "  ".join(f"{l} {c / tot * 100:4.1f}%" for l, c in zip(LAB, counts)))

print("\n" + "=" * 96)
print("E. primary use × 月：該用途的異常在各月的分布（%）")
print("=" * 96)
for u in sorted(pd.unique(use)):
    sub = df[df.use == u]
    tot = int(sub.y.sum())
    if tot == 0:
        print(f"{u}: 異常 0")
        continue
    counts = [int(sub[(sub.m == m)].y.sum()) for m in MONTHS]
    print(f"{u[:26]:<27} 異常 {tot:>6,}  " +
          "  ".join(f"{c / tot * 100:4.1f}%" for c in counts))
print(f"{'（欄位順序）':<27}{'':>13}  " + "  ".join(f"{l:>4}" for l in LAB))

print("\n" + "=" * 96)
print("F. 均勻度：若完全均勻，各月應佔 8.33%")
print("=" * 96)
overall = np.array([int(df[df.m == m].y.sum()) for m in MONTHS], dtype=float)
overall_pct = overall / overall.sum() * 100
print("全體異常的月分布: " + "  ".join(f"{l} {p:4.1f}%" for l, p in zip(LAB, overall_pct)))
print(f"  最低 {overall_pct.min():.1f}%（{LAB[int(overall_pct.argmin())]}）"
      f"　最高 {overall_pct.max():.1f}%（{LAB[int(overall_pct.argmax())]}）")
