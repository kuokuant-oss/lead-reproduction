#!/usr/bin/env python3
"""hot water 月份覆蓋率：所有類別維度。

建物層級維度（site / primary use / square feet / year built / floor count）：
  每月列覆蓋率 = 實際列 / (該組建物數 × 該月小時數)，以及該組異常的月分布。
列層級維度（meter reading 分箱）：每月的列占比與異常月分布。
另加：逐月的零讀數率與 P(anomaly | reading=0)。
唯讀；輸出 markdown 檔。
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
OUTDIR = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff/analysis_2026-09-03")
META = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/building_metadata.csv")

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y = side["y"].astype(np.int8)
bid = side["bid"].astype(np.int32)
sid = side["sid"].astype(np.int32)
reading = side["reading"]
ts = pd.to_datetime(np.load(SCRATCH / "hotwater_time.npz")["ts"])
month = ts.month.to_numpy()

bmeta = pd.read_csv(META).set_index("building_id")
use = bmeta["primary_use"].reindex(bid).to_numpy()
sqft = bmeta["square_feet"].reindex(bid).to_numpy(dtype=float)
year = bmeta["year_built"].reindex(bid).to_numpy(dtype=float)
floors = bmeta["floor_count"].reindex(bid).to_numpy(dtype=float)

MONTHS = list(range(1, 13))
LAB = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
HOURS = {m: calendar.monthrange(2016, m)[1] * 24 for m in MONTHS}
N = len(y)


def qbins(values, n, unit):
    uniq = pd.DataFrame({"b": bid, "v": values}).drop_duplicates("b")
    edges = np.percentile(uniq[np.isfinite(uniq["v"])]["v"], np.linspace(0, 100, n + 1))
    out = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        m = ((values >= lo) & (values <= hi)) if i == n - 1 else ((values >= lo) & (values < hi))
        out.append((f"{lo:,.0f}–{hi:,.0f}{unit}", m))
    if (~np.isfinite(values)).any():
        out.append(("缺值", ~np.isfinite(values)))
    return out


READING_EDGES = [0.0, 1.0, 10.0, 100.0, 1e3, 1e4, np.inf]
READING_KEYS = ["= 0", "> 0 – 1", "1 – 10", "10 – 100", "100 – 1 000",
                "1 000 – 10 000", "> 10 000"]
rbin = np.full(N, READING_KEYS[0], dtype=object)
for i in range(len(READING_EDGES) - 1):
    rbin[(reading > READING_EDGES[i]) & (reading <= READING_EDGES[i + 1])] = READING_KEYS[i + 1]

BUILDING_DIMS = {
    "Site": [(f"Site {s}", sid == s) for s in np.unique(sid)],
    "Primary use": [(str(u), use == u) for u in sorted(pd.unique(use))],
    "Square feet 四分位": qbins(sqft, 4, " sq ft"),
    "Year built": [("1900–1949", (year >= 1900) & (year < 1950)),
                   ("1950–1969", (year >= 1950) & (year < 1970)),
                   ("1970–1989", (year >= 1970) & (year < 1990)),
                   ("1990–2019", (year >= 1990) & (year < 2020)),
                   ("缺值", ~np.isfinite(year))],
    "Floor count": qbins(floors, 3, " 層"),
}
ROW_DIMS = {"Meter reading 分箱": [(k, rbin == k) for k in READING_KEYS]}

lines = ["# Hot water 月份覆蓋率檢查（2016，逐小時）", "",
         "建物層級維度：每月列覆蓋率 = 實際列 ÷（該組建物數 × 該月小時數）。",
         "列層級維度：每月列數占該組的百分比。",
         "每個維度都涵蓋全部 73 棟 / 636,121 列 / 90,691 正樣本，缺值自成類別。", ""]

print("=" * 100)
print("整體：每月列覆蓋率與異常占比")
print("=" * 100)
hdr = f"{'月':<6}{'應有列':>10}{'實際列':>10}{'覆蓋率':>9}{'異常數':>9}{'異常占比':>9}{'零讀數率':>10}{'P(異常|零)':>11}"
print(hdr)
tot_pos = int(y.sum())
for m, lab in zip(MONTHS, LAB):
    sel = month == m
    exp = HOURS[m] * 73
    act = int(sel.sum())
    pos = int(y[sel].sum())
    z = reading[sel] == 0
    print(f"{lab:<6}{exp:>10,}{act:>10,}{act / exp * 100:>8.2f}%{pos:>9,}"
          f"{pos / tot_pos * 100:>8.1f}%{z.mean() * 100:>9.1f}%"
          f"{(y[sel][z] == 1).mean() * 100:>10.1f}%")

lines += ["## 整體", "",
          "| 月 | 應有列 | 實際列 | 覆蓋率 | 異常數 | 異常占比 | 零讀數率 | P(異常\\|零讀數) |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for m, lab in zip(MONTHS, LAB):
    sel = month == m
    exp, act = HOURS[m] * 73, int(sel.sum())
    pos = int(y[sel].sum())
    z = reading[sel] == 0
    lines.append(f"| {lab} | {exp:,} | {act:,} | {act / exp * 100:.2f}% | {pos:,} | "
                 f"{pos / tot_pos * 100:.1f}% | {z.mean() * 100:.1f}% | "
                 f"{(y[sel][z] == 1).mean() * 100:.1f}% |")
lines.append("")

for dim, entries in BUILDING_DIMS.items():
    print("\n" + "=" * 100)
    print(f"{dim} —— 每月列覆蓋率 %（100% = 該組建物在該月的完整小時數）")
    print("=" * 100)
    print(f"{'組別':<24}{'建物':>5}" + "".join(f"{l:>7}" for l in LAB))
    lines += [f"## {dim}", "", "### 每月列覆蓋率 %", "",
              "| 組別 | 建物 | " + " | ".join(LAB) + " |",
              "| --- | ---: | " + " | ".join(["---:"] * 12) + " |"]
    for label, mask in entries:
        nb = len(np.unique(bid[mask]))
        cov = []
        for m in MONTHS:
            act = int((mask & (month == m)).sum())
            cov.append(act / (nb * HOURS[m]) * 100)
        print(f"{label[:23]:<24}{nb:>5}" + "".join(f"{c:>6.1f}%" for c in cov))
        lines.append(f"| {label} | {nb} | " + " | ".join(f"{c:.1f}%" for c in cov) + " |")
    lines.append("")

    print(f"\n{dim} —— 該組異常在各月的分布 %")
    print(f"{'組別':<24}{'異常':>8}" + "".join(f"{l:>7}" for l in LAB))
    lines += ["### 該組異常在各月的分布 %", "",
              "| 組別 | 異常 | " + " | ".join(LAB) + " |",
              "| --- | ---: | " + " | ".join(["---:"] * 12) + " |"]
    for label, mask in entries:
        tot = int(y[mask].sum())
        if tot == 0:
            print(f"{label[:23]:<24}{0:>8}   （無異常，僅負樣本）")
            lines.append(f"| {label} | 0 | " + " | ".join(["—"] * 12) + " |")
            continue
        dist = [int(y[mask & (month == m)].sum()) / tot * 100 for m in MONTHS]
        print(f"{label[:23]:<24}{tot:>8,}" + "".join(f"{d:>6.1f}%" for d in dist))
        lines.append(f"| {label} | {tot:,} | " + " | ".join(f"{d:.1f}%" for d in dist) + " |")
    lines.append("")

for dim, entries in ROW_DIMS.items():
    print("\n" + "=" * 100)
    print(f"{dim} —— 該組列在各月的分布 % / 該組異常在各月的分布 %")
    print("=" * 100)
    print(f"{'組別':<18}{'列':>10}{'異常':>9}" + "".join(f"{l:>7}" for l in LAB))
    lines += [f"## {dim}", "", "### 該組異常在各月的分布 %", "",
              "| 組別 | 列 | 異常 | " + " | ".join(LAB) + " |",
              "| --- | ---: | ---: | " + " | ".join(["---:"] * 12) + " |"]
    for label, mask in entries:
        tot = int(y[mask].sum())
        n_rows = int(mask.sum())
        if tot == 0:
            print(f"{label:<18}{n_rows:>10,}{0:>9}   （無異常）")
            lines.append(f"| {label} | {n_rows:,} | 0 | " + " | ".join(["—"] * 12) + " |")
            continue
        dist = [int(y[mask & (month == m)].sum()) / tot * 100 for m in MONTHS]
        print(f"{label:<18}{n_rows:>10,}{tot:>9,}" + "".join(f"{d:>6.1f}%" for d in dist))
        lines.append(f"| {label} | {n_rows:,} | {tot:,} | "
                     + " | ".join(f"{d:.1f}%" for d in dist) + " |")
    lines.append("")

OUTDIR.mkdir(parents=True, exist_ok=True)
out = OUTDIR / "hot_water_month_coverage.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"\n[saved] {out}")
