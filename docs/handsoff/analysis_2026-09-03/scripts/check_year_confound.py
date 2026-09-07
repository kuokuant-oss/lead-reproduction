#!/usr/bin/env python3
"""檢查 year_built / floor_count 的缺值結構，以及屋齡分箱是否與 site 混淆。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = Path(
    "/mnt/c/Users/User/AppData/Local/Temp/claude/C--Users-User-projects-lead-reproduction-temp"
    "/4888b221-2e78-4bf8-bfd6-7f21071cc05b/scratchpad"
)
META = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/building_metadata.csv")

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y, bid, sid = side["y"], side["bid"], side["sid"]
bmeta = pd.read_csv(META).set_index("building_id")

# 建物層級表（只看有異常的 65 棟）
b_tab = (pd.DataFrame({"b": bid, "site": sid, "y": y})
         .groupby("b").agg(site=("site", "first"), pos=("y", "sum"), rows=("y", "size")))
b_tab = b_tab[b_tab.pos > 0]
b_tab["year_built"] = bmeta["year_built"].reindex(b_tab.index)
b_tab["floor_count"] = bmeta["floor_count"].reindex(b_tab.index)
b_tab["sqft"] = bmeta["square_feet"].reindex(b_tab.index)
b_tab["use"] = bmeta["primary_use"].reindex(b_tab.index)

print(f"有異常的 hot water 建物: {len(b_tab)} 棟, 異常 {int(b_tab.pos.sum()):,}")
for col in ("year_built", "floor_count", "sqft"):
    miss = b_tab[col].isna()
    print(f"  {col:<12} 缺值 {int(miss.sum()):>2}/{len(b_tab)} 棟"
          f"（異常 {int(b_tab.pos[miss].sum()):>6,} = {b_tab.pos[miss].sum() / b_tab.pos.sum():.1%}）")

print("\n=== year_built 缺值 × site（建物數 / 異常數）===")
b_tab["year_missing"] = b_tab.year_built.isna()
ct = b_tab.groupby(["site", "year_missing"]).agg(n=("pos", "size"), pos=("pos", "sum"))
print(ct.to_string())

print("\n=== 各屋齡分箱的 site 組成（異常數占比）===")
BINS = [(1900, 1950), (1950, 1970), (1970, 1990), (1990, 2020)]
for lo, hi in BINS:
    sel = b_tab[(b_tab.year_built >= lo) & (b_tab.year_built < hi)]
    if sel.empty:
        continue
    comp = sel.groupby("site").pos.sum()
    comp = (comp / comp.sum()).sort_values(ascending=False)
    print(f"  {lo}–{hi - 1}  ({len(sel)} 棟, {int(sel.pos.sum()):,} 異常): "
          + ", ".join(f"site {s}: {v:.0%}" for s, v in comp.items()))

print("\n=== 有 year_built 的 31 棟：逐棟明細 ===")
have = b_tab[b_tab.year_built.notna()].sort_values("year_built")
with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.0f}"):
    print(have[["site", "use", "year_built", "sqft", "pos"]].to_string())

print("\n=== year_built 與 square_feet 的關係（有值的 31 棟）===")
print(f"  Spearman = {have[['year_built', 'sqft']].corr(method='spearman').iloc[0, 1]:.3f}")
