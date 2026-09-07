#!/usr/bin/env python3
"""候選案例的異常型態診斷：異常是否以零讀數表現，以及零讀數是否代表異常。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = Path(
    "/mnt/c/Users/User/AppData/Local/Temp/claude/C--Users-User-projects-lead-reproduction-temp"
    "/4888b221-2e78-4bf8-bfd6-7f21071cc05b/scratchpad"
)
META = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/building_metadata.csv")
side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y, bid, sid, reading = side["y"], side["bid"], side["sid"], side["reading"]
per_b = pd.read_json(SCRATCH / "k725_hotwater_per_building.json")
bmeta = pd.read_csv(META).set_index("building_id")


def diag(mask, label):
    yy = y[mask] == 1
    z = reading[mask] == 0
    n = int(mask.sum())
    return {
        "label": label, "rows": n, "pos": int(yy.sum()), "prev": float(yy.mean()),
        "zero_rate": float(z.mean()),
        "P(zero|anom)": float(z[yy].mean()) if yy.any() else np.nan,
        "P(anom|zero)": float(yy[z].mean()) if z.any() else np.nan,
        "n_buildings": int(len(np.unique(bid[mask]))),
    }


CAND_B = [191, 253, 1237, 217, 1017, 1331, 213, 119, 113, 1297, 1311, 1241, 1011, 145]
rows = [diag(bid == b, f"building {b}") for b in CAND_B]
df = pd.DataFrame(rows)
df["ap_tree"] = [per_b.loc[per_b.building == b, "ap_tree"].iloc[0] for b in CAND_B]
df["ap_tab"] = [per_b.loc[per_b.building == b, "ap_tab"].iloc[0] for b in CAND_B]
df["delta"] = [per_b.loc[per_b.building == b, "delta"].iloc[0] for b in CAND_B]
df["delta_se"] = [per_b.loc[per_b.building == b, "delta_se"].iloc[0] for b in CAND_B]
df["use"] = [bmeta.loc[b, "primary_use"] for b in CAND_B]
df["site"] = [bmeta.loc[b, "site_id"] for b in CAND_B]
with pd.option_context("display.width", 300, "display.max_columns", 30,
                       "display.float_format", lambda v: f"{v:.4f}"):
    print("=== 候選建物 ===")
    print(df.sort_values("delta")[
        ["label", "site", "use", "pos", "prev", "zero_rate", "P(zero|anom)",
         "P(anom|zero)", "ap_tree", "ap_tab", "delta", "delta_se"]].to_string(index=False))

print("\n=== 候選群組 ===")
groups = [(sid == 1, "site 1"), (sid == 10, "site 10"), (sid == 2, "site 2"),
          (sid == 14, "site 14"), (sid == 15, "site 15")]
use_arr = bmeta["primary_use"].reindex(bid).to_numpy()
groups += [(use_arr == "Healthcare", "Healthcare"),
           (use_arr == "Technology/science", "Technology/science"),
           (use_arr == "Lodging/residential", "Lodging/residential")]
g = pd.DataFrame([diag(m, lab) for m, lab in groups])
with pd.option_context("display.width", 300, "display.float_format", lambda v: f"{v:.4f}"):
    print(g.to_string(index=False))

print("\n=== 群組是否由單一建物主導 ===")
for m, lab in groups:
    sub = bid[m & (y == 1)]
    if len(sub) == 0:
        continue
    vc = pd.Series(sub).value_counts()
    top = vc.index[0]
    print(f"  {lab:<22} 異常 {len(sub):>6,} · 最大建物 b{top} 佔 {vc.iloc[0] / len(sub):.1%}"
          f" · 建物數 {len(vc)}")
