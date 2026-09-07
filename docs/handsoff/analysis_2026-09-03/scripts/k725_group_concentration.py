#!/usr/bin/env python3
"""K=725 hot water：site 與 primary use 的 PR-AUC 比較，並檢查是否被單一建物主導。"""
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
use_arr = bmeta["primary_use"].reindex(bid).to_numpy()

per_b = pd.read_json(SCRATCH / "k725_hotwater_per_building.json")


def concentration(mask, label):
    pos_b = bid[mask & (y == 1)]
    if len(pos_b) == 0:
        return None
    vc = pd.Series(pos_b).value_counts()
    # 移除最大貢獻建物後，該群其餘建物的落差方向是否一致
    others = per_b[per_b.building.isin(vc.index[1:]) & per_b.building.isin(np.unique(pos_b))]
    return {
        "group": label,
        "buildings_with_anomalies": int(len(vc)),
        "anomalies": int(len(pos_b)),
        "top_building": int(vc.index[0]),
        "top_share": float(vc.iloc[0] / len(pos_b)),
        "others_n": int(len(others)),
        "others_median_delta": float(others["delta"].median()) if len(others) else np.nan,
        "others_frac_negative": float((others["delta"] < 0).mean()) if len(others) else np.nan,
    }


rows = [concentration(sid == s, f"site {s}") for s in np.unique(sid)]
rows += [concentration(use_arr == u, u) for u in pd.unique(use_arr)]
df = pd.DataFrame([r for r in rows if r]).sort_values("anomalies", ascending=False)
with pd.option_context("display.width", 260, "display.max_columns", 20,
                       "display.float_format", lambda v: f"{v:.3f}"):
    print(df.to_string(index=False))
