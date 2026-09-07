#!/usr/bin/env python3
"""K=725 訓練池與 holdout 建物的交集（唯讀）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k")
MAN = REPO / "experiments/m5_building_count_v5_fixed_50k/audit/building_support_full_725.json"
CELL = (REPO / "data/processed/m5_building_curve/v5_fixed_50k/model_runs"
        / "building_seed725/row_seed0/tree_no_es_k725_f137")

man = json.loads(MAN.read_text())
print("manifest 頂層鍵:", list(man.keys()))
cand = man.get("candidate_buildings")
print("candidate_buildings 型別:", type(cand).__name__,
      "長度:" , len(cand) if hasattr(cand, "__len__") else "-")
if isinstance(cand, list) and cand:
    print("前 5 筆:", cand[:5])
cells = man.get("cells")
if isinstance(cells, list) and cells:
    print("cells[0] 鍵:", list(cells[0].keys()))
    print("cells 數:", len(cells))
elif isinstance(cells, dict):
    print("cells 鍵:", list(cells.keys())[:10])

pool = None
if isinstance(cand, list) and cand and isinstance(cand[0], (int, float)):
    pool = np.array(sorted({int(v) for v in cand}))
elif isinstance(cells, list):
    for c in cells:
        if int(c.get("building_budget", c.get("K", -1))) == 725:
            for key in ("building_ids", "buildings", "selected", "support"):
                if key in c and isinstance(c[key], list):
                    pool = np.array(sorted({int(v) for v in c[key]}))
                    break

if pool is None:
    print("\n無法從 manifest 直接取得 725 建物清單，改列出可用欄位供檢查。")
else:
    with np.load(CELL / "predictions.npz") as z:
        meter = np.asarray(z["meter"]).astype(np.int8)
        bid = np.asarray(z["building_id"]).astype(np.int32)
    hold = np.unique(bid)
    hw = np.unique(bid[meter == 3])
    print(f"\n訓練池建物數: {len(pool)}")
    print(f"holdout 建物數: {len(hold)}，其中在訓練池內: {len(np.intersect1d(hold, pool))}")
    print(f"hot water holdout 建物數: {len(hw)}，其中在訓練池內: "
          f"{len(np.intersect1d(hw, pool))}")
