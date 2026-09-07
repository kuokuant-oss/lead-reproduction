#!/usr/bin/env python3
"""釐清 725 與 holdout 建物數的關係（唯讀）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k")
FORMAL = REPO / "data/processed/m5_building_curve/v5_fixed_50k/model_runs"
CELL = FORMAL / "building_seed725/row_seed0/tree_no_es_k725_f137"

with np.load(CELL / "predictions.npz") as z:
    meter = np.asarray(z["meter"]).astype(np.int8)
    bid = np.asarray(z["building_id"]).astype(np.int32)
    sid = np.asarray(z["site_id"]).astype(np.int32)
    anom = np.asarray(z["anomaly"]).astype(np.int8)

print("=== holdout（評估集，所有 K 共用）===")
print(f"  總列數      {len(bid):,}")
print(f"  建物數      {len(np.unique(bid))}")
print(f"  site 數     {len(np.unique(sid))}")
names = {0: "electricity(排除)", 1: "chilled water", 2: "steam", 3: "hot water"}
for m in sorted(np.unique(meter).tolist()):
    sel = meter == m
    b = np.unique(bid[sel])
    b_anom = np.unique(bid[sel & (anom == 1)])
    print(f"  meter {m} {names.get(m, ''):<18} 列 {int(sel.sum()):>9,}  "
          f"建物 {len(b):>3}  其中有異常 {len(b_anom):>3}  異常 {int(anom[sel].sum()):>7,}")

evaluated = np.unique(bid[meter != 0])
print(f"\n  正式評估（meter 1/2/3）涵蓋建物 {len(evaluated)} 棟")

manifest = json.loads((REPO / "experiments/m5_building_count_v5_fixed_50k/audit/contexts"
                       / "k725_full_r0.json").read_text()) \
    if (REPO / "experiments/m5_building_count_v5_fixed_50k/audit/contexts/k725_full_r0.json").is_file() \
    else None
print("\n=== K=725 context（訓練來源池）===")
cell = json.loads((CELL / "cell.json").read_text())
for k in ("building_budget", "available_buildings", "context_rows",
          "holdout_rows", "source_building_manifest"):
    print(f"  {k}: {cell.get(k)}")

src = cell.get("source_building_manifest")
if src and Path(src).is_file():
    man = json.loads(Path(src).read_text())
    ids = None
    for key in ("building_ids", "buildings", "selected_buildings", "ladder"):
        if key in man:
            ids = man[key]
            break
    if isinstance(ids, dict):
        ids = ids.get("725") or ids.get(725)
    if isinstance(ids, list) and ids and not isinstance(ids[0], (int, float)):
        ids = None
    if ids:
        pool = np.array(sorted(set(int(i) for i in ids)))
        print(f"  manifest 建物數: {len(pool)}")
        hw = np.unique(bid[meter == 3])
        print(f"\n=== 交集 ===")
        print(f"  hot water holdout 建物 {len(hw)} 棟")
        print(f"  其中在 725 訓練池內: {len(np.intersect1d(hw, pool))} 棟")
        print(f"  正式評估建物 {len(evaluated)} 棟，其中在訓練池內: "
              f"{len(np.intersect1d(evaluated, pool))} 棟")
    else:
        print(f"  manifest keys: {list(man.keys())[:12]}")
