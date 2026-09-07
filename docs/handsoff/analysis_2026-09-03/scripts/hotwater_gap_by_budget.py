#!/usr/bin/env python3
"""唯讀：hot water (meter=3) 上 TabPFN 相對 Tree 的 PR-AUC 落差結構，跨 K=100/200/400/725。

對每個已完成的 cell 計算：
  - 全域 hot water PR-AUC（Tree / TabPFN）
  - 依 site / primary_use / meter_reading 分箱 的組內 PR-AUC 落差
  - 依 building 的組內落差，以及 leave-one-building-out 的落差回復量（集中度）
再在每個 K 內跨 cell 平均（先在 building seed 內平均 row seed，再跨 building seed）。

只讀 predictions.npz / cell.json / raw CSV；除了 scratchpad 快取外不寫任何檔案。
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PLOT = Path(
    "/mnt/c/Users/User/Documents/Codex/2026-09-01/wsl-ubuntu-m5-building-count-v5"
    "/outputs/scripts/plot_m5_v5_fixed50k_building_scarcity_roc.py"
)
ROOT = Path(
    "/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k/data/processed"
    "/m5_building_curve/v5_fixed_50k/model_runs"
)
TRAIN = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/train.csv")
META = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/building_metadata.csv")
SCRATCH = Path(
    "/mnt/c/Users/User/AppData/Local/Temp/claude/C--Users-User-projects-lead-reproduction-temp"
    "/4888b221-2e78-4bf8-bfd6-7f21071cc05b/scratchpad"
)
CACHE = SCRATCH / "hotwater_sideinfo.npz"
HOT_WATER = 3

spec = importlib.util.spec_from_file_location("m5plot", PLOT)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)

CELLS = [(k, b, r) for k in (100, 200, 400) for b in range(5) for r in (0, 1)]
CELLS += [(725, 725, 0), (725, 725, 1)]


def load_scores(budget: int, bseed: int, rseed: int):
    label = f"k{budget}_b{bseed}_r{rseed}"
    dirs = {m: m5.cell_dir(ROOT, bseed, rseed, budget, m) for m in ("ensemble", "tabpfn")}
    for d in dirs.values():
        if not m5.is_complete(d):
            return None
    meta = {m: m5.validate_cell(d, budget, bseed, rseed, m) for m, d in dirs.items()}
    m5.validate_pair(meta["ensemble"], meta["tabpfn"], label)
    arrays, scores = {}, {}
    for model, d in dirs.items():
        with np.load(d / "predictions.npz") as payload:
            arr = {k: np.asarray(payload[k]) for k in m5.HOLDOUT_ARRAY_KEYS}
            scores[model] = np.asarray(payload[model], dtype=np.float64)
        if arrays:
            for k in m5.HOLDOUT_ARRAY_KEYS:
                if not np.array_equal(arrays[k], arr[k]):
                    raise RuntimeError(f"holdout array {k} differs in {label}")
        else:
            arrays = arr
    keep = arrays["meter"].astype(np.int8, copy=False) == HOT_WATER
    return (arrays, keep, scores["ensemble"][keep], scores["tabpfn"][keep])


# ---------- side info（holdout 固定，只建一次）----------
if CACHE.is_file():
    with np.load(CACHE, allow_pickle=True) as c:
        y = c["y"]; bid = c["bid"]; sid = c["sid"]; reading = c["reading"]; hour = c["hour"]
        use = c["use"].astype(str)
    print(f"[cache] side info 由 {CACHE.name} 載入")
else:
    first = load_scores(*CELLS[0])
    arrays, keep = first[0], first[1]
    y = arrays["anomaly"].astype(np.int8)[keep]
    bid = arrays["building_id"][keep].astype(np.int32)
    sid = arrays["site_id"][keep].astype(np.int32)
    raw_idx = arrays["validation_raw_index"][keep].astype(np.int64)
    del first, arrays

    need = np.zeros(raw_idx.max() + 1, dtype=bool)
    need[raw_idx] = True
    readings = np.full(need.size, np.nan)
    hours = np.full(need.size, -1, dtype=np.int8)
    bids_raw = np.full(need.size, -1, dtype=np.int32)
    meters_raw = np.full(need.size, -1, dtype=np.int8)
    start = 0
    for chunk in pd.read_csv(TRAIN, usecols=["building_id", "meter", "timestamp", "meter_reading"],
                             chunksize=2_000_000):
        stop = start + len(chunk)
        sel = need[start:stop] if stop <= need.size else need[start:]
        if sel.size and sel.any():
            idx = np.nonzero(sel)[0]
            sub = chunk.iloc[idx]
            pos = start + idx
            readings[pos] = sub["meter_reading"].to_numpy(dtype=np.float64)
            hours[pos] = pd.to_datetime(sub["timestamp"]).dt.hour.to_numpy(dtype=np.int8)
            bids_raw[pos] = sub["building_id"].to_numpy(dtype=np.int32)
            meters_raw[pos] = sub["meter"].to_numpy(dtype=np.int8)
        start = stop
        if start > raw_idx.max():
            break
    reading = readings[raw_idx]
    hour = hours[raw_idx]
    if not np.array_equal(bids_raw[raw_idx], bid) or not (meters_raw[raw_idx] == HOT_WATER).all():
        raise RuntimeError("validation_raw_index -> train.csv 對不上")
    bmeta = pd.read_csv(META).set_index("building_id")
    use = bmeta["primary_use"].reindex(bid).fillna("Unknown").to_numpy().astype(str)
    np.savez_compressed(CACHE, y=y, bid=bid, sid=sid, reading=reading, hour=hour, use=use)
    print(f"[cache] side info 建立完成 -> {CACHE.name}")
    del readings, hours, bids_raw, meters_raw, need, raw_idx

n_rows, n_pos = len(y), int(y.sum())
print(f"hot water holdout: rows={n_rows:,} positives={n_pos:,} ({n_pos / n_rows:.2%}) "
      f"buildings={len(np.unique(bid))} sites={len(np.unique(sid))}")
zero = reading == 0
print(f"reading=0 佔 {zero.mean():.2%}；全部正樣本中有 {y[zero].sum() / n_pos:.2%} 落在 reading=0")

# reading 分箱（固定，跨 cell 一致）
bins = np.full(n_rows, "zero", dtype=object)
posr = reading > 0
q = pd.qcut(reading[posr], 10, labels=False, duplicates="drop")
bins[posr] = [f"d{int(v)}" for v in q]
BIN_ORDER = ["zero"] + [f"d{i}" for i in range(10)]

GROUPS = {"site": sid, "use": use, "reading_bin": bins, "building": bid}
group_index = {
    name: {v: np.nonzero(values == v)[0] for v in pd.unique(pd.Series(values))}
    for name, values in GROUPS.items()
}


def ap(mask_idx, s):
    yy = y[mask_idx]
    p = int(yy.sum())
    if p == 0 or p == len(yy):
        return float("nan")
    return float(average_precision_score(yy, s[mask_idx]))


records = []
for budget, bseed, rseed in CELLS:
    got = load_scores(budget, bseed, rseed)
    if got is None:
        print(f"skip k{budget}_b{bseed}_r{rseed}: 尚未完成")
        continue
    _, _, s_tree, s_tab = got
    rec = {
        "budget": budget, "building_seed": bseed, "row_seed": rseed,
        "ap_tree": float(average_precision_score(y, s_tree)),
        "ap_tab": float(average_precision_score(y, s_tab)),
        "roc_tree": float(roc_auc_score(y, s_tree)),
        "roc_tab": float(roc_auc_score(y, s_tab)),
        "groups": {},
    }
    for name, index in group_index.items():
        rec["groups"][name] = {
            str(v): {"ap_tree": ap(idx, s_tree), "ap_tab": ap(idx, s_tab), "n": int(len(idx)),
                     "pos": int(y[idx].sum())}
            for v, idx in index.items()
        }
    # 集中度：leave-one-building-out
    base_gap = rec["ap_tab"] - rec["ap_tree"]
    loo = {}
    for b, idx in group_index["building"].items():
        mask = np.ones(n_rows, dtype=bool)
        mask[idx] = False
        yy = y[mask]
        if not 0 < int(yy.sum()) < len(yy):
            continue
        g = float(average_precision_score(yy, s_tab[mask]) - average_precision_score(yy, s_tree[mask]))
        loo[str(int(b))] = g - base_gap
    rec["loo_recovered"] = loo
    records.append(rec)
    print(f"  k{budget}_b{bseed}_r{rseed}: hot water PR-AUC tree={rec['ap_tree']:.4f} "
          f"tab={rec['ap_tab']:.4f} delta={base_gap:+.4f}")
    del s_tree, s_tab, got

(SCRATCH / "hotwater_gap_by_budget.json").write_text(
    json.dumps(records, indent=2, allow_nan=True), encoding="utf-8")
print(f"\n[saved] {SCRATCH / 'hotwater_gap_by_budget.json'}")
