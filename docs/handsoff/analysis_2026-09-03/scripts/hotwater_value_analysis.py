#!/usr/bin/env python3
"""唯讀：hot water 上 TabPFN vs Tree 的誤差 —— 數值(meter_reading)與時間維度。

用法: python hotwater_value_analysis.py <budget> <building_seed> <row_seed>
以 validation_raw_index 回連 raw train.csv（分塊讀取，低記憶體），不寫入任何檔案。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

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
HOT_WATER = 3

spec = importlib.util.spec_from_file_location("m5plot", PLOT)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)

BUDGET, BSEED, RSEED = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
LABEL = f"k{BUDGET}_b{BSEED}_r{RSEED}"

dirs = {m: m5.cell_dir(ROOT, BSEED, RSEED, BUDGET, m) for m in ("ensemble", "tabpfn")}
meta = {m: m5.validate_cell(d, BUDGET, BSEED, RSEED, m) for m, d in dirs.items()}
m5.validate_pair(meta["ensemble"], meta["tabpfn"], LABEL)
print(f"[gate] {LABEL}: cell identity OK, pair identity OK")

arrays, scores = {}, {}
for model, d in dirs.items():
    with np.load(d / "predictions.npz") as payload:
        arr = {k: np.asarray(payload[k]) for k in m5.HOLDOUT_ARRAY_KEYS}
        scores[model] = np.asarray(payload[model], dtype=np.float64)
    if arrays:
        for k in m5.HOLDOUT_ARRAY_KEYS:
            if not np.array_equal(arrays[k], arr[k]):
                raise RuntimeError(f"holdout array {k} differs")
    else:
        arrays = arr

keep = arrays["meter"].astype(np.int8, copy=False) == HOT_WATER
y = arrays["anomaly"].astype(np.int8, copy=False)[keep]
bid = arrays["building_id"][keep]
raw_idx = arrays["validation_raw_index"][keep].astype(np.int64)
s_tree = scores["ensemble"][keep]
s_tab = scores["tabpfn"][keep]
del scores, arrays

# ---- 分塊回連 raw train.csv 取得 meter_reading / timestamp ----
need = np.zeros(raw_idx.max() + 1, dtype=bool)
need[raw_idx] = True
readings = np.full(raw_idx.max() + 1, np.nan, dtype=np.float64)
hours = np.full(raw_idx.max() + 1, -1, dtype=np.int8)
bids_raw = np.full(raw_idx.max() + 1, -1, dtype=np.int32)
meters_raw = np.full(raw_idx.max() + 1, -1, dtype=np.int8)

start = 0
for chunk in pd.read_csv(TRAIN, usecols=["building_id", "meter", "timestamp", "meter_reading"],
                         chunksize=2_000_000):
    stop = start + len(chunk)
    sel = need[start:stop]
    if sel.any():
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
if not np.array_equal(bids_raw[raw_idx].astype(np.int64), bid.astype(np.int64)):
    raise RuntimeError("validation_raw_index -> train.csv building_id 對不上")
if not (meters_raw[raw_idx] == HOT_WATER).all():
    raise RuntimeError("validation_raw_index -> train.csv meter 對不上")
print("[gate] raw_index 回連 train.csv：building_id 與 meter 逐列相符")
del readings, hours, bids_raw, meters_raw, need

df = pd.DataFrame({"y": y, "bid": bid, "reading": reading, "hour": hour,
                   "tree": s_tree, "tab": s_tab})
print(f"\nhot water rows={len(df):,}  positives={int(df.y.sum()):,}  "
      f"reading NaN={int(df.reading.isna().sum()):,}")

print("\n=== meter_reading == 0 的行為 ===")
zero = df.reading == 0
print(f"reading=0 佔 {zero.mean():.4%}，其中 anomaly 率 {df.y[zero].mean():.4%}；"
      f"reading>0 的 anomaly 率 {df.y[~zero].mean():.4%}")
for name, col in (("Tree", "tree"), ("TabPFN", "tab")):
    print(f"{name}: reading=0 平均分數 {df[col][zero].mean():.4f} / "
          f"reading>0 平均分數 {df[col][~zero].mean():.4f}")

print("\n=== 依 meter_reading 十分位（>0 的部分另外分箱）===")
pos_reading = df.reading > 0
df["bin"] = "zero"
qs = pd.qcut(df.loc[pos_reading, "reading"], 10, labels=False, duplicates="drop")
df.loc[pos_reading, "bin"] = [f"d{int(q)}" for q in qs]
rows = []
for value, sub in df.groupby("bin", observed=True):
    pos = int(sub["y"].sum())
    a_t = average_precision_score(sub["y"], sub["tree"]) if 0 < pos < len(sub) else np.nan
    a_p = average_precision_score(sub["y"], sub["tab"]) if 0 < pos < len(sub) else np.nan
    rows.append({"bin": value, "reading_min": sub["reading"].min(), "reading_max": sub["reading"].max(),
                 "rows": len(sub), "pos": pos, "prev": pos / len(sub),
                 "mean_tree": sub["tree"].mean(), "mean_tab": sub["tab"].mean(),
                 "ap_tree": a_t, "ap_tab": a_p, "delta": a_p - a_t})
out = pd.DataFrame(rows)
order = ["zero"] + [f"d{i}" for i in range(10)]
out["__o"] = out["bin"].map({k: i for i, k in enumerate(order)})
out = out.sort_values("__o").drop(columns="__o")
with pd.option_context("display.width", 240, "display.max_columns", 20,
                       "display.float_format", lambda v: f"{v:.4f}"):
    print(out.to_string(index=False))

print("\n=== 高分區（各自 top-n_pos）誤報的 reading 分布 ===")
n_pos = int(df.y.sum())
for name, col in (("Tree", "tree"), ("TabPFN", "tab")):
    order_idx = np.argsort(-df[col].to_numpy(), kind="stable")[:n_pos]
    sub = df.iloc[order_idx]
    fp = sub[sub.y == 0]
    print(f"{name}: 誤報 {len(fp):,} 筆  reading=0 佔 {float((fp.reading == 0).mean()):.2%}  "
          f"reading 中位數 {fp.reading.median():.3f}  平均 {fp.reading.mean():.3f}")
    miss = df[(df.y == 1)].drop(index=sub.index, errors="ignore")
    print(f"    漏抓正樣本 {len(miss):,} 筆  reading=0 佔 {float((miss.reading == 0).mean()):.2%}  "
          f"reading 中位數 {miss.reading.median():.3f}")

print("\n=== 依小時 ===")
rows = []
for value, sub in df.groupby("hour", observed=True):
    pos = int(sub["y"].sum())
    a_t = average_precision_score(sub["y"], sub["tree"]) if 0 < pos < len(sub) else np.nan
    a_p = average_precision_score(sub["y"], sub["tab"]) if 0 < pos < len(sub) else np.nan
    rows.append({"hour": int(value), "rows": len(sub), "prev": pos / len(sub),
                 "ap_tree": a_t, "ap_tab": a_p, "delta": a_p - a_t})
with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.4f}"):
    print(pd.DataFrame(rows).to_string(index=False))
