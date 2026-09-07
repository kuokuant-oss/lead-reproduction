#!/usr/bin/env python3
"""hot water 的時間分層：月份 / 時段 / 星期，跨 K=100/200/400/725。

先由 validation_raw_index 回連 raw train.csv 取得 timestamp（分塊讀取），
再用與其他維度相同的完整口徑製表（涵蓋全部 73 棟 / 636,121 列 / 90,691 正樣本）。
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PLOT = Path(
    "/mnt/c/Users/User/Documents/Codex/2026-09-01/wsl-ubuntu-m5-building-count-v5"
    "/outputs/scripts/plot_m5_v5_fixed50k_building_scarcity_roc.py"
)
REPO = Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k")
SCRATCH = Path(
    "/mnt/c/Users/User/AppData/Local/Temp/claude/C--Users-User-projects-lead-reproduction-temp"
    "/4888b221-2e78-4bf8-bfd6-7f21071cc05b/scratchpad"
)
TRAIN = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/train.csv")
HOT_WATER = 3
BUDGETS = [100, 200, 400, 725]

spec = importlib.util.spec_from_file_location("m5plot", PLOT)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)
formal_root = REPO / "data/processed/m5_building_curve/v5_fixed_50k/model_runs"
ext_root = REPO / "data/processed/m5_building_curve/v5_fixed_50k_k725_row_seed_extension/model_runs"
colab = REPO / "data/processed/m5_building_curve/v5_fixed_50k_k725_colab"
_, k725_tree_dirs = m5.validate_k725(REPO, formal_root, ext_root,
                                     colab / "model_results/k725_five_seed_summary.json")

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y = side["y"].astype(np.int8)
bid = side["bid"].astype(np.int32)
N, P = len(y), int(y.sum())

# ---- 取得 timestamp ----
cache = SCRATCH / "hotwater_time.npz"
if cache.is_file():
    tc = np.load(cache)
    ts = pd.to_datetime(tc["ts"])
    print("[cache] 時間資訊由快取載入")
else:
    ref = m5.cell_dir(formal_root, 725, 0, 725, "ensemble")
    with np.load(ref / "predictions.npz") as z:
        keep = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        raw_idx = np.asarray(z["validation_raw_index"])[keep].astype(np.int64)
        if not np.array_equal(np.asarray(z["building_id"])[keep].astype(np.int32), bid):
            raise RuntimeError("holdout 不符")
    need = np.zeros(raw_idx.max() + 1, dtype=bool)
    need[raw_idx] = True
    stamps = np.empty(need.size, dtype="datetime64[s]")
    start = 0
    for chunk in pd.read_csv(TRAIN, usecols=["timestamp"], chunksize=2_000_000):
        stop = start + len(chunk)
        sel = need[start:stop]
        if sel.any():
            idx = np.nonzero(sel)[0]
            stamps[start + idx] = pd.to_datetime(
                chunk.iloc[idx]["timestamp"]).to_numpy(dtype="datetime64[s]")
        start = stop
        if start > raw_idx.max():
            break
    ts = pd.to_datetime(stamps[raw_idx])
    np.savez_compressed(cache, ts=ts.to_numpy())
    print("[cache] 時間資訊建立完成")

print(f"holdout 時間範圍: {ts.min()} — {ts.max()}")
print(f"不重複日期數: {ts.normalize().nunique()}，不重複小時: {ts.hour.nunique()}")

month = ts.month.to_numpy()
hour = ts.hour.to_numpy()
dow = ts.dayofweek.to_numpy()

MONTH_NAME = {1: "1月", 2: "2月", 3: "3月", 4: "4月", 5: "5月", 6: "6月",
              7: "7月", 8: "8月", 9: "9月", 10: "10月", 11: "11月", 12: "12月"}
DIMENSIONS = {
    "月份": [(MONTH_NAME[m], month == m) for m in range(1, 13)],
    "時段": [("00–05 深夜", (hour >= 0) & (hour < 6)),
             ("06–11 上午", (hour >= 6) & (hour < 12)),
             ("12–17 下午", (hour >= 12) & (hour < 18)),
             ("18–23 晚間", (hour >= 18) & (hour < 24))],
    "星期": [("平日 (一–五)", dow < 5), ("週末 (六日)", dow >= 5)],
    "季節": [("冬 12–2月", np.isin(month, [12, 1, 2])),
             ("春 3–5月", np.isin(month, [3, 4, 5])),
             ("夏 6–8月", np.isin(month, [6, 7, 8])),
             ("秋 9–11月", np.isin(month, [9, 10, 11]))],
}
for entries in DIMENSIONS.values():
    entries.append(("合計", np.ones(N, dtype=bool)))


def cells_for(budget):
    return [(725, r) for r in range(5)] if budget == 725 else \
           [(b, r) for b in range(5) for r in (0, 1)]


def load_scores(budget, bseed, rseed):
    if budget == 725:
        tdir = k725_tree_dirs[rseed]
        tab = (m5.cell_dir(formal_root, 725, rseed, 725, "tabpfn") / "predictions.npz"
               if rseed < 2 else colab / f"model_results/row_seed{rseed}/predictions.npz")
    else:
        tdir = m5.cell_dir(formal_root, bseed, rseed, budget, "ensemble")
        tab = m5.cell_dir(formal_root, bseed, rseed, budget, "tabpfn") / "predictions.npz"
        m5.validate_pair(m5.validate_cell(tdir, budget, bseed, rseed, "ensemble"),
                         m5.validate_cell(tab.parent, budget, bseed, rseed, "tabpfn"),
                         f"k{budget}_b{bseed}_r{rseed}")
    with np.load(tdir / "predictions.npz") as z:
        keep = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        st = np.asarray(z["ensemble"], dtype=np.float64)[keep]
    with np.load(tab) as z:
        k2 = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        sp = np.asarray(z["tabpfn"], dtype=np.float64)[k2]
    return st, sp


raw: dict = {}
for budget in BUDGETS:
    for bseed, rseed in cells_for(budget):
        st, sp = load_scores(budget, bseed, rseed)
        for dim, entries in DIMENSIONS.items():
            for label, mask in entries:
                yy = y[mask]
                p = int(yy.sum())
                if p == 0 or p == len(yy):
                    continue
                raw.setdefault((budget, dim, label), []).append(
                    ((bseed, rseed),
                     float(average_precision_score(yy, st[mask])),
                     float(average_precision_score(yy, sp[mask]))))
        del st, sp
    print(f"  K={budget} 完成", flush=True)


def agg(entries, budget):
    if budget == 725:
        tr = np.array([e[1] for e in entries])
        tb = np.array([e[2] for e in entries])
    else:
        by = {}
        for (b, _), t, p in entries:
            by.setdefault(b, []).append((t, p))
        tr = np.array([np.mean([x[0] for x in v]) for v in by.values()])
        tb = np.array([np.mean([x[1] for x in v]) for v in by.values()])
    d = tb - tr
    return (float(tr.mean()), float(tb.mean()), float(d.mean()),
            float(d.std(ddof=1) / math.sqrt(len(d))),
            int((np.sign(d) == np.sign(d.mean())).sum()), len(d))


for dim, entries in DIMENSIONS.items():
    print(f"\n{'=' * 108}\n{dim}")
    print(f"{'組別':<14}{'列數':>9}{'佔全部列':>10}{'正樣本':>9}{'負樣本':>10}"
          + "".join(f"{'K=' + str(k) + ' Tree':>13}{'TabPFN':>9}{'差':>10}" for k in BUDGETS))
    tot_rows = tot_pos = 0
    for label, mask in entries:
        n = int(mask.sum())
        p = int(y[mask].sum())
        if label != "合計":
            tot_rows += n
            tot_pos += p
        line = f"{label:<14}{n:>9,}{n / N * 100:>9.2f}%{p:>9,}{n - p:>10,}"
        for k in BUDGETS:
            e = raw.get((k, dim, label))
            if not e:
                line += f"{'—':>13}{'—':>9}{'—':>10}"
            else:
                tr, tb, d, se, same, nn = agg(e, k)
                line += f"{tr:>13.4f}{tb:>9.4f}{d:>+10.4f}"
        print(line)
    print(f"  加總檢核: {tot_rows:,} 列 / {tot_pos:,} 正樣本 "
          f"（應為 {N:,} / {P:,}）{'通過' if tot_rows == N and tot_pos == P else '失敗'}")
