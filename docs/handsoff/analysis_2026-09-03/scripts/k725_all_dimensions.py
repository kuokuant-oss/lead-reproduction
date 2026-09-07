#!/usr/bin/env python3
"""K=725 hot water：所有分類維度的完整分層表。

每個維度都涵蓋全部 73 棟 / 636,121 列 / 90,691 正樣本，缺值自成類別，
沒有異常的組別也列出（它只提供負樣本，PR-AUC 未定義）。
欄位：建物數、列數、佔全部列、正樣本、負樣本、Tree、TabPFN、差、五seed同向。
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
META = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/building_metadata.csv")
HOT_WATER = 3

spec = importlib.util.spec_from_file_location("m5plot", PLOT)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)
formal_root = REPO / "data/processed/m5_building_curve/v5_fixed_50k/model_runs"
ext_root = REPO / "data/processed/m5_building_curve/v5_fixed_50k_k725_row_seed_extension/model_runs"
colab = REPO / "data/processed/m5_building_curve/v5_fixed_50k_k725_colab"
_, tree_dirs = m5.validate_k725(REPO, formal_root, ext_root,
                                colab / "model_results/k725_five_seed_summary.json")

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y, bid, sid = side["y"].astype(np.int8), side["bid"].astype(np.int32), side["sid"].astype(np.int32)
tree_s, tab_s = [], []
for rs in (0, 1, 2, 3, 4):
    with np.load(tree_dirs[rs] / "predictions.npz") as z:
        keep = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        tree_s.append(np.asarray(z["ensemble"], dtype=np.float64)[keep])
    p = (m5.cell_dir(formal_root, 725, rs, 725, "tabpfn") / "predictions.npz" if rs < 2
         else colab / f"model_results/row_seed{rs}/predictions.npz")
    with np.load(p) as z:
        k2 = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        tab_s.append(np.asarray(z["tabpfn"], dtype=np.float64)[k2])

bmeta = pd.read_csv(META).set_index("building_id")
use = bmeta["primary_use"].reindex(bid).to_numpy()
sqft = bmeta["square_feet"].reindex(bid).to_numpy(dtype=float)
year = bmeta["year_built"].reindex(bid).to_numpy(dtype=float)
floors = bmeta["floor_count"].reindex(bid).to_numpy(dtype=float)
N, P = len(y), int(y.sum())


def stats(mask, label):
    n = int(mask.sum())
    yy = y[mask]
    p = int(yy.sum())
    b = len(np.unique(bid[mask]))
    base = {"組別": label, "建物數": b, "列數": n, "佔全部列": n / N * 100,
            "正樣本": p, "負樣本": n - p}
    if p == 0 or p == n:
        base.update({"Tree": np.nan, "TabPFN": np.nan, "差": np.nan,
                     "差SE": np.nan, "同向": "—"})
        return base
    tr = [average_precision_score(yy, s[mask]) for s in tree_s]
    tb = [average_precision_score(yy, s[mask]) for s in tab_s]
    d = np.array(tb) - np.array(tr)
    base.update({"Tree": float(np.mean(tr)), "TabPFN": float(np.mean(tb)),
                 "差": float(d.mean()),
                 "差SE": float(d.std(ddof=1) / math.sqrt(5)),
                 "同向": f"{int((np.sign(d) == np.sign(d.mean())).sum())}/5"})
    return base


def emit(name, entries):
    rows = [stats(m, lab) for m, lab in entries]
    df = pd.DataFrame(rows)
    tot = stats(np.ones(N, dtype=bool), "合計")
    tot["建物數"] = int(df["建物數"].sum())
    tot["列數"] = int(df["列數"].sum())
    tot["佔全部列"] = df["佔全部列"].sum()
    tot["正樣本"] = int(df["正樣本"].sum())
    tot["負樣本"] = int(df["負樣本"].sum())
    df = pd.concat([df, pd.DataFrame([tot])], ignore_index=True)
    ok = (tot["建物數"] == 73 and tot["列數"] == N and tot["正樣本"] == P)
    print(f"\n{'=' * 100}\n{name}   加總檢核: {'通過' if ok else '失敗'}"
          f"（{tot['建物數']} 棟 / {tot['列數']:,} 列 / {tot['正樣本']:,} 正樣本）")
    with pd.option_context("display.width", 300, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(df.to_string(index=False))


def qbins(values, label, n=4, unit=""):
    uniq = pd.DataFrame({"b": bid, "v": values}).drop_duplicates("b")
    ok = uniq[np.isfinite(uniq["v"])]
    edges = np.percentile(ok["v"], np.linspace(0, 100, n + 1))
    out = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        m = ((values >= lo) & (values <= hi)) if i == n - 1 else ((values >= lo) & (values < hi))
        out.append((m, f"{lo:,.0f}–{hi:,.0f}{unit}"))
    miss = ~np.isfinite(values)
    if miss.any():
        out.append((miss, "缺值"))
    return out


emit("A. Site", [(sid == s, f"Site {s}") for s in np.unique(sid)])
emit("B. Primary use", [(use == u, str(u)) for u in sorted(pd.unique(use))])
emit("C. Square feet 四分位", qbins(sqft, "sqft", 4, " sq ft"))
emit("D. Year built", [((year >= lo) & (year < hi), lab) for lo, hi, lab in
                       [(1900, 1950, "1900–1949"), (1950, 1970, "1950–1969"),
                        (1970, 1990, "1970–1989"), (1990, 2020, "1990–2019")]]
     + [(~np.isfinite(year), "缺值")])
emit("E. Floor count", qbins(floors, "floors", 3, " 層"))
