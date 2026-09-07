#!/usr/bin/env python3
"""hot water 分層 PR-AUC，跨 K=100/200/400/725。

分組結構（建物數、列數、正負樣本）在四個 K 完全相同，因為 holdout 固定；
只有 Tree / TabPFN 的值隨 K 改變。
每個維度都涵蓋全部 73 棟 / 636,121 列 / 90,691 正樣本，缺值自成類別。
K≤400：先在 building seed 內平均 row seed，再跨 5 個 building seed（SE 為 seed 層級）。
K=725：跨 5 個 row seed。
輸出 markdown 檔。
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
OUTDIR = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff/analysis_2026-09-03")
META = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/building_metadata.csv")
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
print("[gate] validate_k725 通過", flush=True)

side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y = side["y"].astype(np.int8)
bid = side["bid"].astype(np.int32)
sid = side["sid"].astype(np.int32)
reading = side["reading"]
N, P = len(y), int(y.sum())

bmeta = pd.read_csv(META).set_index("building_id")
use = bmeta["primary_use"].reindex(bid).to_numpy()
sqft = bmeta["square_feet"].reindex(bid).to_numpy(dtype=float)
year = bmeta["year_built"].reindex(bid).to_numpy(dtype=float)
floors = bmeta["floor_count"].reindex(bid).to_numpy(dtype=float)


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


DIMENSIONS = {
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
for name, entries in DIMENSIONS.items():
    entries.append(("合計", np.ones(N, dtype=bool)))


def cells_for(budget):
    if budget == 725:
        return [(725, rs) for rs in (0, 1, 2, 3, 4)]
    return [(b, r) for b in range(5) for r in (0, 1)]


def load_scores(budget, bseed, rseed):
    if budget == 725:
        tdir = k725_tree_dirs[rseed]
        tab = (m5.cell_dir(formal_root, 725, rseed, 725, "tabpfn") / "predictions.npz"
               if rseed < 2 else colab / f"model_results/row_seed{rseed}/predictions.npz")
    else:
        tdir = m5.cell_dir(formal_root, bseed, rseed, budget, "ensemble")
        tab = m5.cell_dir(formal_root, bseed, rseed, budget, "tabpfn") / "predictions.npz"
        m5.validate_pair(
            m5.validate_cell(tdir, budget, bseed, rseed, "ensemble"),
            m5.validate_cell(tab.parent, budget, bseed, rseed, "tabpfn"),
            f"k{budget}_b{bseed}_r{rseed}")
    with np.load(tdir / "predictions.npz") as z:
        keep = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        s_tree = np.asarray(z["ensemble"], dtype=np.float64)[keep]
        if not np.array_equal(np.asarray(z["anomaly"])[keep].astype(np.int8), y):
            raise RuntimeError("holdout 不符")
    with np.load(tab) as z:
        k2 = np.asarray(z["meter"]).astype(np.int8) == HOT_WATER
        s_tab = np.asarray(z["tabpfn"], dtype=np.float64)[k2]
        if not np.array_equal(np.asarray(z["anomaly"])[k2].astype(np.int8), y):
            raise RuntimeError("holdout 不符")
    return s_tree, s_tab


# raw[(budget, dim, group)] = list of ((bseed, rseed), ap_tree, ap_tab)
raw: dict = {}
for budget in BUDGETS:
    for bseed, rseed in cells_for(budget):
        s_tree, s_tab = load_scores(budget, bseed, rseed)
        for dim, entries in DIMENSIONS.items():
            for label, mask in entries:
                yy = y[mask]
                p = int(yy.sum())
                if p == 0 or p == len(yy):
                    continue
                key = (budget, dim, label)
                raw.setdefault(key, []).append(
                    ((bseed, rseed),
                     float(average_precision_score(yy, s_tree[mask])),
                     float(average_precision_score(yy, s_tab[mask]))))
        del s_tree, s_tab
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


def struct(mask):
    n = int(mask.sum())
    p = int(y[mask].sum())
    return len(np.unique(bid[mask])), n, n / N * 100, p, n - p


lines = ["# Hot water 分層 PR-AUC：K = 100 / 200 / 400 / 725", "",
         "分組結構（建物數、列數、正負樣本）在四個 K 完全相同，因為 holdout 固定；",
         "只有 Tree / TabPFN 的值隨 K 改變。",
         "",
         "聚合：K≤400 先在 building seed 內平均 row seed，再跨 5 個 building seed；",
         "K=725 跨 5 個 row seed。SE 為該層級的標準誤。「同向」= 五個 seed 中與平均同號的個數。",
         ""]

for dim, entries in DIMENSIONS.items():
    lines += [f"## {dim}", ""]
    hdr = ("| 組別 | 建物數 | 列數 | 佔全部列 | 正樣本 | 負樣本 | "
           + " | ".join(f"K={k} Tree | K={k} TabPFN" for k in BUDGETS) + " |")
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | " + " | ".join(["---:"] * 8) + " |"
    lines += [hdr, sep]
    for label, mask in entries:
        b, n, pct, pos, neg = struct(mask)
        cells_txt = []
        for k in BUDGETS:
            e = raw.get((k, dim, label))
            if not e:
                cells_txt += ["—", "—"]
            else:
                tr, tb, *_ = agg(e, k)
                cells_txt += [f"{tr:.4f}", f"{tb:.4f}"]
        lines.append(f"| {label} | {b} | {n:,} | {pct:.2f}% | {pos:,} | {neg:,} | "
                     + " | ".join(cells_txt) + " |")
    lines += ["", f"### {dim} — 差（TabPFN − Tree）", "",
              "| 組別 | 正樣本 | " + " | ".join(f"K={k}" for k in BUDGETS) + " | 同向 (100/200/400/725) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | :--- |"]
    for label, mask in entries:
        _, _, _, pos, _ = struct(mask)
        ds, sd = [], []
        for k in BUDGETS:
            e = raw.get((k, dim, label))
            if not e:
                ds.append("—")
                sd.append("—")
            else:
                _, _, d, se, same, nn = agg(e, k)
                ds.append(f"{d:+.4f} ± {se:.4f}")
                sd.append(f"{same}/{nn}")
        lines.append(f"| {label} | {pos:,} | " + " | ".join(ds) + " | " + " ".join(sd) + " |")
    lines.append("")

OUTDIR.mkdir(parents=True, exist_ok=True)
out = OUTDIR / "hot_water_strata_by_budget.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"\n[saved] {out}")

# 終端摘要：只印各維度的「差」
for dim, entries in DIMENSIONS.items():
    print(f"\n=== {dim} — 差 (TabPFN − Tree) ===")
    print(f"{'組別':<30}{'正樣本':>9}" + "".join(f"{'K=' + str(k):>12}" for k in BUDGETS))
    for label, mask in entries:
        _, _, _, pos, _ = struct(mask)
        row = f"{label:<30}{pos:>9,}"
        for k in BUDGETS:
            e = raw.get((k, dim, label))
            row += f"{'—':>12}" if not e else f"{agg(e, k)[2]:>+12.4f}"
        print(row)
