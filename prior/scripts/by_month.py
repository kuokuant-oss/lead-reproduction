"""Is the Tree-vs-TabPFN gap seasonal?  Measure it, per month.

Month is not in the side-info, so every holdout row is aligned to a raw
train.csv timestamp: for each building the raw rows (meter==3, sorted by
timestamp) are checked to have the same count AND the same readings as the
holdout rows, so the k-th holdout row is the k-th raw row.

The alignment is then ANCHORED against the earlier month-coverage table
(docs/handsoff/analysis_2026-09-03/hot_water_month_coverage.md): per-month row
counts, zero rate and P(anomaly | reading==0) must reproduce it exactly.

Then, per month and per budget: ROC-AUC for both models on reading==0,
reading!=0 and pooled, plus the paired gap.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k")
BD = REPO / "data/processed/m5_building_curve"
FORMAL = BD / "v5_fixed_50k/model_runs"
EXT = BD / "v5_fixed_50k_k725_row_seed_extension/model_runs"
COLAB = BD / "v5_fixed_50k_k725_colab/model_results"
SIDE = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
            "/analysis_2026-09-03/data/hotwater_sideinfo.npz")
RAW = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/train.csv")
OUT = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
           "/analysis_2026-09-08/data/by_month.json")
SK = {"tree": "ensemble", "tabpfn": "tabpfn"}
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# from hot_water_month_coverage.md -- the anchor
ANCHOR_ROWS = [53964, 49573, 52656, 52511, 54086, 52420, 54257, 53909, 52418, 54171, 52382, 53774]
ANCHOR_ZERO_RATE = [.113, .163, .200, .256, .297, .323, .334, .347, .372, .389, .240, .153]
ANCHOR_P_ANOM_ZERO = [.576, .514, .533, .623, .466, .386, .355, .371, .418, .539, .627, .529]


def cell_paths(model, budget):
    sub = "tree_no_es" if model == "tree" else "tabpfn"
    if budget != 725:
        return [FORMAL / f"building_seed{b}" / f"row_seed{r}" / f"{sub}_k{budget}_f137"
                / "predictions.npz" for b in range(5) for r in range(2)]
    out = []
    for r in range(5):
        if model == "tree":
            root = FORMAL if r < 2 else EXT
            out.append(root / "building_seed725" / f"row_seed{r}"
                       / "tree_no_es_k725_f137" / "predictions.npz")
        else:
            out.append(FORMAL / "building_seed725" / f"row_seed{r}" / "tabpfn_k725_f137"
                       / "predictions.npz" if r < 2 else COLAB / f"row_seed{r}" / "predictions.npz")
    return out


def load_hot(path, key, y_side):
    with np.load(path) as p:
        hot = np.asarray(p["meter"]).astype(np.int64) == 3
        anom = np.asarray(p["anomaly"]).astype(np.int64)[hot]
        s = np.asarray(p[key]).astype(np.float64)[hot]
    assert np.array_equal(anom, y_side), path
    return s


def main():
    with np.load(SIDE, allow_pickle=True) as z:
        y = np.asarray(z["y"]).astype(np.int64)
        reading = np.asarray(z["reading"]).astype(np.float64)
        bid = np.asarray(z["bid"]).astype(np.int64)
    m0 = reading == 0.0

    # ------------------------------------------------ month for every holdout row
    print("reading raw timestamps for all 73 buildings ...", flush=True)
    parts = []
    for ch in pd.read_csv(RAW, usecols=["building_id", "meter", "timestamp", "meter_reading"],
                          chunksize=2_000_000):
        ch = ch[ch["meter"] == 3]
        if len(ch):
            parts.append(ch)
    raw = pd.concat(parts).sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    month = np.full(y.size, -1, dtype=np.int64)
    bad = []
    for b in np.unique(bid):
        mb = bid == b
        g = raw[raw["building_id"] == b]
        rv = g["meter_reading"].to_numpy(dtype=float)
        hold = reading[mb]
        if rv.size != hold.size or not np.allclose(rv, hold, atol=1e-6, equal_nan=True):
            bad.append((int(b), int(rv.size), int(hold.size)))
            continue
        month[mb] = pd.to_datetime(pd.Series(g["timestamp"].to_numpy())).dt.month.to_numpy() - 1
    print(f"aligned {int((month >= 0).sum()):,} of {y.size:,} rows; "
          f"{len(bad)} buildings failed" + (f" {bad}" if bad else ""))
    assert not bad, "some buildings did not align -- month assignment unsafe"

    # ------------------------------------------------ anchor check
    print("\nANCHOR CHECK vs hot_water_month_coverage.md")
    print(f"{'mon':>4} {'rows mine':>10} {'rows doc':>9} {'zero% mine':>11} {'doc':>6}"
          f" {'P(a|0) mine':>12} {'doc':>6}")
    ok = True
    for i, nm in enumerate(MON):
        sel = month == i
        n = int(sel.sum())
        zr = float((sel & m0).sum()) / n
        pa = float((sel & m0 & (y == 1)).sum()) / max(int((sel & m0).sum()), 1)
        d_n, d_zr, d_pa = ANCHOR_ROWS[i], ANCHOR_ZERO_RATE[i], ANCHOR_P_ANOM_ZERO[i]
        good = (n == d_n) and abs(zr - d_zr) < 0.0006 and abs(pa - d_pa) < 0.0006
        ok &= good
        print(f"{nm:>4} {n:>10,} {d_n:>9,} {zr:>10.1%} {d_zr:>6.1%}"
              f" {pa:>11.1%} {d_pa:>6.1%}  {'ok' if good else 'MISMATCH'}")
    print("=> month alignment", "PASS" if ok else "FAIL")
    assert ok, "month alignment does not reproduce the published coverage table"

    # ------------------------------------------------ scores
    S = {}
    for budget in (400, 725):
        for model in ("tree", "tabpfn"):
            ps = cell_paths(model, budget)
            print(f"averaging {model} K={budget} over {len(ps)} cells ...", flush=True)
            S[budget, model] = np.mean([load_hot(p, SK[model], y) for p in ps], axis=0)

    res = {"months": MON, "rows": []}
    for budget in (400, 725):
        print("\n" + "=" * 104)
        print(f"ROC-AUC BY MONTH, K = {budget}   (gap = Tree - TabPFN, positive favours Tree)")
        print("=" * 104)
        print(f"{'mon':>4} {'rows':>8} {'anom%':>7} {'0-rate':>7} {'P(a|0)':>7} |"
              f" {'r0 T':>6} {'r0 P':>6} {'r0 gap':>8} |"
              f" {'rN T':>6} {'rN P':>6} {'rN gap':>8} |"
              f" {'all T':>6} {'all P':>6} {'all gap':>8}")
        for i, nm in enumerate(MON):
            sel = month == i
            n = int(sel.sum())
            row = {"budget": budget, "month": nm, "n": n,
                   "anom_rate": float((y[sel] == 1).mean()),
                   "zero_rate": float(m0[sel].mean()),
                   "p_anom_zero": float((y[sel & m0] == 1).mean())}
            cells = []
            for tag, msk in (("r0", sel & m0), ("rN", sel & ~m0), ("all", sel)):
                yv = y[msk]
                if yv.size == 0 or yv.min() == yv.max():
                    row[tag] = None
                    cells.append(("   n/a", "   n/a", "     n/a"))
                    continue
                t = roc_auc_score(yv, S[budget, "tree"][msk])
                p = roc_auc_score(yv, S[budget, "tabpfn"][msk])
                row[tag] = {"tree": float(t), "tabpfn": float(p), "gap": float(t - p),
                            "n": int(yv.size), "npos": int((yv == 1).sum())}
                cells.append((f"{t:6.3f}", f"{p:6.3f}", f"{t-p:+8.3f}"))
            res["rows"].append(row)
            print(f"{nm:>4} {n:>8,} {row['anom_rate']:>6.1%} {row['zero_rate']:>6.1%}"
                  f" {row['p_anom_zero']:>6.1%} |"
                  f" {cells[0][0]} {cells[0][1]} {cells[0][2]} |"
                  f" {cells[1][0]} {cells[1][1]} {cells[1][2]} |"
                  f" {cells[2][0]} {cells[2][1]} {cells[2][2]}")
        # season roll-up
        SEASON = {"冬 Dec-Feb": [11, 0, 1], "春 Mar-May": [2, 3, 4],
                  "夏 Jun-Aug": [5, 6, 7], "秋 Sep-Nov": [8, 9, 10]}
        print(f"\n{'season':<12} {'rows':>8} | {'r0 T':>6} {'r0 P':>6} {'r0 gap':>8} |"
              f" {'rN gap':>8} | {'all gap':>8}")
        for nm, idx in SEASON.items():
            sel = np.isin(month, idx)
            out2 = []
            for msk in (sel & m0, sel & ~m0, sel):
                yv = y[msk]
                t = roc_auc_score(yv, S[budget, "tree"][msk])
                p = roc_auc_score(yv, S[budget, "tabpfn"][msk])
                out2.append((t, p))
            res.setdefault("seasons", []).append(
                {"budget": budget, "season": nm, "n": int(sel.sum()),
                 "r0": {"tree": out2[0][0], "tabpfn": out2[0][1], "gap": out2[0][0] - out2[0][1]},
                 "rN": {"tree": out2[1][0], "tabpfn": out2[1][1], "gap": out2[1][0] - out2[1][1]},
                 "all": {"tree": out2[2][0], "tabpfn": out2[2][1], "gap": out2[2][0] - out2[2][1]}})
            print(f"{nm:<12} {int(sel.sum()):>8,} | {out2[0][0]:6.3f} {out2[0][1]:6.3f}"
                  f" {out2[0][0]-out2[0][1]:+8.3f} | {out2[1][0]-out2[1][1]:+8.3f}"
                  f" | {out2[2][0]-out2[2][1]:+8.3f}")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
