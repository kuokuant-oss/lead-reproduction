"""The two cases that REPLACE the three 30-day ones (fig 3a/3b/3c).

Pattern asked for: summer -> TabPFN ahead, winter -> Tree ahead, BOTH on reading == 0.
Note the POOLED summer r0 gap is +0.108 (Tree ahead) -- individual buildings go
both ways (Tree 11/18, TabPFN 7/18), and the case shows one of the TabPFN ones.

Windows are two contiguous calendar months, chosen where the pooled effect is
strongest:
  summer  2016-07-01 .. 2016-08-31   (62 days)   pick a TabPFN-ahead building
  winter  2016-11-01 .. 2016-12-31   (61 days)   pick a Tree-ahead building
Nov+Dec also avoids the calendar wrap that made Dec-Feb impossible.

The building is picked as the TYPICAL one, not the extreme: among buildings that
qualify in that window, the one whose gap is closest to the MEDIAN gap. The full
ranking is emitted so the figure can be shown to be representative.
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
DATA = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
            "/analysis_2026-09-08/data")
CACHE = Path("/home/kuant_kuo/.cache/lead_hw_row_time.npz")
OUT = DATA / "rn_season_cases.json"
SK = {"tree": "ensemble", "tabpfn": "tabpfn"}
BUDGETS = (400, 725)

WINDOWS = {
    "summer": {"months": [7, 8], "label": "夏季 7/1–8/31（62 天）",
               "expect": "tree", "t0": "2016-07-01", "t1": "2016-08-31"},
    "winter": {"months": [11, 12], "label": "冬季 11/1–12/31（61 天）",
               "expect": "tree", "t0": "2016-11-01", "t1": "2016-12-31"},
}
SUBSET = "r0"
MIN_POS, MIN_NEG = 10, 10


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
        use = np.asarray(z["use"])
    m0 = reading == 0.0
    mN = ~m0
    with np.load(CACHE) as z:
        t = z["t"]
    ts = pd.to_datetime(pd.Series(t), unit="s")
    mon = ts.dt.month.to_numpy()

    S, SD = {}, {}
    for budget in BUDGETS:
        for model in ("tree", "tabpfn"):
            ps = cell_paths(model, budget)
            print(f"averaging {model} K={budget} over {len(ps)} cells ...", flush=True)
            st = np.stack([load_hot(p, SK[model], y) for p in ps])
            S[budget, model] = st.mean(axis=0)
            SD[budget, model] = st.std(axis=0, ddof=1)

    out = {}
    for wkey, W in WINDOWS.items():
        wm = np.isin(mon, W["months"])
        # ---- pooled numbers for this window, both subsets, both budgets
        pooled = {}
        for budget in BUDGETS:
            for sub, sm in (("r0", m0), ("rN", mN)):
                sel = wm & sm
                yv = y[sel]
                at = roc_auc_score(yv, S[budget, "tree"][sel])
                ap = roc_auc_score(yv, S[budget, "tabpfn"][sel])
                pooled[f"{budget}|{sub}"] = {"tree": float(at), "tabpfn": float(ap),
                                             "gap": float(at - ap), "n": int(yv.size),
                                             "npos": int((yv == 1).sum())}
        sel0 = wm & m0
        pooled["zero_rate"] = float(wm[wm].size and (wm & m0).sum() / wm.sum())
        pooled["p_anom_zero"] = float((y[sel0] == 1).mean())
        pooled["rows"] = int(wm.sum())

        # ---- per-building ranking on the subset of interest
        rank = []
        for b in np.unique(bid):
            sel = wm & (bid == b) & (m0 if SUBSET == "r0" else mN)
            yv = y[sel]
            na, nn = int((yv == 1).sum()), int((yv == 0).sum())
            if na < MIN_POS or nn < MIN_NEG:
                continue
            at = roc_auc_score(yv, S[400, "tree"][sel])
            ap = roc_auc_score(yv, S[400, "tabpfn"][sel])
            stv, spv = S[400, "tree"][sel], S[400, "tabpfn"][sel]
            rank.append({"bid": int(b), "use": str(use[bid == b][0]),
                         "npos": na, "nneg": nn,
                         "auc_tree": float(at), "auc_tabpfn": float(ap), "gap": float(at - ap),
                         "pos_tree": float(stv[yv == 1].mean()),
                         "pos_tabpfn": float(spv[yv == 1].mean()),
                         "neg_tree": float(stv[yv == 0].mean()),
                         "neg_tabpfn": float(spv[yv == 0].mean())})
        rank.sort(key=lambda r: r["gap"])
        gaps = np.array([r["gap"] for r in rank])
        med = float(np.median(gaps))
        n_expect = (sum(1 for g in gaps if g < 0) if W["expect"] == "tabpfn"
                    else sum(1 for g in gaps if g > 0))
        print("\n" + "=" * 100)
        print(f"{wkey.upper()}  {W['label']}  期待 {W['expect']} 領先")
        print(f"  合併 rN gap K=400: {pooled['400|rN']['gap']:+.3f} "
              f"(Tree {pooled['400|rN']['tree']:.3f} / TabPFN {pooled['400|rN']['tabpfn']:.3f})")
        print(f"  合併 r0 gap K=400: {pooled['400|r0']['gap']:+.3f}"
              f"   零讀數率 {pooled['zero_rate']:.1%}  P(異常|0) {pooled['p_anom_zero']:.1%}")
        print(f"  可評分建築 {len(rank)} 棟；符合方向 {n_expect} 棟；gap 中位數 {med:+.3f}")
        print("=" * 100)
        print(f"{'bid':>6} {'use':<24} {'pos':>5} {'neg':>6} {'AUC T':>7} {'AUC P':>7} {'gap':>8}"
              f" | {'pos T':>6} {'pos P':>6}")
        for r in rank:
            print(f"{r['bid']:>6} {r['use'][:24]:<24} {r['npos']:>5,} {r['nneg']:>6,}"
                  f" {r['auc_tree']:>7.3f} {r['auc_tabpfn']:>7.3f} {r['gap']:>+8.3f}"
                  f" | {r['pos_tree']:>6.3f} {r['pos_tabpfn']:>6.3f}")

        # ---- TYPICAL pick: gap closest to the median, and in the expected direction
        PICK_MIN = 50
        cand = [r for r in rank
                if r["npos"] >= PICK_MIN and r["nneg"] >= PICK_MIN
                if (r["gap"] < 0 if W["expect"] == "tabpfn" else r["gap"] > 0)]
        pool = cand if cand else rank
        med_pool = float(np.median([r["gap"] for r in pool]))
        pick = min(pool, key=lambda r: abs(r["gap"] - med_pool))
        b = pick["bid"]
        print(f"  -> 一般狀況個案: b{b} ({pick['use']}) gap {pick['gap']:+.3f}"
              f"  (符合方向者的中位數 {med_pool:+.3f}，全體中位數 {med:+.3f})")

        idx = np.flatnonzero(wm & (bid == b))
        idx = idx[np.argsort(t[idx])]
        w = ts.iloc[idx].reset_index(drop=True)
        day0 = w.iloc[0].normalize()
        vv, ya = reading[idx], y[idx]
        zz = vv == 0.0
        CAT = {"zero_anom": zz & (ya == 1), "zero_norm": zz & (ya == 0),
               "nz_norm": (~zz) & (ya == 0), "nz_anom": (~zz) & (ya == 1)}
        d = {"key": wkey, "label": W["label"], "expect": W["expect"],
             "bid": b, "use": pick["use"], "n": int(idx.size), "budget": 400,
             "t0": str(w.iloc[0]), "t1": str(w.iloc[-1]),
             "day": ((w.dt.normalize() - day0).dt.days).tolist(),
             "hh": w.dt.hour.tolist(),
             "day_labels": [str(day0.date() + pd.Timedelta(days=i))
                            for i in range(int((w.dt.normalize() - day0).dt.days.max()) + 1)],
             "v": [round(float(x), 2) for x in vv],
             "a": ya.astype(int).tolist(),
             "st": [round(float(x), 4) for x in S[400, "tree"][idx]],
             "sp": [round(float(x), 4) for x in S[400, "tabpfn"][idx]],
             "win": pick, "pooled": pooled, "ranking": rank,
             "n_scorable": len(rank), "n_expect": n_expect, "median_gap": med,
             "median_gap_pool": med_pool, "subset": SUBSET,
             "win_zeros": int(zz.sum()),
             "win_zeros_anom": int((zz & (ya == 1)).sum()),
             "win_zeros_norm": int((zz & (ya == 0)).sum()),
             "win_nz_anom": int(((~zz) & (ya == 1)).sum()),
             "win_nz_norm": int(((~zz) & (ya == 0)).sum()),
             "nz_med": (float(np.median(vv[vv > 0])) if (vv > 0).any() else None),
             "nz_max": (float(vv.max()) if vv.size else None), "stats": {}}
        aa = np.asarray(d["a"])
        d["flips"] = [{"i": int(i), "t": str(w.iloc[int(i)]), "to": int(aa[int(i)])}
                      for i in (np.flatnonzero(np.diff(aa) != 0) + 1)]
        for budget in BUDGETS:
            for model in ("tree", "tabpfn"):
                sm, sd = S[budget, model][idx], SD[budget, model][idx]
                e = {k: (float(sm[c].mean()) if c.any() else None) for k, c in CAT.items()}
                e["sep0"] = (None if e["zero_anom"] is None or e["zero_norm"] is None
                             else e["zero_anom"] - e["zero_norm"])
                e["sepN"] = (None if e["nz_anom"] is None or e["nz_norm"] is None
                             else e["nz_anom"] - e["nz_norm"])
                e["seed_sd"] = float(sd.mean())
                d["stats"][f"{budget}|{model}"] = e
        print(f"     {d['t0'][:10]}..{d['t1'][:10]}  {d['n']:,} h, {len(d['flips'])} flips, "
              f"zeros {d['win_zeros']:,} ({d['win_zeros_anom']}A/{d['win_zeros_norm']}N), "
              f"nz {d['win_nz_anom']}A/{d['win_nz_norm']:,}N")
        for budget in BUDGETS:
            for model in ("tree", "tabpfn"):
                e = d["stats"][f"{budget}|{model}"]
                q = lambda k, s=False: ("  n/a" if e[k] is None
                                        else (f"{e[k]:+.3f}" if s else f"{e[k]:.3f}"))
                print(f"     K={budget} {model:6}: nz-anom {q('nz_anom')} | nz-norm {q('nz_norm')}"
                      f" | sepN {q('sepN', True)} | zero-anom {q('zero_anom')}"
                      f" | zero-norm {q('zero_norm')}")
        out[wkey] = d

    OUT.write_text(json.dumps(out, separators=(",", ":"), default=float), encoding="utf-8")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
