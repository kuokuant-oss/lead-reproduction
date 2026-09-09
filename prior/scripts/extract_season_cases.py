"""Season-length case studies for the TWO seasonal patterns found in by_month.py.

  A. summer / reading == 0   -- Tree's edge on zeros is largest in Jun-Aug
                                (season gap +0.108 vs +0.031 in winter)
  B. winter / reading != 0   -- Tree's edge on non-zero readings is largest in
                                Dec-Feb (+0.093) and negative in summer (-0.067)

Windows are FIXED CALENDAR SPANS, not slid, so each figure really is a whole
season:
  summer = 2016-06-01 00:00 .. 2016-08-31 23:00  (92 days, 2,208 h)
  winter = 2016-01-01 00:00 .. 2016-02-29 23:00  (60 days, 1,440 h)
2016 is one calendar year, so a contiguous Dec-Feb window does not exist --
December sits at the far end of the series. Jan-Feb is the deep-winter core.

Buildings are then ranked, within that fixed window, on the gap for the subset
that pattern is about, and the top two are emitted.
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
DATA = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
            "/analysis_2026-09-08/data")
CACHE = Path("/home/kuant_kuo/.cache/lead_hw_row_time.npz")
OUT = DATA / "season_cases.json"
SK = {"tree": "ensemble", "tabpfn": "tabpfn"}

SEASONS = {
    "summer": ("2016-06-01", "2016-09-01", "夏季 6/1–8/31（92 天）", "r0"),
    "winter": ("2016-01-01", "2016-03-01", "冬季 1/1–2/29（60 天）", "rN"),
}
TOP_N = 3


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


def row_times(reading, bid):
    """Epoch-hours per holdout row, aligned to raw train.csv and cached."""
    if CACHE.is_file():
        with np.load(CACHE) as z:
            print(f"using cached {CACHE.name}")
            return z["t"]
    print("aligning holdout rows to raw timestamps (once, then cached) ...", flush=True)
    parts = []
    for ch in pd.read_csv(RAW, usecols=["building_id", "meter", "timestamp", "meter_reading"],
                          chunksize=2_000_000):
        ch = ch[ch["meter"] == 3]
        if len(ch):
            parts.append(ch)
    raw = pd.concat(parts).sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    t = np.full(reading.size, -1, dtype=np.int64)
    for b in np.unique(bid):
        mb = bid == b
        g = raw[raw["building_id"] == b]
        rv = g["meter_reading"].to_numpy(dtype=float)
        assert rv.size == mb.sum() and np.allclose(rv, reading[mb], atol=1e-6, equal_nan=True), b
        t[mb] = (pd.to_datetime(pd.Series(g["timestamp"].to_numpy()))
                 .astype("datetime64[s]").astype("int64").to_numpy())
    assert (t >= 0).all()
    np.savez_compressed(CACHE, t=t)
    print(f"cached {CACHE}")
    return t


def main():
    with np.load(SIDE, allow_pickle=True) as z:
        y = np.asarray(z["y"]).astype(np.int64)
        reading = np.asarray(z["reading"]).astype(np.float64)
        bid = np.asarray(z["bid"]).astype(np.int64)
        use = np.asarray(z["use"])
    m0 = reading == 0.0
    t = row_times(reading, bid)
    print(f"t range: {t.min()} .. {t.max()}  =&gt; "
          f"{np.datetime64(int(t.min()), chr(115))} .. {np.datetime64(int(t.max()), chr(115))}", flush=True)
    ts = pd.to_datetime(pd.Series(t), unit="s")

    S = {}
    for budget in (400, 725):
        for model in ("tree", "tabpfn"):
            ps = cell_paths(model, budget)
            print(f"averaging {model} K={budget} over {len(ps)} cells ...", flush=True)
            S[budget, model] = np.mean([load_hot(p, SK[model], y) for p in ps], axis=0)

    out = {"seasons": {}}
    for skey, (t0, t1, label, subset) in SEASONS.items():
        lo = int(np.datetime64(t0, "s").astype("int64"))
        hi = int(np.datetime64(t1, "s").astype("int64"))
        inwin = (t >= lo) & (t < hi)
        print("\n" + "=" * 100)
        print(f"{skey.upper()}  {label}  rows in window: {int(inwin.sum()):,}"
              f"  subset of interest: {subset}")
        print("=" * 100)
        rank = []
        for b in np.unique(bid):
            sel = inwin & (bid == b) & (m0 if subset == "r0" else ~m0)
            yv = y[sel]
            na, nn = int((yv == 1).sum()), int((yv == 0).sum())
            need_a, need_n = (30, 30) if subset == "r0" else (20, 100)
            if na < need_a or nn < need_n:
                continue
            at = roc_auc_score(yv, S[400, "tree"][sel])
            ap = roc_auc_score(yv, S[400, "tabpfn"][sel])
            st, sp = S[400, "tree"][sel], S[400, "tabpfn"][sel]
            rank.append({"bid": int(b), "use": str(use[bid == b][0]),
                         "n": int(yv.size), "npos": na, "nneg": nn,
                         "auc_tree": float(at), "auc_tabpfn": float(ap), "gap": float(at - ap),
                         "sep_tree": float(st[yv == 1].mean() - st[yv == 0].mean()),
                         "sep_tabpfn": float(sp[yv == 1].mean() - sp[yv == 0].mean()),
                         "pos_tree": float(st[yv == 1].mean()),
                         "pos_tabpfn": float(sp[yv == 1].mean()),
                         "neg_tree": float(st[yv == 0].mean()),
                         "neg_tabpfn": float(sp[yv == 0].mean())})
        rank.sort(key=lambda r: -r["gap"])
        print(f"{'bid':>5} {'use':<22} {'n':>7} {'pos':>6} {'AUC T':>7} {'AUC P':>7} {'gap':>8}"
              f" | {'pos T':>6} {'pos P':>6} {'neg T':>6} {'neg P':>6}")
        for r in rank[:10]:
            print(f"{r['bid']:>5} {r['use'][:22]:<22} {r['n']:>7,} {r['npos']:>6,}"
                  f" {r['auc_tree']:>7.3f} {r['auc_tabpfn']:>7.3f} {r['gap']:>+8.3f}"
                  f" | {r['pos_tree']:>6.3f} {r['pos_tabpfn']:>6.3f}"
                  f" {r['neg_tree']:>6.3f} {r['neg_tabpfn']:>6.3f}")
        print(f"   ... {len(rank)} buildings qualified; "
              f"gap median {np.median([r['gap'] for r in rank]):+.3f}, "
              f"Tree ahead in {sum(1 for r in rank if r['gap'] > 0)}")

        picks = []
        for r in rank[:TOP_N]:
            b = r["bid"]
            sel = inwin & (bid == b)
            idx = np.flatnonzero(sel)
            idx = idx[np.argsort(t[idx])]
            w = ts.iloc[idx].reset_index(drop=True)
            day0 = w.iloc[0].normalize()
            vv, ya = reading[idx], y[idx]
            zz = vv == 0.0
            CAT = {"zero_anom": zz & (ya == 1), "zero_norm": zz & (ya == 0),
                   "nz_norm": (~zz) & (ya == 0), "nz_anom": (~zz) & (ya == 1)}
            d = {"key": f"{skey}_{b}", "season": skey, "season_label": label, "subset": subset,
                 "bid": b, "use": r["use"], "n": int(idx.size),
                 "t0": str(w.iloc[0]), "t1": str(w.iloc[-1]),
                 "day": ((w.dt.normalize() - day0).dt.days).tolist(),
                 "hh": w.dt.hour.tolist(),
                 "day_labels": [str(day0.date() + pd.Timedelta(days=i))
                                for i in range(int((w.dt.normalize() - day0).dt.days.max()) + 1)],
                 "v": [round(float(x), 2) for x in vv],
                 "a": ya.astype(int).tolist(),
                 "st": [round(float(x), 4) for x in S[400, "tree"][idx]],
                 "sp": [round(float(x), 4) for x in S[400, "tabpfn"][idx]],
                 "win": r,
                 "win_zeros": int(zz.sum()),
                 "win_zeros_anom": int((zz & (ya == 1)).sum()),
                 "win_zeros_norm": int((zz & (ya == 0)).sum()),
                 "win_nz_anom": int(((~zz) & (ya == 1)).sum()),
                 "win_nz_norm": int(((~zz) & (ya == 0)).sum()),
                 "nz_med": (float(np.median(vv[vv > 0])) if (vv > 0).any() else None),
                 "nz_max": (float(vv.max()) if vv.size else None),
                 "budget": 400, "stats": {}}
            aa = np.asarray(d["a"])
            d["flips"] = [{"i": int(i), "t": str(w.iloc[int(i)]), "to": int(aa[int(i)])}
                          for i in (np.flatnonzero(np.diff(aa) != 0) + 1)]
            for budget in (400, 725):
                for model in ("tree", "tabpfn"):
                    sm = S[budget, model][idx]
                    e = {k: (float(sm[c].mean()) if c.any() else None) for k, c in CAT.items()}
                    e["sep0"] = (None if e["zero_anom"] is None or e["zero_norm"] is None
                                 else e["zero_anom"] - e["zero_norm"])
                    e["sepN"] = (None if e["nz_anom"] is None or e["nz_norm"] is None
                                 else e["nz_anom"] - e["nz_norm"])
                    d["stats"][f"{budget}|{model}"] = e
            picks.append(d)
            print(f"   -> emit b{b}: {d['t0'][:10]}..{d['t1'][:10]}  {d['n']:,} h, "
                  f"{len(d['flips'])} flips, zeros {d['win_zeros']:,} "
                  f"({d['win_zeros_anom']:,}A/{d['win_zeros_norm']:,}N), "
                  f"nz {d['win_nz_anom']:,}A/{d['win_nz_norm']:,}N")
        out["seasons"][skey] = {"label": label, "subset": subset, "t0": t0, "t1": t1,
                                "ranking": rank, "picks": picks}

    OUT.write_text(json.dumps(out, separators=(",", ":"), default=float), encoding="utf-8")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
