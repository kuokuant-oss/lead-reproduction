"""Emit the RE-PICKED case windows (see search_cases.py for how they were chosen).

  textbook  b1227 Education  -- a 216 h zero outage that RECOVERS inside the window,
                                so the reader sees normal -> outage -> normal
  tree_win  b255  Education  -- Tree clearly better on this building's zeros:
                                within-building r0 AUC .949 vs .829, score
                                separation +0.567 vs +0.287
  tabpfn_win b145 Office     -- TabPFN's score separation is 3x Tree's here
                                (+0.616 vs +0.206), though the AUC gap is small

Scores: both models, K=400 (10 cells) plotted; K=725 (5 cells) also summarised.
Timestamps come from raw train.csv and are verified against the holdout rows.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
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
           "/analysis_2026-09-08/data/month_cases.json")
BUDGET, BUDGETS = 400, (400, 725)
SK = {"tree": "ensemble", "tabpfn": "tabpfn"}
W = 720

# name -> (building, how to place the 30-day window)
PICKS = [("textbook", 1227, "outage"), ("tree_win", 255, "tree"), ("tabpfn_win", 145, "tabpfn")]


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


def runs(mask):
    d = np.diff(np.r_[0, mask.view(np.int8), 0])
    st = np.flatnonzero(d == 1)
    return st, np.flatnonzero(d == -1) - st


def main():
    with np.load(SIDE, allow_pickle=True) as z:
        y = np.asarray(z["y"]).astype(np.int64)
        reading = np.asarray(z["reading"]).astype(np.float64)
        bid = np.asarray(z["bid"]).astype(np.int64)
        hour = np.asarray(z["hour"]).astype(np.int64)
        use = np.asarray(z["use"])
    SCORE, SPREAD = {}, {}
    for budget in BUDGETS:
        for model in ("tree", "tabpfn"):
            ps = cell_paths(model, budget)
            print(f"averaging {model} K={budget} over {len(ps)} cells ...", flush=True)
            st = np.stack([load_hot(p, SK[model], y) for p in ps])
            SCORE[budget, model] = st.mean(axis=0)
            SPREAD[budget, model] = st.std(axis=0, ddof=1)

    out = {}
    for name, b, how in PICKS:
        mb = bid == b
        v, yb, hb = reading[mb], y[mb], hour[mb]
        z = v == 0.0
        if how in ("tree", "tabpfn"):
            # window the case on MODEL DISAGREEMENT, not on class balance: among
            # windows holding >= MIN_EACH of both kinds of zero, take the one where
            # the favoured model's score separation most exceeds the other's.
            MIN_EACH = 100
            za, zn = z & (yb == 1), z & (yb == 0)
            cza, czn = np.r_[0, np.cumsum(za)], np.r_[0, np.cumsum(zn)]
            cs = {}
            for model in ("tree", "tabpfn"):
                sm = SCORE[BUDGET, model][mb]
                cs[model, "a"] = np.r_[0, np.cumsum(sm * za)]
                cs[model, "n"] = np.r_[0, np.cumsum(sm * zn)]
            s = np.arange(0, v.size - W)
            na = cza[s + W] - cza[s]
            nn = czn[s + W] - czn[s]
            ok = (na >= MIN_EACH) & (nn >= MIN_EACH)
            sep = {}
            for model in ("tree", "tabpfn"):
                ma = np.divide(cs[model, "a"][s + W] - cs[model, "a"][s], na,
                               out=np.zeros(s.size), where=na > 0)
                mn = np.divide(cs[model, "n"][s + W] - cs[model, "n"][s], nn,
                               out=np.zeros(s.size), where=nn > 0)
                sep[model] = ma - mn
            obj = (sep["tree"] - sep["tabpfn"]) if how == "tree" else (sep["tabpfn"] - sep["tree"])
            obj = np.where(ok, obj, -np.inf)
            if not np.isfinite(obj).any():
                raise RuntimeError(f"no window with >= {MIN_EACH} of each zero kind for b{b}")
            i = int(np.argmax(obj))
            start = int(s[i])
            print(f"   [{name}] windowed on disagreement: sep tree {sep['tree'][i]:+.3f} "
                  f"vs tabpfn {sep['tabpfn'][i]:+.3f} (obj {obj[i]:+.3f}), "
                  f"in-window zeros {na[i]}/{nn[i]}")
        else:  # centre on the longest fully-anomalous zero-run that recovers
            st_, ln_ = runs(z)
            cand = [(int(a), int(L)) for a, L in zip(st_, ln_)
                    if 72 <= L <= 240 and yb[a:a + L].mean() >= 0.95
                    and a >= 240 and a + L + 240 <= v.size]
            a, L = max(cand, key=lambda t: t[1])
            start = max(0, a - (W - L) // 2)
        while start > 0 and hb[start] != 0:
            start -= 1
        start = min(start, v.size - W)

        sl = slice(start, start + W)
        vv, ya = v[sl], yb[sl]
        zz = vv == 0.0
        st_, ln_ = runs(zz)
        # whole-year within-building r0 numbers, for the caption
        yr = {}
        m0b = z
        if (yb[m0b] == 1).sum() >= 2 and (yb[m0b] == 0).sum() >= 2:
            for model in ("tree", "tabpfn"):
                s400 = SCORE[400, model][mb][m0b]
                yr[model] = {"auc": float(roc_auc_score(yb[m0b], s400)),
                             "sep": float(s400[yb[m0b] == 1].mean() - s400[yb[m0b] == 0].mean())}
        d = {"key": name, "bid": b, "use": str(use[mb][0]), "start": start, "n": int(vv.size),
             "budget": BUDGET, "why": how,
             "building_n": int(mb.sum()), "building_zeros": int(z.sum()),
             "building_zeros_anom": int((z & (yb == 1)).sum()),
             "building_zeros_norm": int((z & (yb == 0)).sum()),
             "building_anom": int((yb == 1).sum()),
             "year": yr,
             "win_zeros": int(zz.sum()),
             "win_zeros_anom": int((zz & (ya == 1)).sum()),
             "win_zeros_norm": int((zz & (ya == 0)).sum()),
             "win_anom": int((ya == 1).sum()),
             "win_nonzero_anom": int(((~zz) & (ya == 1)).sum()),
             "zero_runs": [(int(a), int(L), int(ya[a:a + L].sum())) for a, L in zip(st_, ln_)],
             "v": [round(float(x), 2) for x in vv],
             "a": ya.astype(int).tolist(),
             "st": [round(float(x), 4) for x in SCORE[BUDGET, "tree"][mb][sl]],
             "sp": [round(float(x), 4) for x in SCORE[BUDGET, "tabpfn"][mb][sl]]}
        nz = vv[vv > 0]
        d["nz_min"] = float(nz.min()) if nz.size else None
        d["nz_med"] = float(np.median(nz)) if nz.size else None
        d["nz_max"] = float(nz.max()) if nz.size else None
        CAT = {"zero_anom": zz & (ya == 1), "zero_norm": zz & (ya == 0),
               "nz_norm": (~zz) & (ya == 0), "nz_anom": (~zz) & (ya == 1)}
        d["stats"] = {}
        for budget in BUDGETS:
            for model in ("tree", "tabpfn"):
                sm, sd = SCORE[budget, model][mb][sl], SPREAD[budget, model][mb][sl]
                e = {k: (float(sm[sel].mean()) if sel.any() else None) for k, sel in CAT.items()}
                e["sep"] = (None if e["zero_anom"] is None or e["zero_norm"] is None
                            else e["zero_anom"] - e["zero_norm"])
                e["seed_sd"] = float(sd.mean())
                d["stats"][f"{budget}|{model}"] = e
        out[name] = d
        print(f"\n[{name}] b{b} {d['use']}  window h={start}  "
              f"zeros {d['win_zeros']} (anom {d['win_zeros_anom']} / norm {d['win_zeros_norm']})"
              f"  non-zero anom {d['win_nonzero_anom']}")
        if yr:
            print(f"   whole-year r0 (K=400): AUC {yr['tree']['auc']:.3f}/{yr['tabpfn']['auc']:.3f}"
                  f"  sep {yr['tree']['sep']:+.3f}/{yr['tabpfn']['sep']:+.3f}")
        for budget in BUDGETS:
            for model in ("tree", "tabpfn"):
                e = d["stats"][f"{budget}|{model}"]
                q = lambda k, s=False: ("  n/a" if e[k] is None
                                        else (f"{e[k]:+.3f}" if s else f"{e[k]:.3f}"))
                print(f"   K={budget} {model:6}: anom-0 {q('zero_anom')} | norm-0 {q('zero_norm')}"
                      f" | sep {q('sep', True)} | nz-norm {q('nz_norm')} | sd {e['seed_sd']:.3f}")

    # ---- timestamps
    import pandas as pd
    want = {d["bid"] for d in out.values()}
    print(f"\ntimestamps for {sorted(want)} ...", flush=True)
    parts = []
    for ch in pd.read_csv(RAW, usecols=["building_id", "meter", "timestamp", "meter_reading"],
                          chunksize=2_000_000):
        ch = ch[(ch["meter"] == 3) & (ch["building_id"].isin(want))]
        if len(ch):
            parts.append(ch)
    raw = pd.concat(parts).sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    for name, d in out.items():
        g = raw[raw["building_id"] == d["bid"]]
        ts = g["timestamp"].to_numpy()
        rv = g["meter_reading"].to_numpy(dtype=float)
        hold = np.asarray(reading[bid == d["bid"]], dtype=float)
        ok = rv.size == hold.size and np.allclose(rv, hold, atol=1e-6, equal_nan=True)
        d["ts_aligned"] = bool(ok)
        if not ok:
            print(f"   [{name}] b{d['bid']}: raw {rv.size} vs holdout {hold.size} -- NOT aligned")
            d["t0"] = d["t1"] = None
            continue
        w = pd.to_datetime(pd.Series(ts[d["start"]:d["start"] + d["n"]]))
        day0 = w.iloc[0].normalize()
        d["t0"], d["t1"] = str(w.iloc[0]), str(w.iloc[-1])
        d["day"] = ((w.dt.normalize() - day0).dt.days).tolist()
        d["hh"] = w.dt.hour.tolist()
        d["day_labels"] = [str(day0.date() + pd.Timedelta(days=i)) for i in range(max(d["day"]) + 1)]
        aa = np.asarray(d["a"])
        d["flips"] = [{"i": int(i), "t": str(w.iloc[int(i)]), "to": int(aa[int(i)])}
                      for i in (np.flatnonzero(np.diff(aa) != 0) + 1)]
        print(f"   [{name}] b{d['bid']}: aligned -> {d['t0']} .. {d['t1']}"
              f"  ({len(d['flips'])} 次標籤翻轉)")

    OUT.write_text(json.dumps(out, separators=(",", ":"), default=int), encoding="utf-8")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
