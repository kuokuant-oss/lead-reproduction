"""Pick and dump one-month (720 h) hourly windows that show what a reading==0
anomaly looks like, next to what a legitimate zero looks like.

Cases wanted:
  A. a building whose SAME month contains both anomalous zeros and normal zeros
     -- that is the discrimination the model actually faces
  B. a building whose zeros are entirely legitimate (0% of its zeros anomalous)
     -- the negative control: this is what "the plant is just off" looks like
  C. a clean transition from normal operation into a multi-week zero outage

Each window also carries the Tree K=725 score per hour (mean over the 5 row
seeds), so the label and the model's reaction can be read side by side.
Rows are strictly hourly and time-ordered within a building (verified).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k")
BD = REPO / "data/processed/m5_building_curve"
FORMAL = BD / "v5_fixed_50k/model_runs"
EXT = BD / "v5_fixed_50k_k725_row_seed_extension/model_runs"
COLAB = BD / "v5_fixed_50k_k725_colab/model_results"
SIDE = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
            "/analysis_2026-09-03/data/hotwater_sideinfo.npz")
OUT = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
           "/analysis_2026-09-08/data/month_cases.json")
W = 720  # 30 days


BUDGET = 400          # the budget the figures plot
BUDGETS = (400, 725)  # both are summarised in the table
SK = {"tree": "ensemble", "tabpfn": "tabpfn"}


def cell_paths(model, budget):
    """K<=400: 5 building seeds x 2 row seeds -- a plain mean over the 10 cells equals
    the published nested mean (mean over row seeds within a building seed, then across
    building seeds) because every building seed has exactly 2 row seeds.
    K=725: 5 row seeds, Tree r2-4 from the extension dir."""
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
    m0 = reading == 0.0
    SCORE, SPREAD = {}, {}
    for budget in BUDGETS:
        for model in ("tree", "tabpfn"):
            ps = cell_paths(model, budget)
            print(f"averaging {model} K={budget} over {len(ps)} cells ...", flush=True)
            stack = np.stack([load_hot(p, SK[model], y) for p in ps])
            SCORE[budget, model] = stack.mean(axis=0)
            SPREAD[budget, model] = stack.std(axis=0, ddof=1)
    score = SCORE[BUDGET, "tree"]   # used for the diagnostics printed below

    bl = np.unique(bid)
    # ---- per-building summary, to choose cases
    summary = []
    for b in bl:
        mb = bid == b
        z0 = int((mb & m0).sum())
        a0 = int((mb & m0 & (y == 1)).sum())
        summary.append({"bid": int(b), "use": str(use[mb][0]), "n": int(mb.sum()),
                        "zeros": z0, "zeros_anom": a0, "zeros_norm": z0 - a0,
                        "anom": int((mb & (y == 1)).sum())})

    def best_window(b, key):
        """slide a 720h window over building b, score it with `key`, return best start."""
        mb = bid == b
        v = reading[mb]
        yb = y[mb]
        z = (v == 0.0)
        za = z & (yb == 1)
        zn = z & (yb == 0)
        cza = np.r_[0, np.cumsum(za)]
        czn = np.r_[0, np.cumsum(zn)]
        n = v.size
        if n <= W:
            return None
        st = np.arange(0, n - W)
        na = cza[st + W] - cza[st]
        nn = czn[st + W] - czn[st]
        sc = key(na, nn)
        i = int(np.argmax(sc))
        return int(st[i]), int(na[i]), int(nn[i]), float(sc[i])

    cases = {}

    # ---- A: same month holds both kinds of zero, as balanced as possible
    best = None
    for s in summary:
        if s["zeros_anom"] < 24 or s["zeros_norm"] < 24:
            continue
        r = best_window(s["bid"], lambda na, nn: np.minimum(na, nn))
        if r and (best is None or r[3] > best[1][3]):
            best = (s, r)
    cases["mixed"] = (best[0], best[1])

    # ---- B: zeros entirely legitimate; take the window with the most zeros
    cand = [s for s in summary if s["zeros_anom"] == 0 and s["zeros"] >= 100]
    cand.sort(key=lambda s: -s["zeros"])
    s = cand[0]
    r = best_window(s["bid"], lambda na, nn: nn)
    cases["legit"] = (s, r)

    # ---- C: clean transition into a long outage (normal before, zeros after)
    best = None
    for s in summary:
        b = s["bid"]
        mb = bid == b
        v = reading[mb]
        yb = y[mb]
        st, ln = runs(v == 0.0)
        for a, L in zip(st, ln):
            if L < 240 or a < 240:
                continue
            if yb[a:a + L].mean() < 0.99:
                continue
            pre = v[a - 240:a]
            if (pre == 0).mean() > 0.2 or pre.max() <= 0:
                continue
            sc = min(L, 480) * float(pre.mean())
            if best is None or sc > best[2]:
                best = (s, a, sc, int(L))
    s, a, _, L = best
    cases["outage"] = (s, (max(0, a - 240), 0, 0, 0.0), L)

    print("\nchosen cases")
    out = {}
    for name, tup in cases.items():
        s = tup[0]
        start = tup[1][0]
        mb = bid == s["bid"]
        # snap the window start back to a midnight row so a 24x30 grid is real days
        hb = hour[mb]
        while start > 0 and hb[start] != 0:
            start -= 1
        v = reading[mb][start:start + W]
        yb = y[mb][start:start + W]
        sc = score[mb][start:start + W]
        hr = hour[mb][start:start + W]
        z = v == 0.0
        st, ln = runs(z)
        za = [(int(a), int(l), int(yb[a:a + l].sum())) for a, l in zip(st, ln)]
        out[name] = {
            "bid": s["bid"], "use": s["use"], "start": start, "n": int(v.size),
            "hour0": int(hr[0]), "budget": BUDGET,
            "st": [round(float(x), 4) for x in SCORE[BUDGET, "tree"][mb][start:start + W]],
            "sp": [round(float(x), 4) for x in SCORE[BUDGET, "tabpfn"][mb][start:start + W]],
            "building_zeros": s["zeros"], "building_zeros_anom": s["zeros_anom"],
            "building_zeros_norm": s["zeros_norm"], "building_anom": s["anom"],
            "building_n": s["n"],
            "win_zeros": int(z.sum()),
            "win_zeros_anom": int((z & (yb == 1)).sum()),
            "win_zeros_norm": int((z & (yb == 0)).sum()),
            "win_anom": int((yb == 1).sum()),
            "win_nonzero_anom": int((~z & (yb == 1)).sum()),
            "zero_runs": za,
            "v": [round(float(x), 2) for x in v],
            "a": yb.astype(int).tolist(),
            "s": [round(float(x), 4) for x in sc],
        }
        d = out[name]
        print(f"\n[{name}] building {d['bid']} ({d['use']})  window start h={start}, hour0={d['hour0']}")
        print(f"   whole year : {d['building_n']} h, zeros {d['building_zeros']} "
              f"(anom {d['building_zeros_anom']} / norm {d['building_zeros_norm']}), "
              f"anomalies {d['building_anom']}")
        print(f"   this month : zeros {d['win_zeros']} (anom {d['win_zeros_anom']} / "
              f"norm {d['win_zeros_norm']}), non-zero anomalies {d['win_nonzero_anom']}, "
              f"total anomalous hours {d['win_anom']}")
        nz = [x for x in v if x > 0]
        if nz:
            print(f"   non-zero readings: min {min(nz):.1f}  median {sorted(nz)[len(nz)//2]:.1f}"
                  f"  max {max(nz):.1f}")
        print(f"   zero-runs in month ({len(za)}): "
              + ", ".join(f"h{a}+{l}h[{'A' if na == l else ('N' if na == 0 else 'MIX')}]"
                          for a, l, na in za[:14]) + (" ..." if len(za) > 14 else ""))
        # category means for BOTH budgets and BOTH models, for the table
        CAT = {"zero_anom": z & (yb == 1), "zero_norm": z & (yb == 0),
               "nz_norm": (~z) & (yb == 0), "nz_anom": (~z) & (yb == 1)}
        out[name]["stats"] = {}
        for budget in BUDGETS:
            for model in ("tree", "tabpfn"):
                sm = SCORE[budget, model][mb][start:start + W]
                sd = SPREAD[budget, model][mb][start:start + W]
                e = {k: (float(np.mean(sm[sel])) if sel.any() else None)
                     for k, sel in CAT.items()}
                e["sep"] = (None if e["zero_anom"] is None or e["zero_norm"] is None
                            else e["zero_anom"] - e["zero_norm"])
                e["seed_sd"] = float(np.mean(sd))
                e["seed_sd_zero"] = float(np.mean(sd[z])) if z.any() else None
                out[name]["stats"][f"{budget}|{model}"] = e
                q = lambda k, sign=False: ("  n/a" if e[k] is None
                                           else (f"{e[k]:+.3f}" if sign else f"{e[k]:.3f}"))
                print(f"   K={budget} {model:6}: anom-zero {q('zero_anom')}"
                      f" | norm-zero {q('zero_norm')}"
                      f" | separation {q('sep', True)}"
                      f" | nz-norm {q('nz_norm')}"
                      f" | seed SD {e['seed_sd']:.3f}")

    # ---- put real timestamps on the windows, from raw train.csv
    import pandas as pd
    want = {d["bid"] for d in out.values()}
    RAW = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/train.csv")
    print(f"\nreading timestamps for buildings {sorted(want)} from {RAW.name} ...", flush=True)
    parts = []
    for ch in pd.read_csv(RAW, usecols=["building_id", "meter", "timestamp", "meter_reading"],
                          chunksize=2_000_000):
        ch = ch[(ch["meter"] == 3) & (ch["building_id"].isin(want))]
        if len(ch):
            parts.append(ch)
    raw = pd.concat(parts).sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    for name, d in out.items():
        b = d["bid"]
        g = raw[raw["building_id"] == b]
        ts = g["timestamp"].to_numpy()
        rv = g["meter_reading"].to_numpy(dtype=float)
        hold = np.asarray(reading[bid == b], dtype=float)
        ok = rv.size == hold.size and np.allclose(rv, hold, rtol=0, atol=1e-6, equal_nan=True)
        d["ts_aligned"] = bool(ok)
        if ok:
            w = pd.to_datetime(pd.Series(ts[d["start"]:d["start"] + d["n"]]))
            d["t0"] = str(w.iloc[0]); d["t1"] = str(w.iloc[-1])
            day0 = w.iloc[0].normalize()
            d["day"] = ((w.dt.normalize() - day0).dt.days).tolist()   # 0..29
            d["hh"] = w.dt.hour.tolist()                              # 0..23
            d["day_labels"] = [str(day0.date() + pd.Timedelta(days=i))
                               for i in range(max(d["day"]) + 1)]
            # where the label flips, as a real timestamp
            a = np.asarray(d["a"])
            flips = np.flatnonzero(np.diff(a) != 0) + 1
            d["flips"] = [{"i": int(i), "t": str(w.iloc[int(i)]), "to": int(a[int(i)])}
                          for i in flips]
            print(f"   [{name}] b{b}: raw rows {rv.size} == holdout {hold.size}, readings match"
                  f" -> {d['t0']} .. {d['t1']}  ({max(d['day'])+1} days)")
            for fl in d["flips"]:
                print(f"        label -> {'ANOMALOUS' if fl['to'] else 'normal'} at {fl['t']}")
        else:
            d["t0"] = d["t1"] = None
            print(f"   [{name}] b{b}: raw {rv.size} vs holdout {hold.size}"
                  f" -- NOT aligned, no timestamps emitted")

    OUT.write_text(json.dumps(out, separators=(",", ":"), default=int), encoding="utf-8")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
