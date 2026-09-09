"""Is F1 ~= 0.3 on the reading != 0 subset good, given 2.12% prevalence?

Three reference points per cell, on each subset:
  1. trivial ceiling  - the best F1 any prevalence-blind rule can get.
     Flagging a fraction r at random gives precision = pi, recall = r, so
     F1 = 2*pi*r/(pi+r), increasing in r -> maximised by flagging EVERYTHING:
     F1_trivial = 2*pi/(1+pi).  No ranking skill, so this is the floor to beat.
  2. F1 @ 0.50        - the reported operating point.
  3. max-F1 over all thresholds - the best this model's own ranking allows.
     At a cutoff of k rows, F1 = 2*TP(k)/(k + Npos); scanned over tie-group ends
     so every candidate is a realisable threshold. Says whether 0.50 is leaving
     anything on the table, separately from whether the ranking is any good.
Repeated with building 1241 removed, since it holds 52.4% of the non-zero anomalies.
"""
from __future__ import annotations
import json, math
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
           "/analysis_2026-09-08/data/f1_context.json")
SK = {"tree": "ensemble", "tabpfn": "tabpfn"}
K = (50, 100, 200, 400, 725)


def groups(model, k):
    sub = "tree_no_es" if model == "tree" else "tabpfn"
    d = {}
    if k != 725:
        for b in range(5):
            d[f"b{b}"] = [FORMAL / f"building_seed{b}" / f"row_seed{r}" / f"{sub}_k{k}_f137"
                          / "predictions.npz" for r in range(2)]
    else:
        for r in range(5):
            if model == "tree":
                root = FORMAL if r < 2 else EXT
                p = root / "building_seed725" / f"row_seed{r}" / "tree_no_es_k725_f137" / "predictions.npz"
            else:
                p = (FORMAL / "building_seed725" / f"row_seed{r}" / "tabpfn_k725_f137" / "predictions.npz"
                     if r < 2 else COLAB / f"row_seed{r}" / "predictions.npz")
            d[f"r{r}"] = [p]
    return d


def load_hot(path, key, y_side):
    with np.load(path) as p:
        hot = np.asarray(p["meter"]).astype(np.int64) == 3
        anom = np.asarray(p["anomaly"]).astype(np.int64)[hot]
        s = np.asarray(p[key]).astype(np.float64)[hot]
    assert np.array_equal(anom, y_side), path
    return s


def f1_at(y, s, t=0.5):
    pred = s >= t
    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def max_f1(y, s):
    """Best F1 over all realisable thresholds. F1(k) = 2*TP(k)/(k + Npos)."""
    o = np.argsort(-s, kind="stable")
    ys, ss = y[o], s[o]
    tp = np.cumsum(ys == 1)
    kk = np.arange(1, ys.size + 1)
    ends = np.r_[np.flatnonzero(np.diff(ss) != 0), ys.size - 1]   # tie-group ends
    npos = int((y == 1).sum())
    f1 = 2 * tp[ends] / (kk[ends] + npos)
    i = int(np.argmax(f1))
    k_at = int(kk[ends][i])
    tp_at = int(tp[ends][i])
    return (float(f1[i]), tp_at / k_at, tp_at / npos, float(ss[ends][i]), k_at / ys.size)


def mean(v): return sum(v) / len(v)
def se(v):
    if len(v) < 2: return None
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) / math.sqrt(len(v))
def pm(v, d=3):
    s = se(v)
    return f"{mean(v):.{d}f}" + (f"±{s:.{d}f}" if s is not None else "")


def main():
    with np.load(SIDE, allow_pickle=True) as z:
        y = np.asarray(z["y"]).astype(np.int64)
        reading = np.asarray(z["reading"]).astype(np.float64)
        bid = np.asarray(z["bid"]).astype(np.int64)
    m0 = reading == 0.0
    mN = ~m0
    keep = bid != 1241

    SUBSETS = [("reading != 0  (all 73 buildings)", mN),
               ("reading != 0  (excluding 1241)", mN & keep),
               ("reading == 0  (for contrast)", m0),
               ("pooled hot water", np.ones_like(m0))]

    print("TRIVIAL CEILING  -- best F1 with no ranking skill at all")
    print("  flag every row: precision = prevalence, recall = 1, F1 = 2*pi/(1+pi)")
    print(f"{'subset':<36} {'n':>9} {'pos':>8} {'prevalence':>11} {'F1 trivial':>11}")
    triv = {}
    for name, msk in SUBSETS:
        n = int(msk.sum()); npos = int((y[msk] == 1).sum()); pi = npos / n
        triv[name] = 2 * pi / (1 + pi)
        print(f"{name:<36} {n:>9,} {npos:>8,} {pi:>10.3%} {triv[name]:>11.4f}")

    print("\nloading scores ...", flush=True)
    SC = {}
    for k in K:
        for model in ("tree", "tabpfn"):
            for g, ps in groups(model, k).items():
                SC[(k, model, g)] = [load_hot(p, SK[model], y) for p in ps]

    res = {"trivial": triv, "rows": []}
    for name, msk in SUBSETS:
        yv = y[msk]
        print("\n" + "=" * 104)
        print(f"{name}   (trivial-ceiling F1 = {triv[name]:.4f})")
        print("=" * 104)
        print(f"{'K':>4} {'model':7} | {'F1@0.50':>14} {'max-F1':>14} {'headroom':>9} |"
              f" {'P@maxF1':>13} {'R@maxF1':>13} {'thr':>7} {'%flagged':>9} | {'xTrivial':>9}")
        for k in K:
            for model in ("tree", "tabpfn"):
                f05, mx, pmx, rmx, thr, fr = [], [], [], [], [], []
                for g, ps in groups(model, k).items():
                    for s in SC[(k, model, g)]:
                        sv = s[msk]
                        a, _, _ = f1_at(yv, sv)
                        b, pb, rb, tb, frb = max_f1(yv, sv)
                        f05.append(a); mx.append(b); pmx.append(pb); rmx.append(rb)
                        thr.append(tb); fr.append(frb)
                row = {"subset": name, "budget": k, "model": model,
                       "f1_05": mean(f05), "f1_05_se": se(f05),
                       "max_f1": mean(mx), "max_f1_se": se(mx),
                       "p_at_max": mean(pmx), "r_at_max": mean(rmx),
                       "thr_at_max": mean(thr), "frac_at_max": mean(fr),
                       "x_trivial": mean(f05) / triv[name]}
                res["rows"].append(row)
                print(f"{k:>4} {model:7} | {pm(f05):>14} {pm(mx):>14}"
                      f" {mean(mx)-mean(f05):>+9.3f} | {mean(pmx):>13.3f} {mean(rmx):>13.3f}"
                      f" {mean(thr):>7.3f} {mean(fr):>9.2%} | {mean(f05)/triv[name]:>8.1f}x")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
