"""What the literature would use instead of a raw F1, computed on this data.

(a) The ratio I used ("x trivial") and its problem: its own CEILING is 1/F1_0,
    which itself moves with prevalence -- so 9.3x vs 1.2x is not a like-for-like
    comparison either. The skill-score form fixes that:
        SS = (F1 - F1_0) / (1 - F1_0),   F1_0 = 2*pi/(1+pi)
    bounded above by 1 at every prevalence.
(b) MCC (Matthews correlation coefficient) at 0.50 and at its own best threshold.
    0 = random at ANY prevalence, uses all four confusion cells (F1 ignores TN).
(c) Average precision against its exact random baseline (AP_random = pi), and the
    prevalence-free rescaling AP_gain = (AP - pi)/(1 - pi).
(d) Calibration: the context is 25k/25k balanced but evaluation prevalence is
    14.26% (2.12% on the non-zero half). If that prior shift is what pushes the
    F1-optimal threshold above 0.50, the scores must be over-confident. Measured
    as mean predicted vs observed rate per score bin, plus ECE.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

REPO = Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k")
BD = REPO / "data/processed/m5_building_curve"
FORMAL = BD / "v5_fixed_50k/model_runs"
EXT = BD / "v5_fixed_50k_k725_row_seed_extension/model_runs"
COLAB = BD / "v5_fixed_50k_k725_colab/model_results"
SIDE = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
            "/analysis_2026-09-03/data/hotwater_sideinfo.npz")
OUT = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
           "/analysis_2026-09-08/data/f1_alternatives.json")
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


def cum(y, s):
    """cumulative TP/FP by descending score, evaluated at tie-group ends."""
    o = np.argsort(-s, kind="stable")
    ys, ss = y[o], s[o]
    tp = np.cumsum(ys == 1)
    k = np.arange(1, ys.size + 1)
    ends = np.r_[np.flatnonzero(np.diff(ss) != 0), ys.size - 1]
    return tp[ends], k[ends], ss[ends], int((y == 1).sum()), ys.size


def best_f1(y, s):
    tp, k, thr, npos, n = cum(y, s)
    f1 = 2 * tp / (k + npos)
    i = int(np.argmax(f1))
    return float(f1[i]), float(thr[i])


def mcc_from(tp, fp, fn, tn):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn - fp * fn) / d) if d > 0 else 0.0


def mcc_at(y, s, t=0.5):
    pred = s >= t
    tp = int(np.count_nonzero(pred & (y == 1))); fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1))); tn = int(np.count_nonzero(~pred & (y == 0)))
    return mcc_from(tp, fp, fn, tn)


def best_mcc(y, s):
    tp, k, thr, npos, n = cum(y, s)
    fp = k - tp
    fn = npos - tp
    tn = n - npos - fp
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp).astype(float) * (tp + fn) * (tn + fp) * (tn + fn))
    m = np.divide(num, den, out=np.zeros_like(den), where=den > 0)
    i = int(np.argmax(m))
    return float(m[i]), float(thr[i])


def mean(v): return sum(v) / len(v)
def se(v):
    if len(v) < 2: return None
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) / math.sqrt(len(v))


def main():
    with np.load(SIDE, allow_pickle=True) as z:
        y = np.asarray(z["y"]).astype(np.int64)
        reading = np.asarray(z["reading"]).astype(np.float64)
        bid = np.asarray(z["bid"]).astype(np.int64)
    m0 = reading == 0.0
    mN = ~m0
    SUBS = [("pooled", np.ones_like(m0)), ("r0", m0), ("rN", mN), ("rN_no1241", mN & (bid != 1241))]

    print("loading scores ...", flush=True)
    SC = {}
    for k in K:
        for model in ("tree", "tabpfn"):
            for g, ps in groups(model, k).items():
                SC[(k, model, g)] = [load_hot(p, SK[model], y) for p in ps]

    res = {"subsets": {}, "calib": {}}
    for name, msk in SUBS:
        yv = y[msk]
        pi = float((yv == 1).mean())
        f1_0 = 2 * pi / (1 + pi)
        res["subsets"][name] = {"pi": pi, "f1_trivial": f1_0,
                                "ratio_ceiling": 1 / f1_0, "rows": {}}
        print("\n" + "=" * 112)
        print(f"{name}   prevalence = {pi:.4%}   F1_trivial = {f1_0:.4f}"
              f"   ratio ceiling = {1/f1_0:.1f}x")
        print("=" * 112)
        print(f"{'K':>4} {'model':7} | {'maxF1':>7} {'xTriv':>6} {'F1 skill':>9} |"
              f" {'MCC@.5':>7} {'maxMCC':>7} {'thr':>6} | {'AP':>7} {'AP gain':>8} {'AP/pi':>7}")
        for k in K:
            for model in ("tree", "tabpfn"):
                mf, mt, m05, mm, mmt, ap = [], [], [], [], [], []
                for g in groups(model, k):
                    for s in SC[(k, model, g)]:
                        sv = s[msk]
                        a, at = best_f1(yv, sv); mf.append(a); mt.append(at)
                        m05.append(mcc_at(yv, sv))
                        b, bt = best_mcc(yv, sv); mm.append(b); mmt.append(bt)
                        ap.append(average_precision_score(yv, sv))
                r = {"max_f1": mean(mf), "x_trivial": mean(mf) / f1_0,
                     "f1_skill": (mean(mf) - f1_0) / (1 - f1_0),
                     "mcc_05": mean(m05), "mcc_05_se": se(m05),
                     "max_mcc": mean(mm), "max_mcc_se": se(mm), "mcc_thr": mean(mmt),
                     "ap": mean(ap), "ap_gain": (mean(ap) - pi) / (1 - pi),
                     "ap_over_pi": mean(ap) / pi, "f1_thr": mean(mt)}
                res["subsets"][name]["rows"][f"{k}|{model}"] = r
                print(f"{k:>4} {model:7} | {r['max_f1']:7.3f} {r['x_trivial']:5.1f}x"
                      f" {r['f1_skill']:9.3f} | {r['mcc_05']:7.3f} {r['max_mcc']:7.3f}"
                      f" {r['mcc_thr']:6.2f} | {r['ap']:7.3f} {r['ap_gain']:8.3f}"
                      f" {r['ap_over_pi']:6.1f}x")

    # ---------------- calibration, K=725 -------------------------------------
    print("\n" + "=" * 112)
    print("CALIBRATION at K=725 (context is 25,000 anomaly / 25,000 normal = 50% prior)")
    print("  a well-calibrated score would sit on the diagonal: mean predicted == observed rate")
    print("=" * 112)
    EDGES = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0000001])
    for name, msk in [("pooled", np.ones_like(m0)), ("rN", mN)]:
        yv = y[msk]
        pi = float((yv == 1).mean())
        for model in ("tree", "tabpfn"):
            ss = [s[msk] for g in groups(model, 725) for s in SC[(725, model, g)]]
            print(f"\n--- {name} · {model} · prevalence {pi:.3%} ---")
            print(f"{'bin':>10} {'rows':>10} {'mean pred':>10} {'observed':>10} {'diff':>8}")
            ece = 0.0
            rows = []
            for i in range(len(EDGES) - 1):
                lo, hi = EDGES[i], EDGES[i+1]
                fr, mp, ob = [], [], []
                for sv in ss:
                    sel = (sv >= lo) & (sv < hi)
                    if sel.sum() == 0:
                        continue
                    fr.append(sel.mean()); mp.append(sv[sel].mean()); ob.append((yv[sel] == 1).mean())
                if not fr:
                    continue
                fr_, mp_, ob_ = mean(fr), mean(mp), mean(ob)
                ece += fr_ * abs(mp_ - ob_)
                rows.append({"lo": float(lo), "frac": fr_, "pred": mp_, "obs": ob_})
                print(f"{lo:.1f}–{min(hi,1):.1f} {fr_*len(yv):>10,.0f} {mp_:>10.3f}"
                      f" {ob_:>10.3f} {mp_-ob_:>+8.3f}")
            allmean = mean([sv.mean() for sv in ss])
            print(f"   ECE = {ece:.4f}   mean score over the subset = {allmean:.4f}"
                  f"   actual rate = {pi:.4f}   over-confidence = {allmean-pi:+.4f}")
            # the prior-shift equivalent of a 0.5 posterior
            print(f"   a 50%-prior score of 0.50 maps to posterior {pi/(pi+(1-pi)):.3f} "
                  f"under a {pi:.3%} prior; the raw score matching a 0.50 posterior is {1-pi:.3f}")
            res["calib"][f"{name}|{model}"] = {"pi": pi, "ece": ece, "mean_score": allmean,
                                               "bins": rows}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
