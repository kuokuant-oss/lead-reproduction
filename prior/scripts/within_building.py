"""Confound test + structure of the reading==0 subset.

1) Per-building composition of the reading==0 subset: does each building have a
   MIX of anomalous and normal zeros, or is "zero" all-anomaly in some buildings
   and all-normal in others?  If the latter, a high ROC-AUC on the reading==0
   subset could be achieved by identifying the building, not the hour.
2) Stratified (within-building) ROC-AUC, averaged over buildings weighted by
   pairs, vs the pooled subset ROC-AUC.  The difference isolates how much of the
   subset ranking is between-building vs within-building.
3) Row order / contiguity check so run-length structure can be described.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k")
BASE_D = REPO / "data/processed/m5_building_curve"
FORMAL = BASE_D / "v5_fixed_50k/model_runs"
EXT = BASE_D / "v5_fixed_50k_k725_row_seed_extension/model_runs"
COLAB = BASE_D / "v5_fixed_50k_k725_colab/model_results"
SIDE = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
            "/analysis_2026-09-03/data/hotwater_sideinfo.npz")
OUT = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
           "/analysis_2026-09-08/data/within_building.json")
SCORE_KEY = {"tree": "ensemble", "tabpfn": "tabpfn"}


def k725_path(model, r):
    if model == "tree":
        root = FORMAL if r < 2 else EXT
        return root / "building_seed725" / f"row_seed{r}" / "tree_no_es_k725_f137" / "predictions.npz"
    return (FORMAL / "building_seed725" / f"row_seed{r}" / "tabpfn_k725_f137" / "predictions.npz"
            if r < 2 else COLAB / f"row_seed{r}" / "predictions.npz")


def sc_path(model, b, r, k):
    sub = "tree_no_es" if model == "tree" else "tabpfn"
    return FORMAL / f"building_seed{b}" / f"row_seed{r}" / f"{sub}_k{k}_f137" / "predictions.npz"


def load_hot(path, key, y_side):
    with np.load(path) as p:
        hot = np.asarray(p["meter"]).astype(np.int64) == 3
        anom = np.asarray(p["anomaly"]).astype(np.int64)[hot]
        s = np.asarray(p[key]).astype(np.float64)[hot]
    assert np.array_equal(anom, y_side), path
    return s


def strat_auc(y, s, grp):
    """Within-group ROC-AUC, pooled over groups by pair count (Mann-Whitney form).
    Equivalent to P(score_pos > score_neg | same group), ties counted 0.5."""
    num = 0.0; den = 0.0; used = 0
    for g in np.unique(grp):
        m = grp == g
        yg, sg = y[m], s[m]
        npos = int((yg == 1).sum()); nneg = int((yg == 0).sum())
        if npos == 0 or nneg == 0:
            continue
        a = roc_auc_score(yg, sg)      # = concordant pair fraction with 0.5 ties
        pairs = npos * nneg
        num += a * pairs; den += pairs; used += 1
    return (num / den if den else float("nan")), used, int(den)


def main():
    with np.load(SIDE, allow_pickle=True) as z:
        y = np.asarray(z["y"]).astype(np.int64)
        reading = np.asarray(z["reading"]).astype(np.float64)
        bid = np.asarray(z["bid"]).astype(np.int64)
        hour = np.asarray(z["hour"]).astype(np.int64)
    m0 = reading == 0.0
    mN = ~m0
    out = {}

    # ---------------------------------------------------- 1) composition per building
    bs = np.unique(bid)
    comp = []
    for b in bs:
        mb = bid == b
        n = int(mb.sum()); z0 = int((mb & m0).sum())
        a0 = int((mb & m0 & (y == 1)).sum()); n0 = z0 - a0
        zN = int((mb & mN).sum()); aN = int((mb & mN & (y == 1)).sum())
        comp.append({"bid": int(b), "n": n, "n_r0": z0, "r0_frac": z0 / n,
                     "r0_anom": a0, "r0_norm": n0,
                     "r0_anom_frac": (a0 / z0) if z0 else None,
                     "n_rN": zN, "rN_anom": aN, "rN_anom_frac": (aN / zN) if zN else None})
    out["per_building"] = comp
    fr = np.array([c["r0_anom_frac"] for c in comp if c["r0_anom_frac"] is not None])
    print(f"buildings={len(comp)}  rows/building: min={min(c['n'] for c in comp)} max={max(c['n'] for c in comp)}")
    print(f"\n1) COMPOSITION OF THE reading==0 SUBSET, PER BUILDING")
    print(f"   fraction of a building's ZERO rows that are labelled anomalous:")
    print(f"     min={fr.min():.3f} p25={np.quantile(fr,.25):.3f} median={np.median(fr):.3f}"
          f" p75={np.quantile(fr,.75):.3f} max={fr.max():.3f}")
    print(f"     buildings with 0% anomalous zeros : {int((fr==0).sum())}")
    print(f"     buildings with 100% anomalous zeros: {int((fr==1).sum())}")
    print(f"     buildings with a MIX (>0 and <1)  : {int(((fr>0)&(fr<1)).sum())}")
    npair0 = sum(c["r0_anom"] * c["r0_norm"] for c in comp)
    tot0 = int((m0 & (y == 1)).sum()) * int((m0 & (y == 0)).sum())
    print(f"   within-building anomaly/normal ZERO pairs: {npair0:,} of {tot0:,} total pairs"
          f" ({npair0/tot0:.1%}) -> {1-npair0/tot0:.1%} of the pooled r0 AUC pairs are CROSS-building")
    out["r0_within_pair_share"] = npair0 / tot0

    # ---------------------------------------------------- 3) row order check
    b0 = bs[0]
    h = hour[bid == b0][:50]
    print(f"\n3) ROW ORDER: first 50 `hour` values for building {b0}: {h.tolist()}")
    consec = np.mean(np.diff(hour[bid == b0]) % 24 == 1)
    print(f"   fraction of consecutive rows advancing exactly +1 hour: {consec:.4f}"
          f"  -> rows {'ARE' if consec > 0.99 else 'are NOT'} in time order within a building")
    out["time_ordered_frac"] = float(consec)

    # ---------------------------------------------------- 2) stratified AUC
    print(f"\n2) POOLED vs WITHIN-BUILDING (stratified) ROC-AUC")
    print(f"{'K':>4} {'model':7} | {'r0 pooled':>10} {'r0 within':>10} {'delta':>8}"
          f" | {'rN pooled':>10} {'rN within':>10} {'delta':>8}")
    print("-" * 82)
    rows = []
    for k in (50, 100, 200, 400, 725):
        for model in ("tree", "tabpfn"):
            paths = ([sc_path(model, b, r, k) for b in range(5) for r in range(2)]
                     if k != 725 else [k725_path(model, r) for r in range(5)])
            p0 = []; w0 = []; pN = []; wN = []
            for pth in paths:
                s = load_hot(pth, SCORE_KEY[model], y)
                p0.append(roc_auc_score(y[m0], s[m0]))
                w0.append(strat_auc(y[m0], s[m0], bid[m0])[0])
                pN.append(roc_auc_score(y[mN], s[mN]))
                wN.append(strat_auc(y[mN], s[mN], bid[mN])[0])
            r = {"budget": k, "model": model,
                 "r0_pooled": float(np.mean(p0)), "r0_within": float(np.mean(w0)),
                 "rN_pooled": float(np.mean(pN)), "rN_within": float(np.mean(wN))}
            rows.append(r)
            print(f"{k:>4} {model:7} | {r['r0_pooled']:10.4f} {r['r0_within']:10.4f}"
                  f" {r['r0_within']-r['r0_pooled']:+8.4f} | {r['rN_pooled']:10.4f}"
                  f" {r['rN_within']:10.4f} {r['rN_within']-r['rN_pooled']:+8.4f}", flush=True)
    out["stratified"] = rows
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
