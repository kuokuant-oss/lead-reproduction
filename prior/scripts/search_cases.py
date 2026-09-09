"""Re-pick the case studies, this time ON MODEL DISAGREEMENT rather than on data
shape alone -- the previous picks were chosen for what the readings looked like,
so the two models unsurprisingly behaved alike on them.

Selection metric: within-building ROC-AUC on that building's reading==0 rows
(anomalous zeros vs legitimate zeros), Tree minus TabPFN, at K=400. That is
exactly the discrimination the case figures are meant to illustrate.
Requires >= 48 of each kind of zero so the AUC means something.

Also looks for a "textbook" outage: a zero-run of 3-10 days with non-zero
readings BOTH BEFORE AND AFTER inside one 30-day window, and a visible
hour-of-day profile (so the reader sees normal -> outage -> recovery).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k")
BD = REPO / "data/processed/m5_building_curve"
FORMAL = BD / "v5_fixed_50k/model_runs"
SIDE = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
            "/analysis_2026-09-03/data/hotwater_sideinfo.npz")
OUT = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
           "/analysis_2026-09-08/data/case_search.json")
BUDGET = 400
SK = {"tree": "ensemble", "tabpfn": "tabpfn"}
W = 720
MIN_EACH = 48


def cell_paths(model):
    sub = "tree_no_es" if model == "tree" else "tabpfn"
    return [FORMAL / f"building_seed{b}" / f"row_seed{r}" / f"{sub}_k{BUDGET}_f137"
            / "predictions.npz" for b in range(5) for r in range(2)]


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

    S = {}
    for model in ("tree", "tabpfn"):
        ps = cell_paths(model)
        print(f"averaging {model} K={BUDGET} over {len(ps)} cells ...", flush=True)
        S[model] = np.mean([load_hot(p, SK[model], y) for p in ps], axis=0)

    # ---------------------------------------------------------------- ranking
    rows = []
    for b in np.unique(bid):
        mb = (bid == b) & m0
        yb = y[mb]
        na, nn = int((yb == 1).sum()), int((yb == 0).sum())
        if na < MIN_EACH or nn < MIN_EACH:
            continue
        at = roc_auc_score(yb, S["tree"][mb])
        ap = roc_auc_score(yb, S["tabpfn"][mb])
        st, sp = S["tree"][mb], S["tabpfn"][mb]
        rows.append({"bid": int(b), "use": str(use[bid == b][0]),
                     "n_zero": na + nn, "zero_anom": na, "zero_norm": nn,
                     "auc_tree": float(at), "auc_tabpfn": float(ap), "gap": float(at - ap),
                     "sep_tree": float(st[yb == 1].mean() - st[yb == 0].mean()),
                     "sep_tabpfn": float(sp[yb == 1].mean() - sp[yb == 0].mean()),
                     "zn_tree": float(st[yb == 0].mean()), "zn_tabpfn": float(sp[yb == 0].mean()),
                     "za_tree": float(st[yb == 1].mean()), "za_tabpfn": float(sp[yb == 1].mean())})
    rows.sort(key=lambda r: -r["gap"])
    print(f"\nbuildings scorable on reading==0 (>= {MIN_EACH} of each kind): {len(rows)} of 73")
    print("\nWITHIN-BUILDING reading==0 ROC-AUC, Tree vs TabPFN, K=400")
    print(f"{'bid':>5} {'primary_use':<22} {'0-hrs':>6} {'anom':>6} {'norm':>6}"
          f" {'AUC T':>7} {'AUC P':>7} {'gap':>8} | {'sep T':>7} {'sep P':>7}")
    def show(rs):
        for r in rs:
            print(f"{r['bid']:>5} {r['use'][:22]:<22} {r['n_zero']:>6,} {r['zero_anom']:>6,}"
                  f" {r['zero_norm']:>6,} {r['auc_tree']:>7.3f} {r['auc_tabpfn']:>7.3f}"
                  f" {r['gap']:>+8.3f} | {r['sep_tree']:>+7.3f} {r['sep_tabpfn']:>+7.3f}")
    print("--- Tree ahead most ---")
    show(rows[:8])
    print("--- TabPFN ahead most ---")
    show(rows[-8:])
    g = np.array([r["gap"] for r in rows])
    print(f"\ngap distribution: min {g.min():+.3f}  p25 {np.quantile(g,.25):+.3f}"
          f"  median {np.median(g):+.3f}  p75 {np.quantile(g,.75):+.3f}  max {g.max():+.3f}")
    print(f"Tree ahead in {int((g>0).sum())} of {len(g)} scorable buildings")

    # -------------------------------------------- best window for a given building
    def best_window(b, need_both=True):
        mb = bid == b
        v, yb = reading[mb], y[mb]
        z = v == 0.0
        za, zn = z & (yb == 1), z & (yb == 0)
        cza, czn = np.r_[0, np.cumsum(za)], np.r_[0, np.cumsum(zn)]
        n = v.size
        if n <= W:
            return None
        st = np.arange(0, n - W)
        na, nn = cza[st + W] - cza[st], czn[st + W] - czn[st]
        sc = np.minimum(na, nn) if need_both else na
        i = int(np.argmax(sc))
        s0 = int(st[i])
        hb = hour[mb]
        while s0 > 0 and hb[s0] != 0:
            s0 -= 1
        return s0, int(na[i]), int(nn[i])

    picks = {}
    for tag, r in (("tree_wins", rows[0]), ("tabpfn_wins", rows[-1])):
        w = best_window(r["bid"])
        picks[tag] = {"bid": r["bid"], "start": w[0], "why": tag, **r}
        print(f"\n[{tag}] building {r['bid']} ({r['use']}): AUC {r['auc_tree']:.3f} vs "
              f"{r['auc_tabpfn']:.3f} (gap {r['gap']:+.3f}); window start h={w[0]}, "
              f"in-window zeros anom/norm = {w[1]}/{w[2]}")

    # -------------------------------- textbook outage: recovers inside the window
    print("\nsearching for a textbook outage (3-10 day zero-run, non-zero before AND after,"
          " visible hour-of-day profile) ...")
    best = None
    for b in np.unique(bid):
        mb = bid == b
        v, yb, hb = reading[mb], y[mb], hour[mb]
        st, ln = runs(v == 0.0)
        for a, L in zip(st, ln):
            if not (72 <= L <= 240):
                continue
            if yb[a:a + L].mean() < 0.95:
                continue
            pre_lo, post_hi = a - 240, a + L + 240
            if pre_lo < 0 or post_hi > v.size:
                continue
            pre, post = v[pre_lo:a], v[a + L:post_hi]
            if (pre == 0).mean() > 0.12 or (post == 0).mean() > 0.12:
                continue
            if pre.max() <= 0 or post.max() <= 0:
                continue
            # visible daily profile: variance explained by the hour-of-day mean
            ctx = np.r_[pre, post]
            hctx = np.r_[hb[pre_lo:a], hb[a + L:post_hi]]
            prof = np.array([ctx[hctx == h].mean() if (hctx == h).any() else ctx.mean()
                             for h in range(24)])
            resid = ctx - prof[hctx]
            r2 = 1 - resid.var() / ctx.var() if ctx.var() > 0 else 0
            score = r2 * min(L, 200)
            if best is None or score > best[0]:
                best = (score, int(b), int(a), int(L), float(r2), float(ctx.mean()))
    _, b, a, L, r2, lvl = best
    mb = bid == b
    hb = hour[mb]
    s0 = max(0, a - 240)
    while s0 > 0 and hb[s0] != 0:
        s0 -= 1
    picks["textbook"] = {"bid": b, "start": s0, "why": "textbook", "run_hours": L,
                         "daily_r2": r2, "level": lvl,
                         "use": str(use[mb][0])}
    print(f"[textbook] building {b} ({picks['textbook']['use']}): {L}h outage, "
          f"hour-of-day R2 {r2:.3f}, typical level {lvl:,.0f}; window start h={s0}")

    OUT.write_text(json.dumps({"ranking": rows, "picks": picks}, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
