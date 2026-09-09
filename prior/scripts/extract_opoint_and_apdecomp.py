"""Operating-point metrics + EXACT additive decomposition of pooled AP.

Decomposition (identity, not a heuristic):
  AP = sum_n (R_n - R_{n-1}) * P_n  ==  mean over positives of precision@its own
  score threshold, where precision@s = (#pos with score>=s)/(#rows with score>=s).
  Hence, with P0/P = share of positives that sit in reading==0,
      AP_pooled = (P0/P)*C0 + (PN/P)*CN ,  C_S = mean of that precision over the
  positives in subset S.  So the pooled AP gap between two models splits exactly:
      dAP = (P0/P)*dC0 + (PN/P)*dCN
  This attributes pooled-AP points to reading==0 positives vs reading!=0 positives.
  Verified against sklearn's average_precision_score per cell (assert < 1e-9).

Also per cell: threshold-0.5 confusion -> precision/recall/F1 for pooled/r0/rN,
score quantiles (threshold comparability), and R-precision (flag exactly as many
rows as there are true positives -- a threshold-free, model-comparable budget).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

REPO = Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k")
BASE_D = REPO / "data/processed/m5_building_curve"
FORMAL = BASE_D / "v5_fixed_50k/model_runs"
EXT = BASE_D / "v5_fixed_50k_k725_row_seed_extension/model_runs"
COLAB = BASE_D / "v5_fixed_50k_k725_colab/model_results"
SIDE = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
            "/analysis_2026-09-03/data/hotwater_sideinfo.npz")
OUT = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
           "/analysis_2026-09-08/data/opoint_apdecomp.json")
SCORE_KEY = {"tree": "ensemble", "tabpfn": "tabpfn"}
THRESH = 0.5


def cell_paths(budget, model):
    sub = "tree_no_es" if model == "tree" else "tabpfn"
    out = []
    if budget != 725:
        for b in range(5):
            for r in range(2):
                out.append((f"b{b}", f"r{r}", FORMAL / f"building_seed{b}" / f"row_seed{r}"
                            / f"{sub}_k{budget}_f137" / "predictions.npz"))
    else:
        for r in range(5):
            if model == "tree":
                root = FORMAL if r < 2 else EXT
                p = root / "building_seed725" / f"row_seed{r}" / "tree_no_es_k725_f137" / "predictions.npz"
            else:
                p = (FORMAL / "building_seed725" / f"row_seed{r}" / "tabpfn_k725_f137" / "predictions.npz"
                     if r < 2 else COLAB / f"row_seed{r}" / "predictions.npz")
            out.append((f"r{r}", f"r{r}", p))
    return out


def load_hot(path, key, y_side):
    with np.load(path) as p:
        hot = np.asarray(p["meter"]).astype(np.int64) == 3
        anom = np.asarray(p["anomaly"]).astype(np.int64)[hot]
        vri = np.asarray(p["validation_raw_index"]).astype(np.int64)[hot]
        s = np.asarray(p[key]).astype(np.float64)[hot]
    if not np.array_equal(anom, y_side):
        o = np.argsort(vri, kind="stable"); anom, s = anom[o], s[o]
        if not np.array_equal(anom, y_side):
            raise RuntimeError(f"align {path}")
    return s


def per_positive_precision(y, s):
    """precision at each positive's own score threshold (ties handled as one group)."""
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    cum_pos = np.cumsum(ys == 1)
    cum_all = np.arange(1, ys.size + 1)
    # last index of each tie-group of equal scores
    last_of_group = np.r_[np.flatnonzero(np.diff(ss) != 0), ys.size - 1]
    grp_id = np.searchsorted(last_of_group, np.arange(ys.size))
    prec_at_group = cum_pos[last_of_group] / cum_all[last_of_group]
    prec_each = prec_at_group[grp_id]           # in sorted order
    out = np.empty(ys.size, dtype=np.float64)
    out[order] = prec_each                      # back to original row order
    return out


def pr_f1(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def conf_block(y, s, t):
    pred = s >= t
    pos = y == 1
    tp = int(np.count_nonzero(pred & pos)); fp = int(np.count_nonzero(pred & ~pos))
    fn = int(np.count_nonzero(~pred & pos)); tn = int(np.count_nonzero(~pred & ~pos))
    d = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_flagged": tp + fp}
    d.update(pr_f1(tp, fp, fn))
    return d


def main():
    with np.load(SIDE, allow_pickle=True) as z:
        y = np.asarray(z["y"]).astype(np.int64)
        reading = np.asarray(z["reading"]).astype(np.float64)
    m0 = reading == 0.0
    mN = ~m0
    P = int((y == 1).sum()); P0 = int((y[m0] == 1).sum()); PN = int((y[mN] == 1).sum())
    w0, wN = P0 / P, PN / P
    print(f"positives: total={P} r0={P0} ({w0:.4%}) rN={PN} ({wN:.4%})", flush=True)

    res = {"thresh": THRESH, "P": P, "P0": P0, "PN": PN, "w0": w0, "wN": wN, "cells": []}
    for budget in (50, 100, 200, 400, 725):
        for model in ("tree", "tabpfn"):
            for gid, rid, path in cell_paths(budget, model):
                s = load_hot(path, SCORE_KEY[model], y)
                # --- exact AP decomposition
                c = per_positive_precision(y, s)
                cpos = c[y == 1]
                ap_id = float(cpos.mean())
                ap_sk = float(average_precision_score(y, s))
                assert abs(ap_id - ap_sk) < 1e-9, (budget, model, gid, rid, ap_id, ap_sk)
                C0 = float(c[(y == 1) & m0].mean())
                CN = float(c[(y == 1) & mN].mean())
                # --- R-precision: flag exactly P rows (top-P by score)
                thr_rp = float(np.partition(s, s.size - P)[s.size - P])
                rp = conf_block(y, s, thr_rp)
                rec = {"budget": budget, "model": model, "group": gid, "rseed": rid,
                       "ap": ap_sk, "ap_identity": ap_id, "C0": C0, "CN": CN,
                       "contrib0": w0 * C0, "contribN": wN * CN,
                       "conf_all": conf_block(y, s, THRESH),
                       "conf_r0": conf_block(y[m0], s[m0], THRESH),
                       "conf_rN": conf_block(y[mN], s[mN], THRESH),
                       "rprec_thresh": thr_rp, "rprec": rp,
                       "q": {str(q): float(np.quantile(s, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
                       "q_pos": {str(q): float(np.quantile(s[y == 1], q)) for q in (0.1, 0.5, 0.9)},
                       "q_neg": {str(q): float(np.quantile(s[y == 0], q)) for q in (0.5, 0.9, 0.99)},
                       "frac_ge_05": float(np.mean(s >= THRESH))}
                res["cells"].append(rec)
                print(f"  ok K={budget:>3} {model:6} {gid:>3}/{rid:>3} AP={ap_sk:.4f}"
                      f" = {w0:.3f}*{C0:.4f} + {wN:.3f}*{CN:.4f}"
                      f" | F1@.5 all={rec['conf_all']['f1']:.3f}"
                      f" r0={rec['conf_r0']['f1']:.3f} rN={rec['conf_rN']['f1']:.3f}"
                      f" | flagged={rec['frac_ge_05']:.3%}", flush=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}  cells={len(res['cells'])}")


if __name__ == "__main__":
    sys.exit(main())
