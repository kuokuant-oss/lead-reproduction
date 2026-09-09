"""Robustness of the subset-level Tree-vs-TabPFN gaps.

(a) unweighted per-building mean gap + sign test across buildings, per K
(b) pooled subset ROC-AUC / gap with building 1241 excluded (it alone holds
    52.4% of all reading!=0 anomalies)
Paired at the replicate level (building seed for K<=400, row seed for K=725).
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k")
BD = REPO / "data/processed/m5_building_curve"
FORMAL = BD / "v5_fixed_50k/model_runs"
EXT = BD / "v5_fixed_50k_k725_row_seed_extension/model_runs"
COLAB = BD / "v5_fixed_50k_k725_colab/model_results"
SIDE = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
            "/analysis_2026-09-03/data/hotwater_sideinfo.npz")
OUT = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
           "/analysis_2026-09-08/data/robustness.json")
SK = {"tree": "ensemble", "tabpfn": "tabpfn"}
DROP = 1241


def paths(model, k):
    """-> {group: [paths]}  group = building seed (K<=400) or row seed (K=725)."""
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


def mean(v):
    return sum(v) / len(v)


def se(v):
    if len(v) < 2:
        return None
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) / math.sqrt(len(v))


def fmt(v):
    m, s = mean(v), se(v)
    return f"{m:+.4f}" + (f"+-{s:.4f}" if s is not None else "")


def sign_p(nw, n):
    """two-sided exact binomial test against p=0.5"""
    c = [math.comb(n, i) for i in range(n + 1)]
    k0 = min(nw, n - nw)
    return min(1.0, 2 * sum(c[:k0 + 1]) / (2.0 ** n))


def main():
    with np.load(SIDE, allow_pickle=True) as z:
        y = np.asarray(z["y"]).astype(np.int64)
        reading = np.asarray(z["reading"]).astype(np.float64)
        bid = np.asarray(z["bid"]).astype(np.int64)
    m0 = reading == 0.0
    mN = ~m0
    keep = bid != DROP
    bl = np.unique(bid)
    out = {}

    print(f"excluding building {DROP}: hot-water rows {int((~keep).sum()):,} of {y.size:,}; "
          f"reading!=0 anomalies removed {int((mN & (y == 1) & ~keep).sum()):,} of "
          f"{int((mN & (y == 1)).sum()):,}; reading==0 anomalies removed "
          f"{int((m0 & (y == 1) & ~keep).sum()):,} of {int((m0 & (y == 1)).sum()):,}")

    print("\nloading scores ...", flush=True)
    SC = {}
    for k in (50, 100, 200, 400, 725):
        for model in ("tree", "tabpfn"):
            for g, ps in paths(model, k).items():
                SC[(k, model, g)] = [load_hot(p, SK[model], y) for p in ps]
    print("loaded", len(SC), "groups", flush=True)

    def mog(k, model, g, fn):
        return mean([fn(s) for s in SC[(k, model, g)]])

    print("\n" + "=" * 104)
    print(f"(b) POOLED SUBSET ROC-AUC GAP (Tree-TabPFN) WITH BUILDING {DROP} EXCLUDED")
    print("=" * 104)
    print(f"{'K':>4} | {'rN gap (all 73b)':>20} {'rN gap (no 1241)':>20} |"
          f" {'r0 gap (all 73b)':>20} {'r0 gap (no 1241)':>20}")
    print("-" * 96)
    tab_b = {}
    for k in (50, 100, 200, 400, 725):
        cols = []
        for mask, lbl in ((mN, "rN"), (m0, "r0")):
            for sel, tag in ((np.ones_like(keep), "all"), (keep, "no1241")):
                mm = mask & sel
                yv = y[mm]
                d = [mog(k, "tree", g, lambda s: roc_auc_score(yv, s[mm]))
                     - mog(k, "tabpfn", g, lambda s: roc_auc_score(yv, s[mm]))
                     for g in sorted(paths("tree", k))]
                cols.append(fmt(d))
                tab_b[f"{k}|{lbl}|{tag}"] = {"mean": mean(d), "se": se(d), "diffs": d}
        print(f"{k:>4} | {cols[0]:>20} {cols[1]:>20} | {cols[2]:>20} {cols[3]:>20}")

    print(f"\nLevels on reading!=0 with 1241 excluded:")
    print(f"{'K':>4} | {'ROC-AUC Tree / TabPFN':>34} | {'PR-AUC Tree / TabPFN':>34}")
    print("-" * 78)
    mm = mN & keep
    yv = y[mm]
    for k in (50, 100, 200, 400, 725):
        vals = []
        for fn in (lambda s: roc_auc_score(yv, s[mm]), lambda s: average_precision_score(yv, s[mm])):
            t = [mog(k, "tree", g, fn) for g in sorted(paths("tree", k))]
            p = [mog(k, "tabpfn", g, fn) for g in sorted(paths("tabpfn", k))]
            vals.append(f"{mean(t):.4f}+-{se(t):.4f} / {mean(p):.4f}+-{se(p):.4f}")
        print(f"{k:>4} | {vals[0]:>34} | {vals[1]:>34}")
    out["drop1241"] = tab_b

    print("\n" + "=" * 104)
    print("(a) PER-BUILDING within-subset ROC-AUC: unweighted mean gap + sign test")
    print("    (a building is scorable only if the subset contains both classes for it)")
    print("=" * 104)
    print(f"{'K':>4} | {'reading!=0 mean gap':>22} {'Tree wins':>10} {'p(sign)':>9} |"
          f" {'reading==0 mean gap':>22} {'Tree wins':>10} {'p(sign)':>9}")
    print("-" * 104)
    per_b_out = {}
    for k in (50, 100, 200, 400, 725):
        cols = []
        for mask, lbl in ((mN, "rN"), (m0, "r0")):
            gaps = []
            for b in bl:
                mm2 = mask & (bid == b)
                yv2 = y[mm2]
                if yv2.size == 0 or yv2.min() == yv2.max():
                    continue
                t = mean([mog(k, "tree", g, lambda s: roc_auc_score(yv2, s[mm2]))
                          for g in sorted(paths("tree", k))])
                p = mean([mog(k, "tabpfn", g, lambda s: roc_auc_score(yv2, s[mm2]))
                          for g in sorted(paths("tabpfn", k))])
                gaps.append((int(b), t - p))
            g = [x[1] for x in gaps]
            nw = sum(1 for x in g if x > 0)
            pv = sign_p(nw, len(g))
            cols += [f"{mean(g):+.4f}+-{se(g):.4f}", f"{nw}/{len(g)}", f"{pv:.2e}"]
            per_b_out[f"{k}|{lbl}"] = {"mean_gap": mean(g), "se": se(g), "tree_wins": nw,
                                       "n": len(g), "p_sign": pv, "gaps": gaps}
        print(f"{k:>4} | {cols[0]:>22} {cols[1]:>10} {cols[2]:>9} |"
              f" {cols[3]:>22} {cols[4]:>10} {cols[5]:>9}")
    out["per_building_signtest"] = per_b_out
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
