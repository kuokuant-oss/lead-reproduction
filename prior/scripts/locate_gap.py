"""Locate the Tree-vs-TabPFN gap: concentration of reading!=0 anomalies, the
stuck-meter pattern, and per-building within-subset ROC-AUC for both models.
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
META = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/building_metadata.csv")
OUT = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
           "/analysis_2026-09-08/data/locate_gap.json")
SK = {"tree": "ensemble", "tabpfn": "tabpfn"}


def k725(model, r):
    if model == "tree":
        root = FORMAL if r < 2 else EXT
        return root / "building_seed725" / f"row_seed{r}" / "tree_no_es_k725_f137" / "predictions.npz"
    return (FORMAL / "building_seed725" / f"row_seed{r}" / "tabpfn_k725_f137" / "predictions.npz"
            if r < 2 else COLAB / f"row_seed{r}" / "predictions.npz")


def k400(model, b, r):
    sub = "tree_no_es" if model == "tree" else "tabpfn"
    return FORMAL / f"building_seed{b}" / f"row_seed{r}" / f"{sub}_k400_f137" / "predictions.npz"


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
    out = {}

    print("=" * 100)
    print("1) CONCENTRATION of the 9,915 reading!=0 anomalies across the 73 buildings")
    print("=" * 100)
    rows = []
    for b in np.unique(bid):
        mb = bid == b
        rows.append((int(b), str(use[mb][0]), int((mb & mN & (y == 1)).sum()),
                     int((mb & mN).sum()), int((mb & m0 & (y == 1)).sum()), int((mb & m0).sum())))
    rows.sort(key=lambda r: -r[2])
    tot = sum(r[2] for r in rows)
    cum = 0
    print(f"{'rank':>4} {'bid':>5} {'primary_use':<22} {'rN anom':>8} {'rN rows':>8} {'rate':>7}"
          f" {'cum% of all rN anom':>20}")
    for i, r in enumerate(rows[:12], 1):
        cum += r[2]
        print(f"{i:>4} {r[0]:>5} {r[1]:<22} {r[2]:>8,} {r[3]:>8,} {r[2]/max(r[3],1):7.1%}"
              f" {cum/tot:20.1%}")
    nz = sum(1 for r in rows if r[2] > 0)
    print(f"   buildings with >=1 reading!=0 anomaly: {nz} of 73")
    print(f"   top 5 buildings hold {sum(r[2] for r in rows[:5])/tot:.1%} of all reading!=0 anomalies")
    print(f"   top 10 buildings hold {sum(r[2] for r in rows[:10])/tot:.1%}")
    out["rN_by_building"] = rows

    print("\n" + "=" * 100)
    print("2) STUCK-METER PATTERN: are reading!=0 anomalies repeated identical values?")
    print("=" * 100)
    rep_a = 0; na = 0; rep_n = 0; nn = 0
    runlen_a = []; runlen_n = []
    for b in np.unique(bid):
        mb = bid == b
        v = reading[mb]; yb = y[mb]
        same = np.r_[False, v[1:] == v[:-1]] & (v != 0)
        # run id over identical consecutive non-zero values
        newrun = ~same
        rid = np.cumsum(newrun)
        _, inv, cnt = np.unique(rid, return_inverse=True, return_counts=True)
        rl = cnt[inv]
        sel = (v != 0)
        rep_a += int(((rl >= 3) & sel & (yb == 1)).sum()); na += int((sel & (yb == 1)).sum())
        rep_n += int(((rl >= 3) & sel & (yb == 0)).sum()); nn += int((sel & (yb == 0)).sum())
        runlen_a.append(rl[sel & (yb == 1)]); runlen_n.append(rl[sel & (yb == 0)])
    ra = np.concatenate(runlen_a); rn = np.concatenate(runlen_n)
    print(f"   non-zero rows sitting in a run of >=3 identical consecutive values:")
    print(f"     anomalous: {rep_a:,}/{na:,} = {rep_a/na:.1%}")
    print(f"     normal   : {rep_n:,}/{nn:,} = {rep_n/nn:.1%}")
    print(f"   identical-value run length, median/p90/max:")
    print(f"     anomalous: {int(np.median(ra))} / {int(np.quantile(ra,.9))} / {int(ra.max())}")
    print(f"     normal   : {int(np.median(rn))} / {int(np.quantile(rn,.9))} / {int(rn.max())}")
    out["stuck"] = {"anom_frac_ge3": rep_a / na, "norm_frac_ge3": rep_n / nn}

    print("\n" + "=" * 100)
    print("3) PER-BUILDING within-subset ROC-AUC at K=725 (mean over 5 row seeds), Tree vs TabPFN")
    print("   Only buildings that have both classes inside the subset can be scored.")
    print("=" * 100)
    S = {}
    for model in ("tree", "tabpfn"):
        S[model] = [load_hot(k725(model, r), SK[model], y) for r in range(5)]

    def per_b(mask):
        res = {}
        for b in np.unique(bid):
            mb = mask & (bid == b)
            yb = y[mb]
            if yb.size == 0 or yb.min() == yb.max():
                continue
            e = {}
            for model in ("tree", "tabpfn"):
                e[model] = float(np.mean([roc_auc_score(yb, s[mb]) for s in S[model]]))
            e["n"] = int(yb.size); e["npos"] = int((yb == 1).sum())
            e["pairs"] = e["npos"] * (e["n"] - e["npos"])
            res[int(b)] = e
        return res

    for lbl, mask in (("reading != 0", mN), ("reading == 0", m0)):
        pb = per_b(mask)
        gaps = sorted(pb.items(), key=lambda kv: kv[1]["tree"] - kv[1]["tabpfn"])
        tot_pairs = sum(v["pairs"] for v in pb.values())
        wt = sum((v["tree"] - v["tabpfn"]) * v["pairs"] for v in pb.values()) / tot_pairs
        nwin_t = sum(1 for _, v in pb.items() if v["tree"] > v["tabpfn"])
        print(f"\n### {lbl}   scorable buildings={len(pb)}   "
              f"pair-weighted mean gap (Tree-TabPFN) = {wt:+.4f}   Tree ahead in {nwin_t}/{len(pb)}")
        print(f"   {'-- TabPFN ahead most --':<34}{'':4}{'-- Tree ahead most --':<34}")
        print(f"   {'bid':>5} {'use':<14} {'T':>6} {'P':>6} {'gap':>7}   |"
              f" {'bid':>5} {'use':<14} {'T':>6} {'P':>6} {'gap':>7}")
        ub = {int(b): str(use[bid == b][0]) for b in np.unique(bid)}
        for i in range(6):
            lo = gaps[i]; hi = gaps[-(i + 1)]
            print(f"   {lo[0]:>5} {ub[lo[0]][:14]:<14} {lo[1]['tree']:6.3f} {lo[1]['tabpfn']:6.3f}"
                  f" {lo[1]['tree']-lo[1]['tabpfn']:+7.3f}   |"
                  f" {hi[0]:>5} {ub[hi[0]][:14]:<14} {hi[1]['tree']:6.3f} {hi[1]['tabpfn']:6.3f}"
                  f" {hi[1]['tree']-hi[1]['tabpfn']:+7.3f}")
        out[f"per_building_{'rN' if lbl.endswith('0') is False else 'r0'}"] = pb
    OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
