"""How many of the 73 buildings are IN the situation the case figure illustrates?

This replaces the wrong question asked by season_prevalence.py. That script tested
whether a single building shows the seasonal pattern ACROSS both seasons (paired) --
which is not the claim. The claim is about the population: in summer the hot-water
meter sits at zero for most of the season and only a small minority of those zeros
is anomalous, and that is what makes the reading==0 task hard in summer.

So the right prevalence questions are per-building situation counts, not paired
tests:

  S1  summer zero-rate >= 50%                      "the meter is mostly off"
  S2  summer zero-rate > winter zero-rate          the composition driver
  S3  has any anomalous zero in summer             who contributes positives
  S4  the b1237 situation: summer zero-rate >= 50% AND 1% <= anomalous share
      of summer zeros <= 20%                       "a haystack with a few needles"
  S5  among buildings whose summer zeros are scorable (>= 10 of each kind),
      how many have Tree ahead                     the model side, unpaired
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k")
FORMAL = REPO / "data/processed/m5_building_curve/v5_fixed_50k/model_runs"
SIDE = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
            "/analysis_2026-09-03/data/hotwater_sideinfo.npz")
DATA = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
            "/analysis_2026-09-08/data")
CACHE = Path("/home/kuant_kuo/.cache/lead_hw_row_time.npz")
OUT = DATA / "season_situation.json"
SK = {"tree": "ensemble", "tabpfn": "tabpfn"}
BUDGET = 400


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
    with np.load(CACHE) as z:
        t = z["t"]
    mon = pd.to_datetime(pd.Series(t), unit="s").dt.month.to_numpy()
    SUM = np.isin(mon, [6, 7, 8])
    WIN = np.isin(mon, [12, 1, 2])

    S = {}
    for model in ("tree", "tabpfn"):
        ps = [FORMAL / f"building_seed{b}" / f"row_seed{r}"
              / f"{'tree_no_es' if model == 'tree' else 'tabpfn'}_k{BUDGET}_f137"
              / "predictions.npz" for b in range(5) for r in range(2)]
        print(f"averaging {model} K={BUDGET} over {len(ps)} cells ...", flush=True)
        S[model] = np.mean([load_hot(p, SK[model], y) for p in ps], axis=0)

    per = []
    bl = np.unique(bid)
    for b in bl:
        mb = bid == b
        e = {"bid": int(b), "use": str(use[mb][0])}
        for tag, wm in (("summer", SUM), ("winter", WIN)):
            sel = mb & wm
            n = int(sel.sum())
            nz = int((sel & m0).sum())
            na = int((sel & m0 & (y == 1)).sum())
            e[f"{tag}_n"] = n
            e[f"{tag}_zero_rate"] = nz / n if n else None
            e[f"{tag}_zeros"] = nz
            e[f"{tag}_zero_anom"] = na
            e[f"{tag}_anom_share"] = na / nz if nz else None
        sel = mb & SUM & m0
        yv = y[sel]
        if int((yv == 1).sum()) >= 10 and int((yv == 0).sum()) >= 10:
            at = roc_auc_score(yv, S["tree"][sel])
            ap = roc_auc_score(yv, S["tabpfn"][sel])
            e["summer_r0_auc"] = {"tree": float(at), "tabpfn": float(ap), "gap": float(at - ap),
                                  "npos": int((yv == 1).sum()), "nneg": int((yv == 0).sum())}
        else:
            e["summer_r0_auc"] = None
        per.append(e)

    def frac(sel_fn, label, denom=len(bl)):
        hits = [e for e in per if sel_fn(e)]
        print(f"   {label}: {len(hits)} / {denom}  ({len(hits)/denom:.0%})")
        return {"hits": len(hits), "denom": denom,
                "bids": [e["bid"] for e in hits]}

    res = {"per_building": per, "counts": {}}
    print("\n" + "=" * 92)
    print("有多少棟建築處於「案例所描繪的情境」？（全 73 棟）")
    print("=" * 92)
    c = res["counts"]
    c["S1_summer_zero_ge50"] = frac(
        lambda e: (e["summer_zero_rate"] or 0) >= 0.5,
        "S1 夏季零讀數率 ≥ 50%（整季錶大多停在 0）")
    c["S2_summer_gt_winter_zero"] = frac(
        lambda e: e["summer_zero_rate"] is not None and e["winter_zero_rate"] is not None
        and e["summer_zero_rate"] > e["winter_zero_rate"],
        "S2 夏季零讀數率 > 冬季（組成效應的來源）")
    c["S3_has_summer_anom_zero"] = frac(
        lambda e: e["summer_zero_anom"] > 0,
        "S3 夏季有任何「異常的 0」（貢獻正樣本者）")
    c["S4_haystack"] = frac(
        lambda e: (e["summer_zero_rate"] or 0) >= 0.5
        and e["summer_zeros"] > 0
        and 0.01 <= (e["summer_anom_share"] or 0) <= 0.20,
        "S4 b1237 情境：夏季零讀數率 ≥ 50% 且其中 1–20% 是異常（大海撈針）")
    c["S4b_haystack_loose"] = frac(
        lambda e: (e["summer_zero_rate"] or 0) >= 0.3
        and e["summer_zeros"] > 0
        and 0 < (e["summer_anom_share"] or 0) <= 0.30,
        "S4′ 放寬：夏季零讀數率 ≥ 30% 且異常占比 0–30%")

    sc = [e for e in per if e["summer_r0_auc"]]
    ahead = [e for e in sc if e["summer_r0_auc"]["gap"] > 0]
    gaps = sorted(e["summer_r0_auc"]["gap"] for e in sc)
    print(f"\n   S5 夏季 reading = 0 可評分（兩類各 ≥ 10 小時）的建築: {len(sc)} / {len(bl)}")
    print(f"      其中 Tree 領先: {len(ahead)} / {len(sc)}  "
          f"gap 中位數 {np.median(gaps):+.3f}  範圍 {gaps[0]:+.3f} .. {gaps[-1]:+.3f}")
    c["S5_summer_r0_scorable"] = {"hits": len(sc), "denom": int(len(bl))}
    c["S5_summer_r0_tree_ahead"] = {"hits": len(ahead), "denom": len(sc),
                                    "median_gap": float(np.median(gaps)),
                                    "min_gap": float(gaps[0]), "max_gap": float(gaps[-1])}

    # distribution of summer anomalous share, for the narrative
    sh = [e["summer_anom_share"] for e in per if e["summer_anom_share"] is not None]
    print(f"\n   夏季「零讀數中異常占比」分布（{len(sh)} 棟有夏季零讀數）:")
    print(f"      min {min(sh):.1%}  p25 {np.quantile(sh,.25):.1%}  median {np.median(sh):.1%}"
          f"  p75 {np.quantile(sh,.75):.1%}  max {max(sh):.1%}")
    print(f"      = 0%（夏季的 0 全部合法）: {sum(1 for x in sh if x == 0)} 棟")
    c["summer_anom_share_dist"] = {"n": len(sh), "min": float(min(sh)),
                                   "p25": float(np.quantile(sh, .25)),
                                   "median": float(np.median(sh)),
                                   "p75": float(np.quantile(sh, .75)),
                                   "max": float(max(sh)),
                                   "all_legit": sum(1 for x in sh if x == 0)}

    print(f"\n   夏季 reading = 0 可評分建築的 gap 明細:")
    print(f"{'bid':>6} {'use':<24} {'pos':>6} {'neg':>6} {'AUC T':>7} {'AUC P':>7} {'gap':>8}")
    for e in sorted(sc, key=lambda e: -e["summer_r0_auc"]["gap"]):
        a = e["summer_r0_auc"]
        print(f"{e['bid']:>6} {e['use'][:24]:<24} {a['npos']:>6,} {a['nneg']:>6,}"
              f" {a['tree']:>7.3f} {a['tabpfn']:>7.3f} {a['gap']:>+8.3f}")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
