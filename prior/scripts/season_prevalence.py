"""How widespread are the two seasonal patterns? -- the "x of 73 buildings" number.

Per building, per season, per subset: ROC-AUC for both models and the gap.
Then count, among buildings scorable in BOTH seasons:

  Pattern A  the reading==0 gap is LARGER in summer than in winter
  Pattern B  the reading!=0 gap is LARGER in winter than in summer

and separately, the mechanism claim that does not involve the models at all:

  Difficulty  the reading==0 task is HARDER in summer (lower AUC) than in winter
  Labels      P(anomaly | reading==0) is LOWER in summer than in winter

Seasons are fixed calendar spans (2016 is one calendar year, so Dec cannot join
Jan-Feb into a contiguous window; Jan-Feb is the deep-winter core):
  summer 2016-06-01 .. 2016-08-31      winter 2016-01-01 .. 2016-02-29
A wider winter (Dec + Jan + Feb, non-contiguous in time but the same three
calendar months) is reported alongside as a robustness check.
"""
from __future__ import annotations
import json, math
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
OUT = DATA / "season_prevalence.json"
SK = {"tree": "ensemble", "tabpfn": "tabpfn"}
BUDGET = 400


def sign_p(nw, n):
    if n == 0:
        return 1.0
    c = [math.comb(n, i) for i in range(n + 1)]
    k0 = min(nw, n - nw)
    return min(1.0, 2 * sum(c[:k0 + 1]) / (2.0 ** n))


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

    S = {}
    for model in ("tree", "tabpfn"):
        ps = [FORMAL / f"building_seed{b}" / f"row_seed{r}"
              / f"{'tree_no_es' if model == 'tree' else 'tabpfn'}_k{BUDGET}_f137"
              / "predictions.npz" for b in range(5) for r in range(2)]
        print(f"averaging {model} K={BUDGET} over {len(ps)} cells ...", flush=True)
        S[model] = np.mean([load_hot(p, SK[model], y) for p in ps], axis=0)

    WIN = {"summer": np.isin(mon, [6, 7, 8]),
           "winter": np.isin(mon, [1, 2]),
           "winter3": np.isin(mon, [12, 1, 2])}
    NEED = {"r0": (30, 30), "rN": (20, 100)}

    per = {}
    for b in np.unique(bid):
        e = {"bid": int(b), "use": str(use[bid == b][0])}
        for wname, wmask in WIN.items():
            for sub, smask in (("r0", m0), ("rN", ~m0)):
                sel = wmask & (bid == b) & smask
                yv = y[sel]
                na, nn = int((yv == 1).sum()), int((yv == 0).sum())
                need_a, need_n = NEED[sub]
                if na < need_a or nn < need_n:
                    e[f"{wname}|{sub}"] = None
                    continue
                at = roc_auc_score(yv, S["tree"][sel])
                ap = roc_auc_score(yv, S["tabpfn"][sel])
                e[f"{wname}|{sub}"] = {"auc_tree": float(at), "auc_tabpfn": float(ap),
                                       "gap": float(at - ap), "npos": na, "nneg": nn}
        # model-free: label difficulty
        for wname, wmask in WIN.items():
            sel = wmask & (bid == b) & m0
            e[f"{wname}|pz"] = (float((y[sel] == 1).mean()) if sel.sum() >= 50 else None)
        per[int(b)] = e

    res = {"per_building": per, "counts": {}}
    print("\n" + "=" * 96)
    print("HOW WIDESPREAD ARE THE TWO PATTERNS?   (73 buildings total)")
    print("=" * 96)

    def report(name, key, cmp_hi, cmp_lo, desc):
        pairs = [(b, e[cmp_hi], e[cmp_lo]) for b, e in per.items()
                 if e.get(cmp_hi) and e.get(cmp_lo)]
        n = len(pairs)
        hits = [(b, hi["gap"] - lo["gap"]) for b, hi, lo in pairs if hi["gap"] > lo["gap"]]
        p = sign_p(len(hits), n)
        med = float(np.median([hi["gap"] - lo["gap"] for _, hi, lo in pairs])) if n else float("nan")
        res["counts"][key] = {"scorable": n, "hits": len(hits), "p_sign": p, "median_diff": med,
                              "desc": desc}
        print(f"\n{name}")
        print(f"   {desc}")
        print(f"   兩季都可評分的建築: {n} / 73")
        print(f"   符合此模式: {len(hits)} / {n} ({len(hits)/max(n,1):.0%})   "
              f"sign test p = {p:.3g}")
        print(f"   逐建築差值中位數: {med:+.3f}")
        return pairs

    report("PATTERN A -- reading==0 gap larger in summer",
           "A_r0_summer_gt_winter", "summer|r0", "winter|r0",
           "gap(夏, r0) > gap(冬, r0)，也就是 Tree 在夏天的零讀數上領先更多")
    report("PATTERN B -- reading!=0 gap larger in winter",
           "B_rN_winter_gt_summer", "winter|rN", "summer|rN",
           "gap(冬, rN) > gap(夏, rN)，也就是 Tree 在冬天的非零讀數上領先更多")
    report("PATTERN A (winter = Dec+Jan+Feb)",
           "A_r0_summer_gt_winter3", "summer|r0", "winter3|r0",
           "同上，但冬季改用 12+1+2 三個月")
    report("PATTERN B (winter = Dec+Jan+Feb)",
           "B_rN_winter_gt_summer3", "winter3|rN", "summer|rN",
           "同上，但冬季改用 12+1+2 三個月")

    # --- model-free mechanism checks
    print("\n" + "-" * 96)
    print("MECHANISM (no models involved)")
    hard = [(b, e["winter|r0"]["auc_tree"], e["summer|r0"]["auc_tree"]) for b, e in per.items()
            if e.get("winter|r0") and e.get("summer|r0")]
    nh = sum(1 for _, w, s in hard if s < w)
    print(f"   reading==0 在夏天比冬天難（Tree AUC 較低）: {nh} / {len(hard)} 棟"
          f"   p = {sign_p(nh, len(hard)):.3g}")
    hardp = [(b, e["winter|r0"]["auc_tabpfn"], e["summer|r0"]["auc_tabpfn"]) for b, e in per.items()
             if e.get("winter|r0") and e.get("summer|r0")]
    nhp = sum(1 for _, w, s in hardp if s < w)
    print(f"   同上，用 TabPFN 的 AUC: {nhp} / {len(hardp)} 棟"
          f"   p = {sign_p(nhp, len(hardp)):.3g}")
    lab = [(b, e["winter|pz"], e["summer|pz"]) for b, e in per.items()
           if e.get("winter|pz") is not None and e.get("summer|pz") is not None]
    nl = sum(1 for _, w, s in lab if s < w)
    print(f"   P(異常|零讀數) 夏天低於冬天: {nl} / {len(lab)} 棟"
          f"   p = {sign_p(nl, len(lab)):.3g}")
    res["counts"]["mech_harder_summer_tree"] = {"hits": nh, "scorable": len(hard),
                                                "p_sign": sign_p(nh, len(hard))}
    res["counts"]["mech_harder_summer_tabpfn"] = {"hits": nhp, "scorable": len(hardp),
                                                  "p_sign": sign_p(nhp, len(hardp))}
    res["counts"]["mech_pz_lower_summer"] = {"hits": nl, "scorable": len(lab),
                                             "p_sign": sign_p(nl, len(lab))}

    # --- how many buildings have a POSITIVE r0 gap in summer at all
    ss = [e["summer|r0"]["gap"] for e in per.values() if e.get("summer|r0")]
    ww = [e["winter|rN"]["gap"] for e in per.values() if e.get("winter|rN")]
    print(f"\n   夏季 r0 gap > 0（Tree 領先）的建築: "
          f"{sum(1 for g in ss if g > 0)} / {len(ss)}  中位數 {np.median(ss):+.3f}")
    print(f"   冬季 rN gap > 0（Tree 領先）的建築: "
          f"{sum(1 for g in ww if g > 0)} / {len(ww)}  中位數 {np.median(ww):+.3f}")
    res["counts"]["summer_r0_tree_ahead"] = {"hits": sum(1 for g in ss if g > 0),
                                             "scorable": len(ss),
                                             "median": float(np.median(ss)) if ss else None}
    res["counts"]["winter_rN_tree_ahead"] = {"hits": sum(1 for g in ww if g > 0),
                                             "scorable": len(ww),
                                             "median": float(np.median(ww)) if ww else None}

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
