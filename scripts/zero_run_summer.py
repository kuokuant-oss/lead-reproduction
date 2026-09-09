"""THE OPEN QUESTION -- can a model tell a long summer zero-run apart?

Hypothesis (from the author, 2026-09-09):
    在夏季，Tree 能辨別「長時間讀數為 0」的區塊是正常還是異常狀態，
    TabPFN 較抓不到。

Why this is not what the earlier analysis measured
--------------------------------------------------
The published report scores every reading == 0 *hour*. Hours inside one outage
are near-duplicates, so an hour-level AUC is dominated by however many hours
the longest runs happen to contain, and its effective sample size is far below
its nominal n. The hypothesis above is about *runs*, so the unit of analysis
here is the run, not the hour.

Design
------
1. Segment each building's hot-water series into maximal runs of consecutive
   reading == 0 hours (rows are strictly hourly and time-ordered within a
   building; this was verified -- consecutive rows advance +1 h, fraction
   1.0000).
2. Keep runs that (a) lie inside the summer window and (b) are LONG
   (length >= MIN_LEN, swept over 24 / 72 / 168 h) and (c) are label-pure,
   i.e. all hours anomalous or all normal. Mixed runs are reported separately
   and excluded from the AUC -- they have no single run-level label.
3. Give each run one score per model: the MEAN of the model's hourly scores
   over the run (also compute median and max as robustness variants).
4. Run-level ROC-AUC, one per model, over the kept runs; gap = Tree - TabPFN.
   Do this pooled, and per building where a building has both classes of run.
5. Report the honest n: number of runs, split by class. With few dozen runs
   the confidence interval is wide, so also give a paired bootstrap CI over
   runs and a per-cell (seed) spread.

What would confirm the hypothesis
---------------------------------
A positive Tree - TabPFN run-level AUC gap in the summer window that
  (a) grows or holds as MIN_LEN increases,
  (b) survives the paired bootstrap CI excluding 0,
  (c) is not produced by a single building (drop-one-building jackknife),
  (d) is larger than the same statistic computed on the winter window.
Report all four. If (b) fails, say the data cannot resolve it at this n --
that is a legitimate result, not a failure of the analysis.

Usage
-----
    python scripts/zero_run_summer.py                  # summer, all MIN_LEN
    python scripts/zero_run_summer.py --window winter
    python scripts/zero_run_summer.py --budget 725
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "hotwater_bundle.npz"
OUTDIR = ROOT / "out"

WINDOWS = {
    "summer": [6, 7, 8],
    "summer2": [7, 8],
    "winter": [12, 1, 2],
    "winter2": [11, 12],
    "all": list(range(1, 13)),
}
MIN_LENS = (24, 72, 168)
CELLS = {400: [f"b{b}r{r}" for b in range(5) for r in range(2)],
         725: [f"r{r}" for r in range(5)]}


def load(budget: int):
    z = np.load(BUNDLE, allow_pickle=False)
    d = {k: z[k] for k in ("y", "reading", "bid", "sid", "hour", "t",
                           "use_code", "use_levels")}
    d["y"] = d["y"].astype(np.int64)
    d["reading"] = d["reading"].astype(np.float64)
    scores = {}
    for model in ("tree", "tabpfn"):
        arrs = [z[f"s_{model}_k{budget}_{c}"].astype(np.float64) / 65535.0
                for c in CELLS[budget]]
        scores[model] = {"cells": arrs, "mean": np.mean(arrs, axis=0)}
    z.close()
    return d, scores


def runs_of(mask: np.ndarray):
    """(start, length) for every maximal True run."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    st = np.flatnonzero(d == 1)
    return st, np.flatnonzero(d == -1) - st


def build_runs(d, months):
    """One record per zero-run whose hours all fall in the month window."""
    # calendar month from epoch seconds, no pandas needed:
    # datetime64[M] as int counts months since 1970-01, so %12 + 1 gives 1..12
    stamp = np.datetime64("1970-01-01", "s") + d["t"].astype("timedelta64[s]")
    mon = stamp.astype("datetime64[M]").astype(np.int64) % 12 + 1
    inwin = np.isin(mon, months)
    out = []
    for b in np.unique(d["bid"]):
        mb = d["bid"] == b
        idx = np.flatnonzero(mb)
        order = np.argsort(d["t"][idx], kind="stable")
        idx = idx[order]
        z = d["reading"][idx] == 0.0
        st, ln = runs_of(z)
        for a, L in zip(st, ln):
            rows = idx[a:a + L]
            if not inwin[rows].all():
                continue                      # run must sit wholly in the window
            na = int((d["y"][rows] == 1).sum())
            out.append({"bid": int(b), "rows": rows, "len": int(L), "n_anom": na,
                        "pure": na == L or na == 0,
                        "label": 1 if na == L else (0 if na == 0 else -1),
                        "t0": int(d["t"][rows[0]]), "t1": int(d["t"][rows[-1]])})
    return out


def run_scores(recs, arr, how="mean"):
    f = {"mean": np.mean, "median": np.median, "max": np.max}[how]
    return np.array([f(arr[r["rows"]]) for r in recs], dtype=np.float64)


def boot_gap(lab, st, sp, n_boot=4000, seed=0):
    """Paired bootstrap over runs on the AUC gap."""
    rng = np.random.default_rng(seed)
    n = lab.size
    gaps = []
    for _ in range(n_boot):
        k = rng.integers(0, n, n)
        yl = lab[k]
        if yl.min() == yl.max():
            continue
        gaps.append(roc_auc_score(yl, st[k]) - roc_auc_score(yl, sp[k]))
    g = np.array(gaps)
    return (float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5)),
            float(g.mean()), int(g.size))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="summer", choices=sorted(WINDOWS))
    ap.add_argument("--budget", type=int, default=400, choices=(400, 725))
    ap.add_argument("--agg", default="mean", choices=("mean", "median", "max"))
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()

    d, S = load(a.budget)
    recs_all = build_runs(d, WINDOWS[a.window])
    print(f"window={a.window} months={WINDOWS[a.window]} budget=K{a.budget} "
          f"run-score={a.agg}")
    print(f"zero-runs wholly inside the window: {len(recs_all)}")
    mixed = [r for r in recs_all if not r["pure"]]
    print(f"  label-pure {sum(r['pure'] for r in recs_all)}, mixed {len(mixed)} "
          f"(mixed are excluded from the AUC; they hold "
          f"{sum(r['len'] for r in mixed):,} hours)")

    res = {"window": a.window, "months": WINDOWS[a.window], "budget": a.budget,
           "agg": a.agg, "levels": []}
    for L in MIN_LENS:
        recs = [r for r in recs_all if r["pure"] and r["len"] >= L]
        lab = np.array([r["label"] for r in recs])
        n1, n0 = int((lab == 1).sum()), int((lab == 0).sum())
        print("\n" + "=" * 78)
        print(f"MIN_LEN >= {L} h   runs={len(recs)}  anomalous={n1}  normal={n0}"
              f"  hours={sum(r['len'] for r in recs):,}")
        if n1 < 3 or n0 < 3:
            print("  too few runs of one class -- not scorable at this MIN_LEN")
            res["levels"].append({"min_len": L, "n_runs": len(recs),
                                  "n_anom": n1, "n_norm": n0, "scorable": False})
            continue
        st = run_scores(recs, S["tree"]["mean"], a.agg)
        sp = run_scores(recs, S["tabpfn"]["mean"], a.agg)
        at, apf = roc_auc_score(lab, st), roc_auc_score(lab, sp)
        lo, hi, bm, nb = boot_gap(lab, st, sp, a.boot)
        # per-cell (seed) spread of the gap
        cg = []
        for ct, cp in zip(S["tree"]["cells"], S["tabpfn"]["cells"]):
            cg.append(roc_auc_score(lab, run_scores(recs, ct, a.agg))
                      - roc_auc_score(lab, run_scores(recs, cp, a.agg)))
        # drop-one-building jackknife
        jk = []
        for b in sorted({r["bid"] for r in recs}):
            keep = np.array([r["bid"] != b for r in recs])
            if lab[keep].min() == lab[keep].max():
                continue
            jk.append((int(b), roc_auc_score(lab[keep], st[keep])
                       - roc_auc_score(lab[keep], sp[keep])))
        print(f"  run-level ROC-AUC   Tree {at:.4f}   TabPFN {apf:.4f}   "
              f"gap {at - apf:+.4f}")
        print(f"  paired bootstrap 95% CI on the gap: [{lo:+.4f}, {hi:+.4f}]  "
              f"(mean {bm:+.4f}, {nb} resamples)")
        print(f"  per-cell gap: mean {np.mean(cg):+.4f}  sd {np.std(cg, ddof=1):+.4f}  "
              f"Tree ahead in {sum(1 for g in cg if g > 0)}/{len(cg)} cells")
        if jk:
            worst = min(jk, key=lambda x: x[1])
            print(f"  drop-one-building jackknife: gap range "
                  f"[{min(g for _, g in jk):+.4f}, {max(g for _, g in jk):+.4f}]"
                  f"  most influential b{worst[0]} -> {worst[1]:+.4f}")
        res["levels"].append({
            "min_len": L, "scorable": True, "n_runs": len(recs),
            "n_anom": n1, "n_norm": n0,
            "auc_tree": float(at), "auc_tabpfn": float(apf),
            "gap": float(at - apf), "boot_ci": [lo, hi], "boot_mean": bm,
            "cell_gaps": [float(x) for x in cg],
            "jackknife": [[b, float(g)] for b, g in jk],
            "runs": [{"bid": r["bid"], "len": r["len"], "label": r["label"],
                      "t0": r["t0"], "t1": r["t1"]} for r in recs]})

    OUTDIR.mkdir(exist_ok=True)
    p = OUTDIR / f"zero_run_{a.window}_k{a.budget}_{a.agg}.json"
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
