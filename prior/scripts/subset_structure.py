"""What actually distinguishes reading==0 anomalies from reading==0 normals,
and what reading!=0 anomalies look like.  Rows are strictly hourly and in time
order within each building (verified separately), so run structure is valid.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SIDE = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
            "/analysis_2026-09-03/data/hotwater_sideinfo.npz")
OUT = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
           "/analysis_2026-09-08/data/subset_structure.json")


def runs(mask):
    """(start, length) of each maximal True run in a 1-D bool array."""
    d = np.diff(np.r_[0, mask.view(np.int8), 0])
    st = np.flatnonzero(d == 1)
    en = np.flatnonzero(d == -1)
    return st, en - st


def main():
    with np.load(SIDE, allow_pickle=True) as z:
        y = np.asarray(z["y"]).astype(np.int64)
        reading = np.asarray(z["reading"]).astype(np.float64)
        bid = np.asarray(z["bid"]).astype(np.int64)
        hour = np.asarray(z["hour"]).astype(np.int64)
        use = np.asarray(z["use"])
    m0 = reading == 0.0
    out = {}

    print("=" * 96)
    print("1) HOUR-OF-DAY: are anomalous zeros at different hours than legitimate zeros?")
    print("=" * 96)
    print(f"{'hour':>4} | {'zeros: %anom':>13} | {'n zero-anom':>12} {'n zero-norm':>12}"
          f" | {'nonzero: %anom':>15}")
    hr = []
    for h in range(24):
        mh = hour == h
        za = int((mh & m0 & (y == 1)).sum()); zn = int((mh & m0 & (y == 0)).sum())
        na = int((mh & ~m0 & (y == 1)).sum()); nn = int((mh & ~m0 & (y == 0)).sum())
        hr.append({"hour": h, "z_anom": za, "z_norm": zn, "n_anom": na, "n_norm": nn})
        print(f"{h:>4} | {za/(za+zn):13.1%} | {za:>12,} {zn:>12,} | {na/(na+nn):15.2%}")
    out["by_hour"] = hr
    zaf = np.array([r["z_anom"] / (r["z_anom"] + r["z_norm"]) for r in hr])
    print(f"   spread of %anomalous-among-zeros across hours: {zaf.min():.1%} .. {zaf.max():.1%}"
          f"  (range {zaf.max()-zaf.min():.1%})")

    print("\n" + "=" * 96)
    print("2) RUN STRUCTURE of consecutive reading==0 hours (within building, time-ordered)")
    print("=" * 96)
    all_runs = []
    for b in np.unique(bid):
        mb = bid == b
        z = m0[mb]; yb = y[mb]
        st, ln = runs(z)
        for s, L in zip(st, ln):
            na = int((yb[s:s + L] == 1).sum())
            all_runs.append((int(b), int(L), na))
    L = np.array([r[1] for r in all_runs]); NA = np.array([r[2] for r in all_runs])
    pure_a = NA == L; pure_n = NA == 0; mixed = ~pure_a & ~pure_n
    print(f"   zero-runs total: {len(all_runs):,}   (a run = a maximal block of consecutive zero hours)")
    print(f"   run length: median={int(np.median(L))}h  p75={int(np.quantile(L,.75))}h"
          f"  p95={int(np.quantile(L,.95))}h  max={int(L.max())}h")
    print(f"   fully-anomalous runs : {int(pure_a.sum()):>6,}  ({pure_a.sum()/len(L):5.1%})"
          f"  holding {int(L[pure_a].sum()):>8,} zero-hours")
    print(f"   fully-normal runs    : {int(pure_n.sum()):>6,}  ({pure_n.sum()/len(L):5.1%})"
          f"  holding {int(L[pure_n].sum()):>8,} zero-hours")
    print(f"   MIXED runs           : {int(mixed.sum()):>6,}  ({mixed.sum()/len(L):5.1%})"
          f"  holding {int(L[mixed].sum()):>8,} zero-hours")
    print(f"   median length: anom-runs={int(np.median(L[pure_a])) if pure_a.any() else 0}h"
          f"  normal-runs={int(np.median(L[pure_n])) if pure_n.any() else 0}h"
          f"  mixed-runs={int(np.median(L[mixed])) if mixed.any() else 0}h")
    # short (<=3h) vs long zero runs: anomaly rate
    for lo, hi, lbl in ((1, 1, "1h"), (2, 3, "2-3h"), (4, 24, "4-24h"),
                        (25, 168, "1-7d"), (169, 10 ** 9, ">7d")):
        sel = (L >= lo) & (L <= hi)
        if not sel.any():
            continue
        print(f"     runs of {lbl:>6}: n={int(sel.sum()):>6,}  zero-hours={int(L[sel].sum()):>8,}"
              f"  %of those hours anomalous={NA[sel].sum()/L[sel].sum():6.1%}")
    out["zero_runs"] = {"n": len(all_runs), "pure_anom": int(pure_a.sum()),
                        "pure_norm": int(pure_n.sum()), "mixed": int(mixed.sum()),
                        "hours_pure_anom": int(L[pure_a].sum()),
                        "hours_pure_norm": int(L[pure_n].sum()),
                        "hours_mixed": int(L[mixed].sum())}

    print("\n" + "=" * 96)
    print("3) reading != 0 anomalies: how extreme are they vs the SAME building's own non-zero level?")
    print("=" * 96)
    z_a = []; z_n = []
    for b in np.unique(bid):
        mb = (bid == b) & ~m0
        v = reading[mb]; yb = y[mb]
        med = np.median(v)
        iqr = np.subtract(*np.quantile(v, [0.75, 0.25])) or 1.0
        rz = (v - med) / iqr
        z_a.append(rz[yb == 1]); z_n.append(rz[yb == 0])
    za = np.concatenate(z_a); zn = np.concatenate(z_n)
    print(f"   robust z = (reading - building median) / building IQR, non-zero rows only")
    print(f"{'':>10} {'p05':>9} {'p25':>9} {'median':>9} {'p75':>9} {'p95':>9} {'%z>3':>7} {'%z<-1':>7}")
    for lbl, v in (("anomalous", za), ("normal", zn)):
        q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"{lbl:>10} {q[0]:9.2f} {q[1]:9.2f} {q[2]:9.2f} {q[3]:9.2f} {q[4]:9.2f}"
              f" {np.mean(v>3):7.1%} {np.mean(v<-1):7.1%}")
    out["rN_robust_z"] = {"anom_q": np.quantile(za, [.05, .25, .5, .75, .95]).tolist(),
                          "norm_q": np.quantile(zn, [.05, .25, .5, .75, .95]).tolist()}

    print("\n" + "=" * 96)
    print("4) SHARE OF EACH BUILDING'S ANOMALIES THAT ARE ZEROS (does every building look the same?)")
    print("=" * 96)
    sh = []
    for b in np.unique(bid):
        mb = bid == b
        na = int((mb & (y == 1)).sum())
        if na == 0:
            sh.append(None); continue
        sh.append(int((mb & (y == 1) & m0).sum()) / na)
    v = np.array([x for x in sh if x is not None])
    print(f"   buildings with >=1 anomaly: {v.size} of 73")
    print(f"   share of a building's anomalies that sit on reading==0:")
    print(f"     min={v.min():.1%} p25={np.quantile(v,.25):.1%} median={np.median(v):.1%}"
          f" p75={np.quantile(v,.75):.1%} max={v.max():.1%}")
    print(f"     buildings where ALL anomalies are zeros: {int((v==1).sum())}")
    print(f"     buildings where NO anomaly is a zero   : {int((v==0).sum())}")
    out["anom_zero_share_per_building"] = v.tolist()

    print("\n" + "=" * 96)
    print("5) primary_use breakdown")
    print("=" * 96)
    print(f"{'primary_use':<26} {'n':>9} {'%r0':>7} {'%anom':>7} {'%anom|r0':>9} {'%anom|rN':>9}")
    for u in sorted(set(use.tolist())):
        mu = use == u
        n = int(mu.sum())
        print(f"{str(u):<26} {n:>9,} {(mu&m0).sum()/n:7.1%} {(mu&(y==1)).sum()/n:7.1%}"
              f" {((mu&m0&(y==1)).sum()/max((mu&m0).sum(),1)):9.1%}"
              f" {((mu&~m0&(y==1)).sum()/max((mu&~m0).sum(),1)):9.1%}")

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
