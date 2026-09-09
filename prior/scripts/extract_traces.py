"""Pull two REAL hot-water traces for the report figure:
  (a) a building entering a multi-week zero outage that is labelled anomalous
  (b) a stretch of building 1241 where non-zero anomalies sit on a flat value
Rows are strictly hourly and in time order within a building (verified).
Emits compact JSON: value series + anomaly flags, downsampled for SVG.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SIDE = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
            "/analysis_2026-09-03/data/hotwater_sideinfo.npz")
OUT = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
           "/analysis_2026-09-08/data/traces.json")


def runs(mask):
    d = np.diff(np.r_[0, mask.view(np.int8), 0])
    st = np.flatnonzero(d == 1)
    return st, np.flatnonzero(d == -1) - st


def main():
    with np.load(SIDE, allow_pickle=True) as z:
        y = np.asarray(z["y"]).astype(np.int64)
        reading = np.asarray(z["reading"]).astype(np.float64)
        bid = np.asarray(z["bid"]).astype(np.int64)
        use = np.asarray(z["use"])
    out = {}

    # ---- (a) find a building whose zero outage is long AND labelled anomalous,
    #          and which has healthy non-zero readings before it (so the drop shows).
    best = None
    for b in np.unique(bid):
        mb = bid == b
        v = reading[mb]
        yb = y[mb]
        st, ln = runs(v == 0.0)
        for s, L in zip(st, ln):
            if L < 400 or s < 300:
                continue
            if yb[s:s + L].mean() < 0.99:
                continue
            pre = v[max(0, s - 300):s]
            if (pre == 0).mean() > 0.15 or pre.max() <= 0:
                continue
            score = L * pre.mean()
            if best is None or score > best[0]:
                best = (score, int(b), int(s), int(L), mb)
    _, b, s, L, mb = best
    v = reading[mb]
    yb = y[mb]
    lo = max(0, s - 336)                       # two weeks before the drop
    hi = min(v.size, s + min(L, 336) + 168)    # into the outage
    seg_v, seg_y = v[lo:hi], yb[lo:hi]
    out["outage"] = {"bid": b, "use": str(use[mb][0]), "start_hour_in_window": s - lo,
                     "run_hours": L, "n": int(seg_v.size),
                     "v": [round(float(x), 2) for x in seg_v],
                     "a": seg_y.astype(int).tolist()}
    print(f"(a) outage: bid={b} use={out['outage']['use']} run={L}h window={seg_v.size}h "
          f"pre-mean={float(v[lo:s].mean()):.1f}")

    # ---- (b) the longest ANOMALOUS run of identical non-zero values, shown with the
    #          building's own varying normal readings on either side.
    best = None
    for b in np.unique(bid):
        mb = bid == b
        v = reading[mb]
        yb = y[mb]
        nzero = v != 0
        same = np.r_[True, v[1:] != v[:-1]] | ~nzero      # run boundary
        rid = np.cumsum(same)
        _, first, cnt = np.unique(rid, return_index=True, return_counts=True)
        for f, L in zip(first, cnt):
            if L < 24 or not nzero[f]:
                continue
            if yb[f:f + L].mean() < 0.99:
                continue
            if best is None or L > best[0]:
                best = (int(L), int(b), int(f), mb)
    L, b, f, mb = best
    v = reading[mb]
    yb = y[mb]
    pad = max(120, (720 - L) // 2)
    lo = max(0, f - pad)
    hi = min(v.size, f + L + pad)
    seg_v, seg_y = v[lo:hi], yb[lo:hi]
    out["flatline"] = {"bid": b, "use": str(use[mb][0]), "n": int(seg_v.size),
                       "flat_start": f - lo, "flat_hours": L,
                       "flat_value": round(float(v[f]), 2),
                       "n_anom": int((seg_y == 1).sum()),
                       "v": [round(float(x), 2) for x in seg_v],
                       "a": seg_y.astype(int).tolist()}
    print(f"(b) flatline: bid={b} use={out['flatline']['use']} flat={L}h at value "
          f"{out['flatline']['flat_value']} window={seg_v.size}h "
          f"anomalies={int((seg_y==1).sum())}")

    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
