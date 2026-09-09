"""Per-cell standard metrics for hot water (meter==3), split by meter_reading==0.

Independent re-implementation (2026-09-08 session). Writes ONE json holding raw
per-cell numbers; all aggregation/interpretation happens downstream so the heavy
npz pass runs once.

Metrics per cell x subset in {all, r0 (reading==0), rN (reading!=0)}:
  pr   = average_precision_score
  roc  = roc_auc_score
  tp/fp/fn/tn at a FIXED threshold score >= 0.5
Plus, for the pooled set, where the threshold-0.5 errors land (r0 vs rN).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k")
BASE = REPO / "data/processed/m5_building_curve"
FORMAL = BASE / "v5_fixed_50k/model_runs"
EXT = BASE / "v5_fixed_50k_k725_row_seed_extension/model_runs"
COLAB = BASE / "v5_fixed_50k_k725_colab/model_results"
SIDE = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
            "/analysis_2026-09-03/data/hotwater_sideinfo.npz")
OUT = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
           "/analysis_2026-09-08/data/cell_metrics.json")

SCARCITY = (50, 100, 200, 400)
BSEEDS = (0, 1, 2, 3, 4)
RSEEDS = (0, 1)
THRESH = 0.5
SCORE_KEY = {"tree": "ensemble", "tabpfn": "tabpfn"}


def cell_paths(budget: int, model: str):
    """-> list of (group_id, path). group_id = building seed (scarcity) or row seed (725)."""
    sub = "tree_no_es" if model == "tree" else "tabpfn"
    out = []
    if budget != 725:
        for b in BSEEDS:
            for r in RSEEDS:
                p = FORMAL / f"building_seed{b}" / f"row_seed{r}" / f"{sub}_k{budget}_f137" / "predictions.npz"
                out.append((f"b{b}", f"r{r}", p))
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


def load_hot(path: Path, key: str, y_side: np.ndarray):
    """Return hot-water scores aligned to sideinfo row order."""
    with np.load(path) as p:
        meter = np.asarray(p["meter"]).astype(np.int64)
        hot = meter == 3
        anom = np.asarray(p["anomaly"]).astype(np.int64)[hot]
        vri = np.asarray(p["validation_raw_index"]).astype(np.int64)[hot]
        score = np.asarray(p[key]).astype(np.float64)[hot]
    if not np.array_equal(anom, y_side):
        o = np.argsort(vri, kind="stable")
        anom, score = anom[o], score[o]
        if not np.array_equal(anom, y_side):
            raise RuntimeError(f"alignment failed: {path}")
        return score, "sorted"
    return score, "same-order"


def conf(y: np.ndarray, s: np.ndarray, t: float = THRESH):
    pred = s >= t
    pos = y == 1
    tp = int(np.count_nonzero(pred & pos))
    fp = int(np.count_nonzero(pred & ~pos))
    fn = int(np.count_nonzero(~pred & pos))
    tn = int(np.count_nonzero(~pred & ~pos))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def subset_metrics(y: np.ndarray, s: np.ndarray):
    d = {"n": int(y.size), "npos": int(np.count_nonzero(y == 1))}
    d["pr"] = float(average_precision_score(y, s))
    d["roc"] = float(roc_auc_score(y, s))
    d.update(conf(y, s))
    return d


def main():
    with np.load(SIDE, allow_pickle=True) as z:
        y = np.asarray(z["y"]).astype(np.int64)
        reading = np.asarray(z["reading"]).astype(np.float64)
    m0 = reading == 0.0
    mN = ~m0
    print(f"sideinfo: n={y.size} pos={int(y.sum())} | r0 n={int(m0.sum())} pos={int(y[m0].sum())}"
          f" | rN n={int(mN.sum())} pos={int(y[mN].sum())}", flush=True)

    res = {"thresh": THRESH, "cells": []}
    for budget in (*SCARCITY, 725):
        for model in ("tree", "tabpfn"):
            for gid, rid, path in cell_paths(budget, model):
                if not path.is_file():
                    print(f"MISSING {budget} {model} {gid} {rid} :: {path}", flush=True)
                    res["cells"].append({"budget": budget, "model": model, "group": gid,
                                         "rseed": rid, "path": str(path), "missing": True})
                    continue
                s, how = load_hot(path, SCORE_KEY[model], y)
                rec = {"budget": budget, "model": model, "group": gid, "rseed": rid,
                       "path": str(path), "align": how,
                       "smin": float(s.min()), "smax": float(s.max()),
                       "all": subset_metrics(y, s),
                       "r0": subset_metrics(y[m0], s[m0]),
                       "rN": subset_metrics(y[mN], s[mN])}
                # where pooled threshold-0.5 errors land
                pred = s >= THRESH
                rec["pooled_split"] = {
                    "tp_r0": int(np.count_nonzero(pred & (y == 1) & m0)),
                    "tp_rN": int(np.count_nonzero(pred & (y == 1) & mN)),
                    "fn_r0": int(np.count_nonzero(~pred & (y == 1) & m0)),
                    "fn_rN": int(np.count_nonzero(~pred & (y == 1) & mN)),
                    "fp_r0": int(np.count_nonzero(pred & (y == 0) & m0)),
                    "fp_rN": int(np.count_nonzero(pred & (y == 0) & mN)),
                }
                res["cells"].append(rec)
                print(f"  ok K={budget:>3} {model:6} {gid:>3}/{rid:>3} {how:11}"
                      f" prAll={rec['all']['pr']:.4f} pr0={rec['r0']['pr']:.4f}"
                      f" prN={rec['rN']['pr']:.4f} rocAll={rec['all']['roc']:.4f}"
                      f" roc0={rec['r0']['roc']:.4f} rocN={rec['rN']['roc']:.4f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    n_ok = sum(1 for c in res["cells"] if not c.get("missing"))
    print(f"\nwrote {OUT}  cells_ok={n_ok} cells_missing={len(res['cells'])-n_ok}")


if __name__ == "__main__":
    sys.exit(main())
