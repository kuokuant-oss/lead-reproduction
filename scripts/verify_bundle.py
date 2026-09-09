"""RUN THIS FIRST. Proves the bundle is intact before any analysis is trusted.

Recomputes pooled hot-water PR-AUC and ROC-AUC from the bundle, using the
published aggregation, and compares to data/anchor_published.json --
per building seed (K=400) and per row seed (K=725), not just the means.

The check on the original 90 predictions.npz files agreed with the published
figure to 2.22e-16. The bundle quantises scores to uint16 (step 1/65535 =
1.5e-5), so exact agreement is not expected here, and the two metrics degrade
very differently:

  ROC-AUC   |d| <= 4e-7   -- rank-based, quantisation barely moves it
  PR-AUC    |d| <= 2e-4   -- average precision is tie-sensitive, and
                             quantisation collapses near-equal scores into the
                             same bucket, creating ties

KNOWN LIMITATION: this bundle is built for rank-based work (ROC-AUC, run-level
AUC, score orderings). It does NOT reproduce the published average precision
to full precision. Anyone who needs exact AP must go back to the original
predictions.npz files in the lead-reproduction working tree. The gate below is
5e-4 -- still ~40x tighter than the smallest effect the report discusses (the
smallest quoted per-building gap is 0.019).

Exit code 0 = bundle good. Non-zero = do not use the bundle.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "hotwater_bundle.npz"
ANCHOR = ROOT / "data" / "anchor_published.json"
TOL = 5e-4

CELLS = {400: [f"b{b}r{r}" for b in range(5) for r in range(2)],
         725: [f"r{r}" for r in range(5)]}
MODEL_KEY = {"tree": "ensemble", "tabpfn": "tabpfn"}


def group_of(budget: int, tag: str) -> str:
    """Replicate level: building seed for K<=400, row seed for K=725."""
    return tag[:2] if budget != 725 else tag


def main() -> int:
    z = np.load(BUNDLE, allow_pickle=False)
    y = z["y"].astype(np.int64)
    n = y.size
    print(f"bundle: {n:,} rows, {len(z.files)} arrays")
    print(f"  positives {int((y == 1).sum()):,}  "
          f"reading==0 {int((z['reading'] == 0).sum()):,}")
    assert n == 636121, f"expected 636,121 hot-water rows, got {n}"
    assert int((y == 1).sum()) == 90691, "expected 90,691 anomalies"

    anchor = json.loads(ANCHOR.read_text(encoding="utf-8"))
    print(f"  anchor status={anchor['status']}  context_rows={anchor['context_rows']}"
          f"  class_balance={anchor['class_balance']}")

    worst = {"pr_auc": 0.0, "roc_auc": 0.0}
    fails = 0
    for budget in (400, 725):
        for model in ("tree", "tabpfn"):
            per = {}
            for tag in CELLS[budget]:
                s = z[f"s_{model}_k{budget}_{tag}"].astype(np.float64) / 65535.0
                per.setdefault(group_of(budget, tag), []).append(
                    (average_precision_score(y, s), roc_auc_score(y, s)))
            key = f"{budget}|{MODEL_KEY[model]}"
            a = anchor["hot_water"][key]
            gk = ("by_building_seed" if budget != 725 else "by_row_seed")
            print(f"\nK={budget} {model}")
            for metric, idx in (("pr_auc", 0), ("roc_auc", 1)):
                pub = a[f"{gk}_{metric}"]
                mine = {g: sum(v[idx] for v in vs) / len(vs) for g, vs in sorted(per.items())}
                for g, val in mine.items():
                    gg = g.lstrip("br") if budget != 725 else g.lstrip("r")
                    ref = pub[gg]
                    d = abs(val - ref)
                    worst[metric] = max(worst[metric], d)
                    flag = "" if d < TOL else "   <-- FAIL"
                    print(f"  {metric:8} {g:>4}  bundle {val:.6f}  published {ref:.6f}"
                          f"  |d| {d:.2e}{flag}")
                    fails += d >= TOL
                mm = sum(mine.values()) / len(mine)
                ref = a[f"mean_{metric}"]
                d = abs(mm - ref)
                worst[metric] = max(worst[metric], d)
                fails += d >= TOL
                print(f"  {metric:8} MEAN  bundle {mm:.6f}  published {ref:.6f}"
                      f"  |d| {d:.2e}{'' if d < TOL else '   <-- FAIL'}")

    z.close()
    print(f"\nworst absolute difference vs published, by metric:")
    print(f"  ROC-AUC  {worst['roc_auc']:.3e}   (rank-based; quantisation-insensitive)")
    print(f"  PR-AUC   {worst['pr_auc']:.3e}   (tie-sensitive; see the docstring)")
    print(f"  tolerance {TOL:.0e}")
    if fails:
        print(f"BUNDLE VERIFY FAILED ({fails} comparisons over tolerance)")
        return 1
    print("BUNDLE VERIFY PASS -- scores, labels and row order are intact.")
    print("Rank-based analysis (ROC-AUC, run-level AUC) is exact to ~1e-6.")
    print("For exact average precision, use the original predictions.npz files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
