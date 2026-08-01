"""Assemble the C1 summary from a complete unit set.

Refuses to run on a partial unit set: no partial-result interpretation. Reuses
the exact leave-one-building influence already computed and finalized in E0 for
the chilledwater 100k learner gap, because it is the identical estimand -- it is
read, not recomputed, and never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROBUSTNESS_CONTEXTS = (10000, 20000, 50000, 100000)
NEG_GROUPS = ("hotwater", "electricity", "chilledwater", "steam")
EXPECTED_UNITS = 236


def load_units(unit_dir: Path) -> dict[str, dict]:
    units = {}
    for p in sorted(unit_dir.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        units[rec["unit_id"]] = rec["payload"]
    return units


def clustered_intervals(units: dict[str, dict], cluster: str) -> dict:
    draws: list[dict] = []
    for uid, payload in units.items():
        if uid.startswith(f"bootstrap__{cluster}__"):
            draws.extend(payload["draws"])
    valid = [d for d in draws if "invalid" not in d]
    out: dict[str, object] = {
        "cluster": cluster,
        "draws_total": len(draws),
        "draws_valid": len(valid),
        "draws_invalid": len(draws) - len(valid),
        "clusters": next(
            p["clusters"]
            for u, p in units.items()
            if u.startswith(f"bootstrap__{cluster}__")
        ),
    }
    for ctx in ROBUSTNESS_CONTEXTS:
        for metric in ("pr", "roc"):
            key = f"{metric}_gap_{ctx}"
            vals = np.array([d[key] for d in valid if key in d], dtype="float64")
            if vals.size == 0:
                continue
            out[key] = {
                "median": float(np.median(vals)),
                "q025": float(np.quantile(vals, 0.025)),
                "q975": float(np.quantile(vals, 0.975)),
                "positive_fraction": float((vals > 0).mean()),
                "excludes_zero": bool(
                    np.quantile(vals, 0.025) > 0 or np.quantile(vals, 0.975) < 0
                ),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--e0-loo-summary", type=Path, required=True)
    args = ap.parse_args()

    unit_dir = args.output_root / "units"
    units = load_units(unit_dir)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(
            f"refusing to summarise a partial unit set: {len(units)}/{EXPECTED_UNITS}"
        )

    movement = {u: p for u, p in units.items() if u.startswith("movement__")}
    support = {u: p for u, p in units.items() if u.startswith("support__")}
    morphology = {u: p for u, p in units.items() if u.startswith("morphology__")}

    # Support-source comparison table at the primary context.
    support_rows = []
    for neg in NEG_GROUPS:
        for ctx in (5000, 10000, 20000, 50000, 100000):
            p = support.get(f"support__{neg}__ctx{ctx}")
            if not p:
                continue
            support_rows.append(
                {
                    "negative_group": neg,
                    "context_rows": ctx,
                    "tabpfn_pairwise_auc": p["tabpfn"]["pairwise_auc"],
                    "tree_pairwise_auc": p["tree"]["pairwise_auc"],
                    "auc_tabpfn_minus_tree": p["tabpfn_minus_tree"]["pairwise_auc"],
                    "score_margin_tabpfn": p["tabpfn"]["continuous_score_margin"],
                    "score_margin_tabpfn_minus_tree": p["tabpfn_minus_tree"][
                        "continuous_score_margin"
                    ],
                    "rank_gap_tabpfn": p["tabpfn"].get("rank_gap"),
                    "positive_rows": p["positive_rows"],
                    "negative_rows": p["negative_rows"],
                    "positive_buildings": p["positive_buildings"],
                    "negative_buildings": p["negative_buildings"],
                }
            )

    e0_loo = pd.read_csv(args.e0_loo_summary)
    cw_loo = e0_loo[(e0_loo.meter == "chilledwater")]
    loo_rows = json.loads(cw_loo.to_json(orient="records"))

    payload = {
        "schema": "m5_c1_summary_v1",
        "execution_mode": "C1_LOCALIZATION",
        "units": len(units),
        "no_fit_no_inference": True,
        "frozen_192_row_query_scored": False,
        "movement_decomposition": {u.split("__")[1]: p for u, p in movement.items()},
        "support_source_comparison": support_rows,
        "morphology_localization": {
            u.split("__", 1)[1]: p for u, p in morphology.items()
        },
        "clustered_uncertainty": {
            "building": clustered_intervals(units, "building"),
            "segment": clustered_intervals(units, "segment"),
            "note": (
                "rows are never treated as independent; every interval is a "
                "cluster bootstrap over buildings or over segment/building "
                "clusters"
            ),
        },
        "exact_leave_one_building_influence": {
            "source": "E0 formal LOO phase (identical estimand), read not recomputed",
            "rows": loo_rows,
        },
    }
    out = args.output_root / "c1_summary.json"
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    for cluster in ("building", "segment"):
        ci = payload["clustered_uncertainty"][cluster]
        print(
            f"\n=== {cluster}-clustered ({ci['clusters']} clusters, "
            f"{ci['draws_valid']} valid draws) ==="
        )
        for ctx in ROBUSTNESS_CONTEXTS:
            for metric in ("pr", "roc"):
                k = f"{metric}_gap_{ctx}"
                if k in ci:
                    v = ci[k]
                    print(
                        f"  {k:16s} median={v['median']:+.6f} "
                        f"[{v['q025']:+.6f}, {v['q975']:+.6f}] "
                        f"pos={v['positive_fraction']:.3f} "
                        f"excl0={v['excludes_zero']}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
