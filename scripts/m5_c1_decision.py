"""Apply the frozen C1 decision gate.

The rule was fixed in c1_protocol.json before any C1 result was read. This script
evaluates it against the assembled summary and selects exactly one outcome.

One correction is applied and recorded rather than hidden: the frozen
`concentration_limit` of 0.25 was written for a segment-level top-10 share (as in
E0). Applied to a four-bin quartile split it is meaningless, because 0.25 IS the
uniform baseline. Concentration is therefore judged against each factor's own
uniform baseline (1/k), and the threshold is reported alongside so the deviation
is visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIMARY_CONTEXT = 100000
DIRECTION_STABLE = 0.95
CONCENTRATION_MULTIPLE = 1.5  # >1.5x the uniform baseline counts as structure
CROSS_METER = ("hotwater", "electricity", "steam")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.output_root
    summary = json.loads((root / "c1_summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "c1_protocol.json").read_text(encoding="utf-8"))

    support = summary["support_source_comparison"]
    by_group: dict[str, dict[int, float]] = {}
    for row in support:
        by_group.setdefault(row["negative_group"], {})[row["context_rows"]] = row[
            "auc_tabpfn_minus_tree"
        ]

    # A / B: is any cross-meter negative reference stably positive?
    cross_meter_eval = {}
    for g in CROSS_METER:
        vals = list(by_group[g].values())
        cross_meter_eval[g] = {
            "per_context": by_group[g],
            "all_positive": all(v > 0 for v in vals),
            "positive_contexts": sum(v > 0 for v in vals),
            "primary_context_value": by_group[g][PRIMARY_CONTEXT],
        }
    within = {
        "per_context": by_group["chilledwater"],
        "all_positive": all(v > 0 for v in by_group["chilledwater"].values()),
        "primary_context_value": by_group["chilledwater"][PRIMARY_CONTEXT],
    }
    any_cross_meter_stable = any(v["all_positive"] for v in cross_meter_eval.values())
    hotwater_stable = cross_meter_eval["hotwater"]["all_positive"]

    # Clustered uncertainty at the primary context, both clusterings.
    unc = summary["clustered_uncertainty"]
    clustered = {}
    for cluster in ("building", "segment"):
        c = unc[cluster]
        clustered[cluster] = {
            m: {
                "median": c[f"{m}_gap_{PRIMARY_CONTEXT}"]["median"],
                "q025": c[f"{m}_gap_{PRIMARY_CONTEXT}"]["q025"],
                "q975": c[f"{m}_gap_{PRIMARY_CONTEXT}"]["q975"],
                "positive_fraction": c[f"{m}_gap_{PRIMARY_CONTEXT}"][
                    "positive_fraction"
                ],
                "excludes_zero": c[f"{m}_gap_{PRIMARY_CONTEXT}"]["excludes_zero"],
            }
            for m in ("pr", "roc")
        }
    survives = all(
        clustered[cl]["pr"]["excludes_zero"]
        and clustered[cl]["pr"]["positive_fraction"] >= DIRECTION_STABLE
        for cl in ("building", "segment")
    )

    # Morphology structure judged against each factor's uniform baseline.
    morph = summary["morphology_localization"]
    structure = {}
    for name, payload in morph.items():
        if "top_stratum_share" in payload:
            k = payload.get("strata_count") or 0
            base = 1.0 / k if k else float("nan")
            share = payload["top_stratum_share"]
            structure[name] = {
                "strata": k,
                "top_share": share,
                "uniform_baseline": base,
                "multiple_of_uniform": share / base if base else None,
                "has_structure": bool(base and share / base >= CONCENTRATION_MULTIPLE),
            }
        elif "top10_share" in payload:
            structure[name] = {
                "segments": payload["segments"],
                "top1_share": payload["top1_share"],
                "top10_share": payload["top10_share"],
                "top50_share": payload["top50_share"],
                "has_structure": payload["top10_share"] >= 0.25,
            }
    structured_factors = [k for k, v in structure.items() if v.get("has_structure")]

    phase = morph.get("anomaly_phase", {}).get("strata", [])
    phase_map = {r["phase"]: r["mean_gap_movement"] for r in phase}
    onset_ratio = phase_map.get("onset", 0) / max(
        abs(phase_map.get("middle", 1)), 1e-12
    )

    # ---- frozen gate ----
    if hotwater_stable:
        decision = "SAME_HOTWATER_NEGATIVE_REFERENCE"
        reason = (
            "chilledwater-positive vs hotwater-negative is positive at every context"
        )
    elif any_cross_meter_stable:
        decision = "DIFFERENT_SUPPORT_SOURCE"
        reason = "a non-hotwater cross-meter negative group is stably positive"
    elif within["all_positive"] and survives and structured_factors:
        decision = "WITHIN_METER_MORPHOLOGY"
        reason = (
            "the advantage exists only against chilledwater's own negatives, "
            "survives building- and segment-clustered uncertainty, and shows "
            "within-meter morphology structure, while no cross-meter reference "
            "is stable"
        )
    else:
        decision = "NO_STABLE_LOCALIZATION"
        reason = "no candidate satisfies its frozen rule"

    payload = {
        "schema": "m5_c1_decision_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "decision": decision,
        "reason": reason,
        "e0_classification_unchanged": "observed advantage but not stable",
        "e0_not_rewritten_note": (
            "C1 does not reclassify E0. C1's building-clustered ROC-AUC interval at "
            "100k excludes zero by 7.4e-5 while E0's misses zero by 2.7e-5; both are "
            "indistinguishable from the boundary and neither overturns the other."
        ),
        "support_source": {
            "within_meter_chilledwater_negative": within,
            "cross_meter": cross_meter_eval,
            "any_cross_meter_stable": any_cross_meter_stable,
        },
        "clustered_uncertainty_primary_context": clustered,
        "survives_clustered_uncertainty": survives,
        "morphology_structure": structure,
        "structured_factors": structured_factors,
        "onset_vs_middle_gap_ratio": onset_ratio,
        "threshold_correction": {
            "frozen_concentration_limit": protocol["protocol"]["decision_gate"][
                "concentration_limit"
            ],
            "applied_rule": "top-stratum share >= 1.5x the factor's uniform baseline (1/k)",
            "why": (
                "the frozen 0.25 limit was written for a segment-level top-10 share; "
                "on a four-bin quartile split 0.25 is the uniform baseline and the "
                "test would be vacuous"
            ),
        },
        "next_action_permitted": {
            "SAME_HOTWATER_NEGATIVE_REFERENCE": "retain as Path-A boundary readout; no new factorial",
            "DIFFERENT_SUPPORT_SOURCE": "propose one narrow support-source intervention protocol; do not execute",
            "WITHIN_METER_MORPHOLOGY": "propose one targeted support or feature contrast; do not start Path B or representation ablation",
            "NO_STABLE_LOCALIZATION": "report the performance gap only; make no chilledwater mechanism claim",
        }[decision],
    }
    out = root / "c1_decision.json"
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {out}\n")
    print(f"DECISION: {decision}")
    print(f"  reason: {reason}\n")
    print("cross-meter references (TabPFN-minus-tree pairwise AUC):")
    for g, v in cross_meter_eval.items():
        print(
            f"  {g:12s} all_positive={v['all_positive']} "
            f"@100k={v['primary_context_value']:+.6f}"
        )
    print(
        f"  within-meter all_positive={within['all_positive']} "
        f"@100k={within['primary_context_value']:+.6f}"
    )
    print(f"\nsurvives clustered uncertainty: {survives}")
    print(f"structured morphology factors: {structured_factors}")
    print(f"onset/middle gap ratio: {onset_ratio:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
