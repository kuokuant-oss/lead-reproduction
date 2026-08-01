"""Freeze the M5 chilledwater C1 localization protocol.

Written and executed BEFORE any C1 result is read. It fixes the primary
question, the four candidate explanations, the analysis dimensions, the
estimands, and the decision gate, so that none of them can be added or altered
after seeing results.

C1 is CPU-only localization over existing artifacts. It performs no fit, no
refit, no TabPFN inference, no tree refit, and never scores the frozen 192-row
query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

E0_COMMIT = "5e44479632ac57929f260bf12953df4002b7ad41"
PINNED_ANALYSIS_COMMIT = "d8e59da2c40cb5102367d6a73299e807680f6ca6"

PROTOCOL: dict[str, object] = {
    "schema": "m5_chilledwater_c1_protocol_v1",
    "execution_mode": "C1_LOCALIZATION",
    "e0_completion_commit": E0_COMMIT,
    "pinned_analysis_commit": PINNED_ANALYSIS_COMMIT,
    "primary_question": (
        "Why does chilledwater show a stable PR-AUC advantage at 100k context "
        "while its ROC-AUC bootstrap interval very nearly includes zero?"
    ),
    "e0_inputs_fixed": {
        "chilledwater_pr_auc_gap": 0.048618,
        "chilledwater_pr_auc_ci": [0.001685, 0.096540],
        "chilledwater_roc_auc_gap": 0.004285,
        "chilledwater_roc_auc_ci": [-0.000027, 0.008991],
        "chilledwater_classification": "observed advantage but not stable",
        "note": (
            "E0 classification is an input, not an output. C1 must not rewrite it, "
            "and must not reclassify it merely because the ROC-AUC interval misses "
            "zero by 2.7e-5."
        ),
    },
    "candidate_explanations": [
        {
            "id": "A",
            "key": "SAME_HOTWATER_NEGATIVE_REFERENCE",
            "statement": (
                "Chilledwater-positive moves against hotwater-negative in the same "
                "stable direction as steam does; the reference group is shared."
            ),
        },
        {
            "id": "B",
            "key": "DIFFERENT_SUPPORT_SOURCE",
            "statement": (
                "Another meter/label support group explains the stable movement "
                "better than hotwater-negative does."
            ),
        },
        {
            "id": "C",
            "key": "WITHIN_METER_MORPHOLOGY",
            "statement": (
                "The effect localizes to temporal morphology or representation "
                "inside chilledwater rather than to a cross-meter reference."
            ),
        },
        {
            "id": "D",
            "key": "NO_STABLE_LOCALIZATION",
            "statement": (
                "The PR-AUC performance gap exists but no candidate mechanism "
                "localizes stably."
            ),
        },
    ],
    "analysis_dimensions": {
        "context_rows": [5000, 10000, 20000, 50000, 100000],
        "learners_separated": ["tabpfn", "matched_row_tree"],
        "movement_separated": ["score_movement", "rank_movement"],
        "row_classes_separated": ["anomaly_rows", "normal_rows"],
        "comparison_scopes_separated": ["within_chilledwater", "cross_meter"],
        "effect_types_separated": ["calibration_absolute_score", "ranking"],
    },
    "morphology_factors": [
        "raw_reading_quartile",
        "anomaly_phase_onset_middle_recovery",
        "duration_rows",
        "reading_slope",
        "deviation_24h",
        "deviation_168h",
        "diff_1h_mean",
        "ratio_1h_mean",
        "building_id",
        "segment_id",
    ],
    "support_source_comparisons": [
        "chilledwater_positive_vs_hotwater_negative",
        "chilledwater_positive_vs_electricity_negative",
        "chilledwater_positive_vs_chilledwater_negative",
        "chilledwater_positive_vs_steam_negative",
    ],
    "per_comparison_readouts": [
        "pairwise_auc",
        "continuous_score_margin",
        "rank_gap",
        "context_size_movement",
        "tabpfn_minus_tree_movement",
        "building_coverage",
        "segment_coverage",
    ],
    "robustness_requirements": {
        "context_directions_checked": [10000, 20000, 50000, 100000],
        "building_clustered_uncertainty": True,
        "segment_clustered_uncertainty": True,
        "exact_leave_one_building_influence": True,
        "concentration_reported": True,
        "valid_pair_resolution_reported": True,
        "rows_are_not_treated_as_independent": True,
        "note": (
            "Row-level independence would produce spuriously narrow intervals; "
            "all uncertainty is clustered by building and by segment."
        ),
    },
    "query_resolution_audit": {
        "query": "original 352-row screening query",
        "strata_required": [
            "chilledwater_positive",
            "chilledwater_negative",
            "electricity_negative",
            "steam_negative",
            "hotwater_negative",
        ],
        "metrics_required": [
            "row_count",
            "positive_count",
            "negative_count",
            "buildings",
            "segments",
            "valid_positive_negative_pairs",
            "pairwise_auc_resolution",
            "building_concentration",
            "segment_concentration",
        ],
        "rule": (
            "A continuous margin is a feasibility readout only; it cannot "
            "substitute for inadequate valid-pair resolution."
        ),
    },
    "decision_gate": {
        "exactly_one_outcome": True,
        "outcomes": [
            "SAME_HOTWATER_NEGATIVE_REFERENCE",
            "DIFFERENT_SUPPORT_SOURCE",
            "WITHIN_METER_MORPHOLOGY",
            "NO_STABLE_LOCALIZATION",
        ],
        "direction_stability_threshold": 0.95,
        "concentration_limit": 0.25,
        "context_consistency_required_contexts": [20000, 50000, 100000],
        "rule_same_reference": (
            "chilledwater-positive vs hotwater-negative is directionally stable "
            "across contexts and clustered resamples, and matches the steam sign"
        ),
        "rule_different_source": (
            "a non-hotwater negative group shows strictly greater directional "
            "stability and larger separation than hotwater-negative"
        ),
        "rule_within_meter": (
            "movement concentrates in identifiable within-chilledwater morphology "
            "strata and survives building- and segment-clustered uncertainty, "
            "while no cross-meter reference is stable"
        ),
        "rule_no_localization": (
            "no candidate satisfies its rule under clustered uncertainty"
        ),
    },
    "prohibitions": [
        "no model fit or refit",
        "no TabPFN inference",
        "no tree refit",
        "no scoring of the frozen 192-row query",
        "no context-curve rerun",
        "no resegmentation or post-hoc cutpoint adjustment",
        "no new candidate mechanism after results are seen",
        "no change to the primary endpoint",
        "no manuscript edit",
        "no Path A, Path B, E3 variance pilot, site transfer, 500k, or "
        "full-holdout refit",
        "no GPU use; remote host is a CPU compute node only",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    body = json.dumps(PROTOCOL, indent=2, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    payload = json.dumps(
        {"protocol_sha256": digest, "protocol": PROTOCOL},
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    path = args.output_root / "c1_protocol.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") != digest:
            raise SystemExit(
                "refusing to overwrite a frozen protocol with different content: "
                f"{existing.get('protocol_sha256')} != {digest}"
            )
        print(f"protocol already frozen and identical: {digest}")
        return 0
    path.write_text(payload, encoding="utf-8")
    print(f"froze C1 protocol -> {path}")
    print(f"protocol_sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
