"""Freeze the M5 E3 variance pilot protocol.

Written and executed before any TabPFN fit. It records the canonical policy it
inherits, the five supplementary decisions issued by the human operator on
2026-08-01, the realised cell execution order, and the pre-fit reference IQRs
used by the continuous-margin precision gate.

The base policy file is read but never modified: its historical
`designed_not_running` status is preserved and the new authorization is recorded
here instead.

No fit, no inference, no scoring of the frozen 192-row query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BASE_POLICY = REPO / "docs" / "reports" / "m5-tabpfn-repeated-inference-policy.json"

SCHEDULE_SEED = 42
CELLS = {
    "11": "hw_pos_present__hw_neg_present",
    "10": "hw_pos_present__hw_neg_excluded",
    "01": "hw_pos_excluded__hw_neg_present",
    "00": "hw_pos_excluded__hw_neg_excluded",
}
CELL_ORDER_KEYS = ["11", "10", "01", "00"]  # canonical policy order, pre-shuffle
REPEAT_BATCHES = [8, 16, 24, 32, 40]
BOUNDED_HALF_WIDTH_TARGET = 0.015
MARGIN_IQR_MULTIPLIER = 0.02
FRESH_PROCESS_CELL = "11"
FRESH_PROCESS_RUNS = 2
METER = {"electricity": 0, "chilledwater": 1, "steam": 2, "hotwater": 3}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def realised_cell_order() -> list[str]:
    """One-time random order of the four cells from PCG64(42).

    The order itself is persisted, not only the seed, so the schedule is
    reproducible even if the generator implementation changes.
    """
    rng = np.random.Generator(np.random.PCG64(SCHEDULE_SEED))
    order = list(np.array(CELL_ORDER_KEYS)[rng.permutation(len(CELL_ORDER_KEYS))])
    return [str(c) for c in order]


def reference_iqr(data_root: Path, cell_dir: str) -> dict:
    """Pre-fit reference IQR for the continuous-margin gate.

    Same cell, same query rows, fixed matched-tree comparator, steam-positive
    versus hotwater-negative, IQR of the pair-level score-margin distribution.
    Deliberately NOT derived from any TabPFN repeat.
    """
    proc = data_root / "processed"
    qpath = proc / "m5_context_stories" / "queries" / "screening" / "queries.npz"
    with np.load(qpath, allow_pickle=False) as q:
        q_raw = q["raw_index"].astype("int64")
        q_meter = q["meter"].astype("int8")
        q_anom = q["anomaly"].astype("int8")

    tpath = (
        proc
        / "m5_hotwater_label_factorial"
        / "predictions"
        / "trees"
        / "seed42"
        / cell_dir
        / "cell_specific"
        / "predictions.npz"
    )
    tree = np.load(tpath, allow_pickle=False)
    if not np.array_equal(tree["raw_index"].astype("int64"), q_raw):
        raise RuntimeError(f"tree comparator rows differ from the query: {cell_dir}")
    score = tree["score"].astype("float64")

    pos = score[(q_meter == METER["steam"]) & (q_anom == 1)]
    neg = score[(q_meter == METER["hotwater"]) & (q_anom == 0)]
    margins = (pos[:, None] - neg[None, :]).ravel()
    q1, q3 = np.quantile(margins, [0.25, 0.75])
    return {
        "cell_dir": cell_dir,
        "tree_predictions_path": str(tpath),
        "tree_predictions_sha256": sha256_file(tpath),
        "scaler_arm": "cell_specific",
        "steam_positive_rows": int(pos.size),
        "hotwater_negative_rows": int(neg.size),
        "pairs": int(margins.size),
        "q1": float(q1),
        "q3": float(q3),
        "reference_iqr": float(q3 - q1),
        "margin_half_width_target": float(MARGIN_IQR_MULTIPLIER * (q3 - q1)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--query-audit", type=Path, required=True)
    args = ap.parse_args()

    base = json.loads(BASE_POLICY.read_text(encoding="utf-8"))
    audit = json.loads(args.query_audit.read_text(encoding="utf-8"))
    order = realised_cell_order()
    iqr = {c: reference_iqr(args.data_root, CELLS[c]) for c in order}

    manifests = {}
    for c, d in CELLS.items():
        p = (
            args.data_root
            / "processed"
            / "m5_hotwater_label_factorial"
            / "manifests"
            / "seed42"
            / f"{d}.json"
        )
        m = json.loads(p.read_text(encoding="utf-8"))
        manifests[c] = {
            "cell_dir": d,
            "manifest_path": str(p),
            "manifest_sha256": sha256_file(p),
            "factorial": m["factorial"],
            "context_rows": m["context_rows"],
            "context_seed": m["context_seed"],
            "model_seed": m["model_seed"],
            "feature_tag": m["feature_tag"],
            "feature_count": m["feature_count"],
            "raw_index_sha256": m["raw_index_sha256"],
            "scaler_provenance": m["scaler_provenance"],
            "label_counts": m["label_counts"],
        }

    protocol = {
        "schema": "m5_e3_variance_pilot_protocol_v1",
        "execution_mode": "E3_VARIANCE_PILOT",
        "base_commit": "e9cb59b9bbf7977f4bda2dfbb9779fb0659d9168",
        "base_policy": {
            "path": str(BASE_POLICY.relative_to(REPO)).replace("\\", "/"),
            "sha256": sha256_file(BASE_POLICY),
            "status_in_base_policy": base["variance_pilot"]["status"],
            "note": (
                "The base policy file is preserved unmodified, including its "
                "historical designed_not_running status and its "
                "forbidden_this_round list. Authorization is recorded here, "
                "not by rewriting history."
            ),
        },
        "authorization": {
            "human_authorization_date": "2026-08-01",
            "execution_status": "HUMAN_AUTHORIZED_FOR_EXECUTION",
            "scientific_design_unchanged": True,
        },
        # ---- inherited unchanged from canonical policy ----
        "inherited": {
            "scientific_tabpfn_version": base["scientific_tabpfn_version"],
            "checkpoint": base["variance_pilot"]["checkpoint"],
            "feature_set": base["variance_pilot"]["feature_set"],
            "context_n": base["variance_pilot"]["context_n"],
            "context_seed": base["variance_pilot"]["context_seed"],
            "model_seed": base["variance_pilot"]["model_seed"],
            "scaler_arm": base["variance_pilot"]["scaler_arm"],
            "cells": base["variance_pilot"]["cells"],
            "query": base["variance_pilot"]["query"],
            "fits_per_cell": base["variance_pilot"]["fits_per_cell"],
            "main_inference_mode": base["variance_pilot"]["main_inference_mode"],
            "initial_replicates_per_cell": base["variance_pilot"][
                "initial_replicates_per_cell"
            ],
            "maximum_replicates_per_cell": base["variance_pilot"][
                "maximum_replicates_per_cell"
            ],
            "bounded_metric_ci_half_width_target": base["variance_pilot"][
                "bounded_metric_ci_half_width_target"
            ],
            "continuous_margin_ci_half_width_iqr_multiplier": base["variance_pilot"][
                "continuous_margin_ci_half_width_iqr_multiplier"
            ],
        },
        # ---- five supplementary decisions, human operator 2026-08-01 ----
        "supplementary_decisions": {
            "1_schedule_seed": {
                "schedule_seed": SCHEDULE_SEED,
                "generator": "numpy Generator(PCG64, seed=42)",
                "realised_cell_order": order,
                "note": "the realised order is frozen here, not only the seed",
            },
            "2_execution_lifecycle": {
                "cross_cell_repeat_interleaving": False,
                "rule": (
                    "seed 42 gives a one-time random order of the four cells; each "
                    "cell then runs to completion — exactly one fit, then its "
                    "same-process repeats — before the next cell starts"
                ),
                "reload_backfill_of_same_process_repeats": "forbidden",
            },
            "3_repeat_batches": {
                "batches": REPEAT_BATCHES,
                "increment": 8,
                "cap": 40,
            },
            "4_precision_gate_endpoints": {
                "gating": [
                    "steam_positive_vs_hotwater_negative_pairwise_auc",
                    "steam_positive_vs_hotwater_negative_continuous_score_margin",
                ],
                "evaluated_per_cell": True,
                "explicitly_non_gating": [
                    "chilledwater_secondary_readouts",
                    "global_rank",
                    "within_meter_rank",
                    "fresh_process_diagnostics",
                    "lifecycle_diagnostics",
                    "engineering_metrics",
                ],
            },
            "5_ci_half_width": {
                "method": "two-sided 95% Student-t on the repeat-level endpoint mean",
                "formula": "t.ppf(0.975, df=n-1) * std(ddof=1) / sqrt(n)",
                "forbidden": [
                    "normal_approximation",
                    "percentile_bootstrap",
                    "averaging_row_probabilities_then_scoring_once",
                ],
                "bounded_metric_pass": f"half_width <= {BOUNDED_HALF_WIDTH_TARGET}",
                "continuous_margin_pass": f"half_width <= {MARGIN_IQR_MULTIPLIER} * reference_IQR",
            },
        },
        "reference_iqr": {
            "definition": (
                "same cell, same query rows, fixed matched-tree comparator, "
                "steam-positive vs hotwater-negative, IQR of the pair-level "
                "score-margin distribution"
            ),
            "computed_before_any_tabpfn_fit": True,
            "not_derived_from_tabpfn_repeats": True,
            "per_cell": iqr,
        },
        "cells": manifests,
        "readouts": {
            "steam_primary": [
                "steam_positive_vs_hotwater_negative_pairwise_auc",
                "steam_positive_minus_hotwater_negative_score_margin",
                "steam_positive_global_rank",
                "steam_positive_within_meter_rank",
            ],
            "chilledwater_secondary_within_meter": [
                "chilledwater_positive_vs_chilledwater_negative_pairwise_auc",
                "chilledwater_positive_minus_chilledwater_negative_score_margin",
                "chilledwater_within_meter_pr_auc",
                "chilledwater_within_meter_roc_auc",
                "chilledwater_score_movement",
                "chilledwater_rank_movement",
            ],
            "reported_separately": ["tabpfn", "matched_tree", "tabpfn_minus_tree"],
            "score_and_rank_never_pooled": True,
            "onset_phase_contrast": {
                "status": "UNRESOLVED_NOT_EXECUTED",
                "reason": audit["phase_contrast"]["reason"],
                "consequence": (
                    "does not block the core variance pilot; query is not "
                    "modified, no rows are added, the 192-row query is not read"
                ),
            },
            "chilledwater_vs_hotwater_negative": {
                "status": "RESOLUTION_LIMITED_DIAGNOSTIC",
                "excluded_from_scientific_gate": True,
                "mechanism_bearing": False,
                "hotwater_negative_rows": 16,
                "valid_pairs": 1024,
                "auc_resolution": 9.765625e-4,
                "note": (
                    "about 4.4x resolution against a 0.0043 effect; a continuous "
                    "margin may not substitute for pairwise resolution"
                ),
            },
        },
        "cell_decision_rule": {
            "check_at": REPEAT_BATCHES,
            "pass": "both gate endpoints meet their half-width target",
            "escalate": "add 8 repeats",
            "cap_failure_status": "MEASUREMENT_UNSTABLE_AT_CAP",
        },
        "e3_decision_rule": {
            "E3_MEASUREMENT_PROCESS_ACCEPTABLE": "all four cells pass both gate endpoints",
            "E3_MEASUREMENT_PROCESS_UNSTABLE": "any cell still failing at n=40",
            "E3_MORE_REPEATS_REQUIRED": "predeclared continuation still pending under the cap",
            "E3_EXECUTION_INCOMPLETE": (
                "fit, state, same-process lifecycle or provenance incomplete"
            ),
        },
        "fresh_process_diagnostic": {
            "cell": FRESH_PROCESS_CELL,
            "runs": FRESH_PROCESS_RUNS,
            "when": "after that cell's same-process repeats and state save",
            "process": "independent fresh process per run, same persisted state",
            "query_and_schema": "identical to the same-process repeats",
            "kept_separate_from_same_process_statistics": True,
            "controls_repeat_count": False,
            "scientific_estimate": False,
            "hard_failure_conditions": [
                "state load failure",
                "version or digest mismatch",
                "row identity mismatch",
                "non-finite output",
                "schema error",
            ],
            "numerical_difference_policy": (
                "recorded and reported; no new pass/fail threshold is invented"
            ),
        },
        "trees": {
            "role": "matched_row_fixed_comparator",
            "refit": False,
            "tuned": False,
            "artificial_replicates": False,
            "scaler_arm": "cell_specific",
            "gap_rule": (
                "per repeat r: compute TabPFN metric Y[c,r] independently, then "
                "gap[c,r] = Y[c,r] - fixed tree metric[c]"
            ),
        },
        "prohibitions": [
            "E4 formal Path A",
            "Path B",
            "representation ablation",
            "frozen 192-row query",
            "site transfer",
            "500k",
            "full-holdout refit",
            "tree refit",
            "TabPFN 8.1.0 as science",
            "manuscript change",
            "changing N or cells",
            "24-cell grid",
            "undocumented reruns",
            "mixing fresh-process reload repeats into the same-process estimand",
            "ProcessPoolExecutor or parallel GPU workers",
        ],
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    body = json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    payload = json.dumps(
        {"protocol_sha256": digest, "protocol": protocol},
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    out = args.output_root / "e3_protocol.json"
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") != digest:
            raise SystemExit(
                f"refusing to overwrite a frozen protocol: "
                f"{existing.get('protocol_sha256')} != {digest}"
            )
        print(f"protocol already frozen and identical: {digest}")
        return 0
    out.write_text(payload, encoding="utf-8")
    print(f"froze E3 protocol -> {out}")
    print(f"protocol_sha256: {digest}")
    print(f"realised cell order: {order}")
    for c in order:
        r = iqr[c]
        print(
            f"  cell {c}: reference_IQR={r['reference_iqr']:.6f} "
            f"margin_target={r['margin_half_width_target']:.6f} "
            f"pairs={r['pairs']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
