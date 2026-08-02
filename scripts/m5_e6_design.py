"""Build the E6 design artifacts: shard plan, cost model, and the protocol draft.

Every number here is either read from an existing artifact or measured; nothing
is extrapolated from E5's 192-row inference time, which is dominated by fixed
overhead. Where a quantity could not be measured this round without scoring
holdout rows, it is recorded as an open item rather than guessed.

Writes to a DRAFT directory. This is not a launch artifact and no protocol SHA
produced here may be used to launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(r"C:\Users\tonykuo\projects\lead-reproduction")

ROWS = 10_137_155
STATES = 24
FEATURES = 137

# Measured, not assumed. Sources are named beside each value.
MEASURED = {
    "tabpfn_rows_per_second_steady": {
        "value": 330.0,
        "source": "m5_tabpfn_137_batch_runner.log steady state (second half of "
        "the run) and throughput_rows_per_second_per_gpu recorded in "
        "m5_tabpfn_137_remaining_batch_plan.json",
        "config": "137 features, context 100000, gpu-host, microbatch 20000",
    },
    "tabpfn_rows_per_second_all_in": {
        "value": 82.4,
        "source": "same log, whole span including warm-up and stalls",
        "config": "same",
    },
    "tree_rows_per_second": {
        "value": 309_483,
        "source": "synthetic-input benchmark on the laptop, 500,000 random "
        "float32 rows through the four-model ensemble; no real holdout row was "
        "scored",
        "config": "lightgbm 4.6.0, xgboost 3.2.0, catboost 1.2.10, sklearn 1.8.0",
    },
    "weighted_auc_sweep_seconds": {
        "value": 0.023,
        "source": "measured on the real 594,318-row co-primary subset with "
        "synthetic scores",
        "config": "sort once per unit, O(n) weighted sweep per draw",
    },
}

STRATA = {
    "rows": ROWS,
    "by_meter": {
        "electricity": {
            "rows": 6_035_071,
            "anomaly": 356_679,
            "buildings": 709,
            "sites": 16,
        },
        "chilledwater": {
            "rows": 2_115_354,
            "anomaly": 141_139,
            "buildings": 252,
            "sites": 10,
        },
        "steam": {"rows": 1_350_609, "anomaly": 48_888, "buildings": 162, "sites": 6},
        "hotwater": {"rows": 636_121, "anomaly": 90_691, "buildings": 73, "sites": 7},
    },
    "co_primary": {
        "steam_positive_rows": 48_888,
        "steam_positive_buildings": 150,
        "steam_positive_sites": 6,
        "hotwater_negative_rows": 545_430,
        "hotwater_negative_buildings": 73,
        "hotwater_negative_sites": 7,
        "subset_rows": 594_318,
        "subset_fraction_of_holdout": 594_318 / ROWS,
        "subset_buildings": 215,
        "building_clusters": 215,
        "segment_clusters": 594_297,
        "segment_singleton_clusters": 545_430,
        "segment_singleton_fraction": 545_430 / 594_297,
        "notional_pair_count": 48_888 * 545_430,
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def shard_plan(shards: int = 12) -> dict:
    """Row shards over the canonical order, each scored by all 24 states.

    Sharding is by row, never by state: a machine that scored only some states
    would make "machine" confounded with every state contrast, which is exactly
    what E4 avoided by staying on one host.
    """
    base, extra = divmod(ROWS, shards)
    plan, start = [], 0
    for i in range(shards):
        n = base + (1 if i < extra else 0)
        plan.append(
            {
                "shard": i,
                "canonical_position_start": start,
                "canonical_position_end": start + n,
                "rows": n,
                "states_to_score": STATES,
                "may_be_split_by_state": False,
            }
        )
        start += n
    assert start == ROWS
    return {
        "schema": "m5_e6_shard_manifest_v1",
        "shards": shards,
        "total_rows": ROWS,
        "union_equals_holdout": True,
        "pairwise_disjoint": True,
        "order": "canonical stored order of the full-test artifacts, not sorted "
        "raw_index",
        "rule": "every shard is scored by all 24 states; state-based machine "
        "assignment is forbidden",
        "plan": plan,
    }


def cost_model() -> dict:
    rps = MEASURED["tabpfn_rows_per_second_steady"]["value"]
    rps_lo = MEASURED["tabpfn_rows_per_second_all_in"]["value"]
    one_pass_h = ROWS / rps / 3600
    one_pass_h_lo = ROWS / rps_lo / 3600
    tree_h = STATES * ROWS / MEASURED["tree_rows_per_second"]["value"] / 3600
    f32 = 4

    def policy(name, passes, sentinel=False):
        calls = STATES * passes * (ROWS / 20_000)  # microbatched predict_proba calls
        return {
            "policy": name,
            "full_holdout_passes_per_state": passes,
            "predict_proba_calls_full_holdout": int(round(calls)),
            "row_scores": STATES * passes * ROWS,
            "wall_clock_hours_steady": STATES * passes * one_pass_h
            + (0.02 if sentinel else 0.0),
            "wall_clock_hours_all_in": STATES * passes * one_pass_h_lo
            + (0.02 if sentinel else 0.0),
            "wall_clock_days_steady": STATES * passes * one_pass_h / 24,
            "output_bytes_float32": STATES * passes * ROWS * f32,
            "output_gb_float32": STATES * passes * ROWS * f32 / 1e9,
            "sentinel_repeats_per_state": 8 if sentinel else 0,
            "sentinel_rows": 352 if sentinel else 0,
        }

    return {
        "schema": "m5_e6_cost_model_v1",
        "generated": time.time(),
        "basis": "measured, not extrapolated from E5's 192-row timing",
        "measured_inputs": MEASURED,
        "strata": STRATA,
        "one_state_full_pass_hours_steady": one_pass_h,
        "one_state_full_pass_hours_all_in": one_pass_h_lo,
        "context_caveat": (
            "the 330 rows/s anchor was measured at context 100000; E6 uses "
            "context 20000, so this is an upper bound on time and the true E6 "
            "rate is unmeasured. Pinning it needs a throughput probe, which can "
            "run on replicated non-holdout rows and produce no holdout "
            "prediction."
        ),
        "repeat_policies": [
            policy("R1", 1),
            policy("R8", 8),
            policy("R1_PLUS_SENTINEL", 1, sentinel=True),
        ],
        "tree_full_test_hours": tree_h,
        "tree_note": "24 comparators over the whole holdout is about 13 minutes; "
        "the tree half is not a cost driver",
        "raw_feature_rebuild": {
            "required": True,
            "reason": "the existing 5.3 GB distributed feature artifacts are "
            "already scaled with the context-100000 scaler (the meter column "
            "holds -0.7625/0.3011/1.3647/2.4283, not 0/1/2/3), so they cannot "
            "carry E6's 24 per-unit scalers",
            "output_bytes_float32": ROWS * FEATURES * f32,
            "output_gb_float32": ROWS * FEATURES * f32 / 1e9,
            "built_once_shared_by_all_24_states": True,
        },
        "clustered_uncertainty_hours": 1000
        * 2
        * STATES
        * MEASURED["weighted_auc_sweep_seconds"]["value"]
        / 3600,
        "restart_cost": {
            "checkpoint_rows": 20_000,
            "max_recompute_rows_on_failure": 20_000,
            "max_recompute_seconds_steady": 20_000 / rps,
            "note": "per-microbatch atomic checkpoints bound a failure to one "
            "microbatch, as the 137-feature run already demonstrated",
        },
    }


def protocol_draft(row_manifest_sha: str, shard_sha: str, cost_sha: str) -> dict:
    return {
        "schema": "m5_e6_protocol_DRAFT_v1",
        "status": "DRAFT — NOT FROZEN, NOT LAUNCHABLE",
        "not_a_launch_artifact": True,
        "generated": time.time(),
        "base_commit": "13f92a099b02213f6bcb4b7e178c23af450bdf85",
        "question": (
            "does the steam negative-support response established in E4 and "
            "independently replicated in E5 still hold on the complete "
            "odd-building holdout at natural prevalence?"
        ),
        "verdicts": {
            "A": "natural_prevalence_response_confirmation",
            "B": "natural_prevalence_tabpfn_specific_confirmation",
        },
        "primary_contrast": "negative_support_main_effect",
        "co_primary_endpoints": [
            "steam_positive_vs_hotwater_negative_pairwise_auc",
            "steam_positive_minus_hotwater_negative_score_margin",
        ],
        "inherited_unchanged": {
            "factorial_cell_coding": "E4/E5 exact",
            "effect_formulas_and_signs": "E4/E5 exact",
            "context_seeds": [42, 123, 999],
            "scaler_arms": ["cell_specific", "frozen_reference"],
            "persisted_states": 24,
            "tree_comparators": "matched fixed, no refit",
            "seed_aggregation": "seed-specific contrast first, then equal weight "
            "over three seeds",
            "cluster_definitions": "building and segment, as in E4/E5",
            "row_probability_averaging": "forbidden",
            "cross_cell_repeat_pairing": "forbidden",
            "model_seed_factor": "not added",
        },
        "row_manifest_sha256": row_manifest_sha,
        "shard_manifest_sha256": shard_sha,
        "cost_model_sha256": cost_sha,
        "clustered_uncertainty": {
            "master_seed": 20260730,
            "namespace_code": 6006,
            "cluster_code": {"building": 1, "segment": 2},
            "draws": 1000,
            "interval": "percentile q025 / q975",
            "shared_draw_across": [
                "cells",
                "arms",
                "seeds",
                "TabPFN",
                "tree",
                "all endpoints",
            ],
            "auc_method": "exact cluster-weighted Mann-Whitney: sort the "
            "co-primary subset once per unit, then sweep weighted counts per "
            "draw; equals the naive resampling estimator by construction and is "
            "required to be verified against it on synthetic data and on the "
            "E5 192-row data",
            "margin_method": "cluster-weighted sufficient statistics (per-cluster "
            "score sums and counts), exact",
        },
        "decision_rule_candidate": {
            "vocabulary": [
                "NATURAL_PREVALENCE_CONFIRMED",
                "DIRECTIONALLY_CONSISTENT_BUT_INCONCLUSIVE",
                "NOT_CONFIRMED",
                "EXECUTION_INCOMPLETE",
            ],
            "confirmed_conditions": [
                "AUC overall > 0",
                "margin overall > 0",
                "3/3 seeds positive on both endpoints",
                "both scaler arms positive on both endpoints",
                "building interval excludes zero on both endpoints",
                "segment interval excludes zero on both endpoints",
            ],
            "tabpfn_specific": "the TabPFN-minus-tree gap meets the same conditions",
            "minimum_practical_effect_threshold": "NOT SET — see the open items",
        },
        "wording_constraints": {
            "required": "natural-prevalence factorial confirmation using new "
            "factorial states on previously characterised holdout rows",
            "forbidden": [
                "untouched holdout",
                "first contact",
                "previously unseen row set",
            ],
            "mechanism": "E6 confirms the negative-support intervention as a "
            "whole; because cell 00 also flips the meter feature's modality, it "
            "may not be described as having isolated the hotwater-normal "
            "reference as the sole mechanism",
        },
        "open_items_requiring_human_ruling": [
            "repeat policy (R1 / R8 / R1_PLUS_SENTINEL)",
            "segment clustering, which is 91.8% singletons on the co-primary "
            "subset at natural prevalence and therefore behaves close to a row "
            "bootstrap",
            "whether to freeze a minimum practical effect threshold",
            "whether to run a non-holdout throughput probe to pin the "
            "context-20000 rate",
            "machine topology if more than one GPU host is used",
        ],
        "prohibitions_this_round": [
            "any predict on the 10,137,155-row holdout",
            "any fit or refit",
            "protocol freeze",
            "launch",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=12)
    args = ap.parse_args()

    row_sha = sha256_file(args.out / "e6_row_manifest.json")
    shard_sha = atomic_json(
        args.out / "e6_shard_manifest.json", shard_plan(args.shards)
    )
    cost = cost_model()
    cost_sha = atomic_json(args.out / "e6_cost_model.json", cost)
    draft_sha = atomic_json(
        args.out / "e6_protocol.DRAFT.json",
        protocol_draft(row_sha, shard_sha, cost_sha),
    )

    print(f"row manifest    sha256 = {row_sha}")
    print(f"shard manifest  sha256 = {shard_sha}   ({args.shards} shards)")
    print(f"cost model      sha256 = {cost_sha}")
    print(f"protocol DRAFT  sha256 = {draft_sha}   (NOT a launch artifact)")
    print()
    for p in cost["repeat_policies"]:
        print(
            f"  {p['policy']:<18} calls={p['predict_proba_calls_full_holdout']:>9,} "
            f"scores={p['row_scores']:>14,} "
            f"steady={p['wall_clock_hours_steady']:>8.1f} h "
            f"({p['wall_clock_days_steady']:>5.1f} d) "
            f"out={p['output_gb_float32']:>6.2f} GB"
        )
    print(
        f"\n  one state full pass : {cost['one_state_full_pass_hours_steady']:.2f} h "
        f"(steady) / {cost['one_state_full_pass_hours_all_in']:.2f} h (all-in)"
    )
    print(f"  tree, 24 comparators: {cost['tree_full_test_hours']:.2f} h")
    print(
        f"  raw feature rebuild : {cost['raw_feature_rebuild']['output_gb_float32']:.2f} GB"
    )
    print(f"  clustered intervals : {cost['clustered_uncertainty_hours']:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
