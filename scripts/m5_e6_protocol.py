"""Freeze the M5 E6 natural-prevalence factorial protocol.

Runs only after the throughput gate has passed. Writes the full artifact set and
nothing else; if any input cannot be located or verified this fails rather than
writing a partial protocol.

Every census here is read from `e6_microbatch_manifest.json`, never divided out
of a row count -- that was the defect the audit correction fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(r"C:\Users\tonykuo\projects\lead-reproduction")
E4_ROOT = ROOT / "data" / "processed" / "m5_e4_formal_path_a"
E5_ROOT = ROOT / "data" / "processed" / "m5_e5_independent_replication"
FACTORIAL = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"
SCREENING = ROOT / "data/processed/m5_context_stories/queries/screening"

ROWS = 10_137_155
STATES = 24
CONTEXT_SEEDS = [42, 123, 999]
SCALER_ARMS = ["cell_specific", "frozen_reference"]
CELLS = {
    "11": "hw_pos_present__hw_neg_present",
    "10": "hw_pos_present__hw_neg_excluded",
    "01": "hw_pos_excluded__hw_neg_present",
    "00": "hw_pos_excluded__hw_neg_excluded",
}
E4_ORDER_DIGEST = "63ca76f1167768252b29992fd791c450ba33447f5908b8938f1b67d0ecc732e3"
HOLDOUT_DIGEST = "f0867d3e86ae2b017ea6fee2d1b9f6dead2ee241948346a467ea06305e220e76"

SOURCE_FILES = [
    "scripts/m5_e6_protocol.py",
    "scripts/m5_e6_microbatch.py",
    "scripts/m5_e6_row_audit.py",
    "scripts/m5_e6_clustered.py",
    "scripts/m5_e6_probe.py",
    "scripts/m5_e6_features.py",
    "scripts/m5_e6_preflight.py",
    "scripts/m5_e6_runner.py",
    "scripts/m5_e6_trees.py",
    "scripts/m5_e6_analysis.py",
    "scripts/m5_e6_decision.py",
    "scripts/m5_e5_guard.py",
    "scripts/m5_e4_endpoints.py",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_source(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def atomic_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if sha256_file(path) != digest:
        raise AssertionError(f"{path.name} does not match its body digest")
    return digest


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def state_manifest() -> list[dict]:
    specs = read_json(E5_ROOT / "e5_state_manifest.json")["states"]
    out = []
    for s in specs:
        state = ROOT / s["state_path"]
        digest = sha256_file(state)
        if digest != s["state_sha256"]:
            raise SystemExit(f"{s['unit_id']}: state digest drifted")
        complete = read_json(E4_ROOT / s["unit_id"] / "FIT_COMPLETE.json")
        if complete["ensemble"]["effective_n_estimators_"] != 8:
            raise SystemExit(f"{s['unit_id']}: effective n_estimators_ is not 8")
        out.append(
            {
                "position": s["position"],
                "unit_id": s["unit_id"],
                "context_seed": s["context_seed"],
                "cell": s["cell"],
                "cell_dir": s["cell_dir"],
                "scaler_arm": s["scaler_arm"],
                "state_path": s["state_path"],
                "state_sha256": digest,
                "context_manifest": s["context_manifest"],
                "context_manifest_sha256": s["context_manifest_sha256"],
                "tree_comparator": s["tree_comparator"],
                "tree_comparator_sha256": s["tree_comparator_sha256"],
                "scaler_source": s["scaler_source"],
            }
        )
    order = hashlib.sha256(
        "\n".join(u["unit_id"] for u in out).encode("utf-8")
    ).hexdigest()
    if order != E4_ORDER_DIGEST:
        raise SystemExit("state order does not reproduce E4's frozen order digest")
    if len(out) != STATES:
        raise SystemExit(f"expected {STATES} states, built {len(out)}")
    return out


def sentinel_manifest() -> dict:
    npz = SCREENING / "queries.npz"
    qm = read_json(SCREENING / "manifest.json")
    return {
        "schema": "m5_e6_sentinel_manifest_v1",
        "purpose": "reload-time inference variability, environment and runtime "
        "drift, and whether both sit inside the range E4 and E5 established",
        "query": "original 352-row screening query",
        "path": rel(SCREENING),
        "npz_sha256": sha256_file(npz),
        "raw_index_sha256": qm["raw_index_sha256"],
        "rows": 352,
        "repeats_per_state": 8,
        "states": STATES,
        "total_calls": 8 * STATES,
        "total_row_scores": 352 * 8 * STATES,
        "is_a_full_holdout_repeat": False,
        "may_enter_any_endpoint": False,
        "stored_separately_from_holdout_scores": True,
        "e4_e5_reference_half_widths": {
            "e5_steam_auc_half_width_range": [0.000415, 0.009790],
            "e5_steam_margin_half_width_range": [0.000334, 0.006440],
        },
    }


def tree_manifest(states: list[dict]) -> dict:
    recs = []
    for s in states:
        d = (
            FACTORIAL
            / "recovery"
            / "states"
            / "trees"
            / f"seed{s['context_seed']}"
            / s["cell_dir"]
            / s["scaler_arm"]
        )
        ens, sc = d / "tree_ensemble.joblib", d / "scaler.joblib"
        if not (ens.exists() and sc.exists()):
            raise SystemExit(f"{s['unit_id']}: tree state missing")
        recs.append(
            {
                "unit_id": s["unit_id"],
                "tree_ensemble": rel(ens),
                "tree_ensemble_sha256": sha256_file(ens),
                "tree_scaler": rel(sc),
                "tree_scaler_sha256": sha256_file(sc),
                "e4_comparator": s["tree_comparator"],
                "e4_comparator_sha256": s["tree_comparator_sha256"],
            }
        )
    return {
        "schema": "m5_e6_tree_manifest_v1",
        "execution_host": "original laptop environment",
        "gpu_host_tree_outputs_prohibited": True,
        "refit": False,
        "artificial_replicates": False,
        "identity_gate": {
            "query": "E4 frozen 352-row comparator",
            "units_required": STATES,
            "max_abs_diff_required": 0.0,
            "exact_rows_required": "352/352",
            "run_before_and_after_holdout_scoring": True,
            "sampling_not_permitted": True,
            "tolerance_not_permitted": True,
        },
        "environment_contract": {
            "platform": "Windows-11-10.0.26200-SP0",
            "python": "3.13.13",
            "lightgbm": "4.6.0",
            "xgboost": "3.2.0",
            "catboost": "1.2.10",
            "sklearn": "1.8.0",
            "numpy": "2.4.6",
            "joblib": "1.5.3",
        },
        "units": recs,
    }


def bootstrap_manifest() -> dict:
    return {
        "schema": "m5_e6_bootstrap_manifest_v1",
        "master_seed": 20260730,
        "namespace_code": 6006,
        "cluster_code": {"building": 1, "segment": 2},
        "construction": "np.random.SeedSequence([20260730, 6006, "
        "cluster_code[cluster_type], draw_id]) then PCG64 Generator",
        "draws": 1000,
        "draw_ids": [0, 999],
        "interval": "percentile q025 / q975",
        "co_primary_subset": {
            "steam_positive_rows": 48_888,
            "hotwater_negative_rows": 545_430,
            "total_rows": 594_318,
            "building_clusters": 215,
            "segment_clusters": 594_297,
            "segment_singleton_clusters": 545_430,
            "segment_singleton_fraction": 545_430 / 594_297,
        },
        "auc_estimator": "exact cluster-weighted Mann-Whitney; sort once per "
        "unit, O(n) weighted sweep per draw; ties contribute 0.5",
        "margin_estimator": "exact cluster-weighted sufficient statistics",
        "approximate_auc": "forbidden",
        "verified_against_naive_resampling": {
            "synthetic_with_ties": True,
            "real_e5_192_row_data": True,
            "tolerance": "1e-12 relative",
        },
        "shared_within_a_draw": [
            "all cells",
            "all arms",
            "all context seeds",
            "TabPFN",
            "tree",
            "all co-primary endpoints",
        ],
        "segment_degeneracy_disclosure_required": (
            "the co-primary subset has 594,297 segment clusters of which 545,430 "
            "hotwater-negative rows are singletons, 91.8% of the total; the "
            "negative side therefore behaves close to a row bootstrap and the "
            "segment interval must not be described as cluster-level "
            "corroboration equal in independence to the building interval"
        ),
        "segment_may_not_be_redefined": True,
        "segment_may_not_become_site_cluster": True,
        "segment_may_not_be_removed_from_the_decision_rule": True,
    }


def decision_rules() -> dict:
    return {
        "schema": "m5_e6_decision_rules_v1",
        "vocabulary": [
            "NATURAL_PREVALENCE_CONFIRMED",
            "DIRECTIONALLY_CONSISTENT_BUT_INCONCLUSIVE",
            "NOT_CONFIRMED",
            "EXECUTION_INCOMPLETE",
        ],
        "primary_contrast": "negative_support_main_effect",
        "co_primary_endpoints": [
            "steam_positive_vs_hotwater_negative_pairwise_auc",
            "steam_positive_minus_hotwater_negative_score_margin",
        ],
        "response_confirmation_conditions": [
            "AUC overall > 0",
            "margin overall > 0",
            "AUC and margin both positive in 3/3 context seeds",
            "both scaler arms positive",
            "building interval excludes zero on both endpoints",
            "segment interval excludes zero on both endpoints",
        ],
        "tabpfn_specific_conditions": "the TabPFN-minus-tree gap meets every "
        "condition above on both endpoints",
        "minimum_practical_effect_threshold": "NOT SET",
        "mandatory_comparison_columns": [
            "E4 effect",
            "E5 effect",
            "E6 effect",
            "E6/E5 ratio",
            "E6 minus E5 absolute difference",
        ],
        "magnitude_claim_rule": "an interval excluding zero does not license "
        "the claim that the magnitude reproduced; discrepant values must be "
        "reported as discrepant",
        "interaction": "secondary; AUC and margin must be reported together and "
        "neither may be selected alone",
        "thresholds_may_not_change_after_seeing_results": True,
        "interval_containing_zero_is_not_proof_of_absence": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--base-commit", required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--feature-sha256", required=True)
    args = ap.parse_args()

    probe = read_json(args.probe)
    if not probe["gate"]["passed"]:
        raise SystemExit("the throughput gate did not pass; refusing to freeze")

    mb = read_json(args.out / "e6_microbatch_manifest.json")
    row = read_json(args.out / "e6_row_manifest.json")
    if row["row_set"]["sorted_raw_index_sha256"] != HOLDOUT_DIGEST:
        raise SystemExit("row manifest does not match the frozen holdout digest")

    states = state_manifest()
    sent = sentinel_manifest()
    trees = tree_manifest(states)
    boot = bootstrap_manifest()
    rules = decision_rules()

    sources = {}
    for r in SOURCE_FILES:
        p = args.repo / r
        if not p.exists():
            raise SystemExit(f"source file missing, cannot freeze: {r}")
        sources[r] = sha256_source(p)

    census = mb["census"]
    protocol = {
        "schema": "m5_e6_natural_prevalence_protocol_v1",
        "execution_mode": "E6_NATURAL_PREVALENCE_FACTORIAL_CONFIRMATION",
        "base_commit": args.base_commit,
        "e5_completion_commit": "13f92a099b02213f6bcb4b7e178c23af450bdf85",
        "design_audit_commit": "dc7445fa57ac33fb202a8b7367b9294baf4c7c46",
        "authorization": {
            "execution_status": "HUMAN_AUTHORIZED_FOR_EXECUTION",
            "human_authorization_date": "2026-08-02",
            "authorized_scope": "E6 natural-prevalence factorial confirmation only",
        },
        "question": (
            "does the steam negative-support response established in E4 and "
            "independently replicated in E5 still hold on the complete "
            "odd-building holdout at natural prevalence?"
        ),
        "required_wording": (
            "E6 is a natural-prevalence factorial confirmation using new "
            "factorial states on previously characterised holdout rows"
        ),
        "forbidden_wording": [
            "untouched replication",
            "first contact with the holdout",
            "previously unseen row set",
        ],
        "mechanism_limitation": (
            "cell 00 removes hotwater support and simultaneously flips the meter "
            "feature from numerical to categorical, so E6 confirms the "
            "negative-support intervention as a whole and may not be described "
            "as having isolated the hotwater-normal reference as the sole "
            "mechanism"
        ),
        "repeat_policy": {
            "name": "R1_PLUS_SENTINEL",
            "full_holdout_passes_per_state": 1,
            "sentinel_repeats_per_state": 8,
            "eight_full_holdout_passes": "forbidden",
            "sentinel_may_enter_endpoints": False,
        },
        "estimand": {
            "name": "canonical single-process batched pass",
            "definition": [
                "one persisted state",
                "one fresh process UUID",
                "a fixed microbatch partition in a fixed order",
                "one predict_proba per microbatch",
                "every holdout row scored exactly once",
                "all microbatches completed inside that one process",
                "the state-level score field formed only after all complete",
            ],
            "not_called": [
                "one inference realization",
                "one predict_proba realization",
                "one full-vector repeat",
            ],
            "interval_conditioning": "clustered intervals are conditional on the "
            "realized batched pass and do not cover same-state full-holdout "
            "inference-repeat variation",
            "differs_from_e4_e5": "E4 and E5 averaged 8 repeats per fit before "
            "forming contrasts; E6 does not",
        },
        "grid": {
            "context_seeds": CONTEXT_SEEDS,
            "cells": {c: CELLS[c] for c in ("11", "10", "01", "00")},
            "scaler_arms": SCALER_ARMS,
            "states": STATES,
            "model_seed": 42,
        },
        "tabpfn": {
            "version": "8.0.8",
            "requested_n_estimators": 8,
            "auto_scale_n_estimators": False,
            "required_effective_n_estimators_": 8,
            "mismatch_policy": "hard_failure",
            "checkpoint_sha256": "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988",
            "cpu_fallback": "prohibited",
        },
        "execution_order": {
            "inherited_from_e4": True,
            "realised_order_digest": E4_ORDER_DIGEST,
            "reshuffling": "forbidden",
            "units": [
                {"position": s["position"], "unit_id": s["unit_id"]} for s in states
            ],
        },
        "holdout": {
            "rows": ROWS,
            "sorted_raw_index_sha256": HOLDOUT_DIGEST,
            "buildings": 724,
            "sites": 16,
            "anomaly_rows": 637_397,
            "natural_prevalence": row["row_set"]["natural_prevalence"],
            "order": "canonical stored order, not sorted by raw_index",
        },
        "feature_artifact": {
            "shape": [ROWS, 137],
            "dtype": "float32",
            "sha256": args.feature_sha256,
            "built_once_shared_by_both_hosts": True,
            "float64_upcast": "hard_failure",
            "not_the_context100000_scaled_artifact": True,
        },
        "census": census,
        "machine_topology": {
            "tabpfn_host": "gpu-host, single GPU worker, single tmux session",
            "multi_machine_tabpfn": "forbidden",
            "state_based_machine_assignment": "forbidden",
            "process_pool_executor": "forbidden",
            "tree_host": "original laptop environment",
            "gpu_host_tree_outputs": "forbidden",
        },
        "resume_and_fail_closed": {
            "microbatch_atomic_write": True,
            "microbatch_completion_is_not_scientific_completion": True,
            "state_completion_requires": [
                "all expected microbatches complete",
                "row union equals the full holdout",
                "no missing or duplicate rows",
                "a single process UUID",
                "one state digest throughout",
                "scaler verification complete",
                "all outputs finite",
                "sentinel complete",
                "atomic completion marker",
            ],
            "on_interruption": [
                "write INTERRUPTED_INCOMPLETE",
                "stop the whole run",
                "do not resume from the next microbatch",
                "do not splice new-process output onto old-process output",
                "quarantine the entire partial state",
                "restart that state from canonical row 0",
                "completed states may be skipped",
            ],
            "maximum_recomputation_on_failure": "one complete state pass",
        },
        "prohibitions": [
            "any fit or refit",
            "a second full-holdout pass",
            "Path B",
            "representation ablation",
            "500k",
            "site transfer",
            "TabPFN 8.1.0 as science",
            "manuscript change",
            "changing states, seeds, cells, arms or endpoints",
            "gpu-host tree outputs",
            "multi-machine TabPFN",
            "pooling meters into one primary AUC",
            "site-transfer claims",
            "best seed, best arm or best site selection",
        ],
        "source_digest_method": "sha256 over newline-normalised content",
        "source_digests": sources,
    }

    proto_sha = atomic_json(args.out / "e6_protocol.json", {"protocol": protocol})
    state_sha = atomic_json(args.out / "e6_state_manifest.json", {"states": states})
    sent_sha = atomic_json(args.out / "e6_sentinel_manifest.json", sent)
    tree_sha = atomic_json(args.out / "e6_tree_manifest.json", trees)
    boot_sha = atomic_json(args.out / "e6_bootstrap_manifest.json", boot)
    rules_sha = atomic_json(args.out / "e6_decision_rules.json", rules)

    inputs = {
        "schema": "m5_e6_input_manifest_v1",
        "generated": time.time(),
        "base_commit": args.base_commit,
        "protocol_sha256": proto_sha,
        "row_manifest_sha256": sha256_file(args.out / "e6_row_manifest.json"),
        "shard_manifest_sha256": sha256_file(args.out / "e6_shard_manifest.json"),
        "microbatch_manifest_sha256": sha256_file(
            args.out / "e6_microbatch_manifest.json"
        ),
        "state_manifest_sha256": state_sha,
        "sentinel_manifest_sha256": sent_sha,
        "tree_manifest_sha256": tree_sha,
        "bootstrap_manifest_sha256": boot_sha,
        "decision_rules_sha256": rules_sha,
        "throughput_probe_sha256": sha256_file(args.probe),
        "cost_model_sha256": sha256_file(args.out / "e6_cost_model.json"),
        "feature_artifact_sha256": args.feature_sha256,
        "e4_state_digests": {s["unit_id"]: s["state_sha256"] for s in states},
        "tree_ensembles": {
            u["unit_id"]: u["tree_ensemble_sha256"] for u in trees["units"]
        },
        "source_digests": sources,
    }
    inputs_sha = atomic_json(args.out / "e6_input_manifest.json", inputs)

    print(f"protocol         sha256 = {proto_sha}")
    print(f"input manifest   sha256 = {inputs_sha}")
    print(f"state manifest   sha256 = {state_sha}   ({len(states)} states)")
    print(f"sentinel         sha256 = {sent_sha}")
    print(f"tree manifest    sha256 = {tree_sha}   ({len(trees['units'])} units)")
    print(f"bootstrap        sha256 = {boot_sha}")
    print(f"decision rules   sha256 = {rules_sha}")
    print(f"execution order digest  = {E4_ORDER_DIGEST}")
    print(f"microbatches per state  = {census['microbatches_per_state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
