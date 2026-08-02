"""Freeze the M5 E5 independent replication protocol before any scoring.

E5 is pure re-scoring: it reloads E4's 24 persisted states and scores the frozen
192-row independent query. Nothing is fitted. The protocol records that as a
hard constraint, along with the pre-declared decision rules, so the verdict
cannot be chosen after seeing the numbers.

Five artifacts are written and nothing else. If any input cannot be located or
verified, this fails rather than writing a partial protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

CELLS = {
    "11": "hw_pos_present__hw_neg_present",
    "10": "hw_pos_present__hw_neg_excluded",
    "01": "hw_pos_excluded__hw_neg_present",
    "00": "hw_pos_excluded__hw_neg_excluded",
}
CONTEXT_SEEDS = [42, 123, 999]
SCALER_ARMS = ["cell_specific", "frozen_reference"]
REPEATS = 8

ROOT = Path(r"C:\Users\tonykuo\projects\lead-reproduction")
E4_ROOT = ROOT / "data" / "processed" / "m5_e4_formal_path_a"
FACTORIAL_ROOT = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"
QUERY_DIR = FACTORIAL_ROOT / "independent_query"
SCREENING = ROOT / "data" / "processed" / "m5_context_stories" / "queries" / "screening"

QUERY_NPZ_SHA = "d780f0f8a96c47f49ffe061a72906728f1301056555350cabd979348aa41a2a0"
QUERY_RAW_SHA = "2fc4a638a2a0880f2b4d7feac87875c941d155f5fe5172b75b13d041b654fa16"
E4_REALISED_ORDER_DIGEST = (
    "63ca76f1167768252b29992fd791c450ba33447f5908b8938f1b67d0ecc732e3"
)

SOURCE_FILES = [
    "scripts/m5_e5_protocol.py",
    "scripts/m5_e5_runner.py",
    "scripts/m5_e5_guard.py",
    "scripts/m5_e5_analysis.py",
    "scripts/m5_e5_decision.py",
    "scripts/m5_e4_endpoints.py",
    "scripts/m5_e4_clustered.py",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_source(path: Path) -> str:
    """Newline-normalised, so a Windows checkout and a Linux clone agree."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def atomic_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if sha256_file(path) != digest:
        raise AssertionError(f"{path.name} on disk does not match its body digest")
    return digest


def query_audit() -> dict:
    """Identity of the 192-row query, checked against the frozen digests."""
    npz = QUERY_DIR / "queries.npz"
    if sha256_file(npz) != QUERY_NPZ_SHA:
        raise SystemExit("192-row queries.npz does not match the frozen digest")
    with np.load(npz, allow_pickle=True) as z:
        raw = np.asarray(z["raw_index"], dtype="int64")
        meter = np.asarray(z["meter"], dtype="int8")
        anomaly = np.asarray(z["anomaly"], dtype="int8")
        building = np.asarray(z["building_id"], dtype="int64")
        stratum = np.asarray(z["stratum"])
        segment = np.asarray(z["segment_id"])
    if hashlib.sha256(raw.tobytes()).hexdigest() != QUERY_RAW_SHA:
        raise SystemExit("192-row raw_index does not match the frozen digest")
    if raw.size != 192 or np.unique(raw).size != 192:
        raise SystemExit("the 192-row query is not 192 unique rows")

    strata = {s: int((stratum == s).sum()) for s in sorted(set(stratum.tolist()))}
    if strata != {"hw01_negative": 64, "hw01_positive": 64, "steam_positive": 64}:
        raise SystemExit(f"unexpected strata: {strata}")

    with np.load(SCREENING / "queries.npz") as z:
        raw352 = np.asarray(z["raw_index"], dtype="int64")
    overlap = int(np.intersect1d(raw, raw352).size)
    if overlap:
        raise SystemExit(f"the two queries overlap in {overlap} rows")

    # The endpoint code selects hotwater negatives as meter==3 & anomaly==0.
    # On this query that must be exactly the hw01_negative stratum, or the
    # co-primary endpoints would silently mean something else than in E4.
    hw_neg = (meter == 3) & (anomaly == 0)
    steam_pos = (meter == 2) & (anomaly == 1)
    if not np.array_equal(hw_neg, stratum == "hw01_negative"):
        raise SystemExit("meter==3 & anomaly==0 is not the hw01_negative stratum")
    if not np.array_equal(steam_pos, stratum == "steam_positive"):
        raise SystemExit("meter==2 & anomaly==1 is not the steam_positive stratum")

    return {
        "path": rel(QUERY_DIR),
        "queries_npz_sha256": QUERY_NPZ_SHA,
        "manifest_sha256": sha256_file(QUERY_DIR / "manifest.json"),
        "raw_index_sha256": QUERY_RAW_SHA,
        "rows": 192,
        "unique_rows": 192,
        "strata": strata,
        "overlap_with_screening_query": overlap,
        "meter_counts": {
            str(k): int(v) for k, v in zip(*np.unique(meter, return_counts=True))
        },
        "anomaly_counts": {
            str(k): int(v) for k, v in zip(*np.unique(anomaly, return_counts=True))
        },
        "buildings": int(np.unique(building).size),
        "segments": int(np.unique(segment).size),
        "endpoint_selector_check": {
            "steam_positive_is_meter2_anomaly1": True,
            "hw01_negative_is_meter3_anomaly0": True,
        },
        "chilledwater_rows": int((meter == 1).sum()),
        "rows_may_not_be_added_removed_replaced_or_resampled": True,
    }


def state_manifest() -> list[dict]:
    """The 24 E4 states, in E4's frozen execution order, by digest."""
    e4 = json.loads((E4_ROOT / "e4_protocol.json").read_text(encoding="utf-8"))[
        "protocol"
    ]
    if e4["schedule"]["realised_order_digest"] != E4_REALISED_ORDER_DIGEST:
        raise SystemExit("E4 realised order digest does not match the frozen value")
    specs = json.loads((E4_ROOT / "e4_fit_manifest.json").read_text(encoding="utf-8"))[
        "fits"
    ]
    units = []
    for spec in specs:
        uid = spec["unit_id"]
        state = E4_ROOT / uid / "model.tabpfn_fit"
        complete = json.loads(
            (E4_ROOT / uid / "FIT_COMPLETE.json").read_text(encoding="utf-8")
        )
        digest = sha256_file(state)
        if digest != complete["state_sha256"]:
            raise SystemExit(f"{uid}: state digest does not match FIT_COMPLETE")
        if complete["ensemble"]["effective_n_estimators_"] != 8:
            raise SystemExit(f"{uid}: E4 effective n_estimators_ is not 8")
        tree = ROOT / spec["tree_comparator"]
        if sha256_file(tree) != spec["tree_comparator_sha256"]:
            raise SystemExit(f"{uid}: tree comparator digest drifted")
        scaler_source: dict[str, Any]
        if spec["scaler_arm"] == "frozen_reference":
            sp = ROOT / spec["frozen_scaler"]
            scaler_source = {
                "kind": "persisted",
                "path": spec["frozen_scaler"],
                "sha256": sha256_file(sp),
            }
        else:
            scaler_source = {
                "kind": "rebuilt_and_verified",
                "rule": "StandardScaler().fit(this cell's own 20,000 context rows), "
                "exactly as E4's runner did",
                "verification": "transform the same raw context matrix and compare "
                "to the scaled X_train stored inside the E4 state's executor; the "
                "rebuild is rejected unless it reproduces it",
            }
        units.append(
            {
                "position": spec["position"],
                "unit_id": uid,
                "context_seed": spec["context_seed"],
                "cell": spec["cell"],
                "cell_dir": CELLS[spec["cell"]],
                "scaler_arm": spec["scaler_arm"],
                "state_path": f"data/processed/m5_e4_formal_path_a/{uid}/model.tabpfn_fit",
                "state_sha256": digest,
                "e4_process_uuid": complete["process_uuid"],
                "context_manifest": spec["context_manifest"],
                "context_manifest_sha256": spec["context_manifest_sha256"],
                "tree_comparator": spec["tree_comparator"],
                "tree_comparator_sha256": spec["tree_comparator_sha256"],
                "scaler_source": scaler_source,
                "repeats": REPEATS,
            }
        )
    if len(units) != 24:
        raise SystemExit(f"expected 24 states, built {len(units)}")
    order_digest = hashlib.sha256(
        "\n".join(u["unit_id"] for u in units).encode("utf-8")
    ).hexdigest()
    if order_digest != E4_REALISED_ORDER_DIGEST:
        raise SystemExit("state manifest order does not reproduce E4's order digest")
    return units


def decision_rules() -> dict:
    """Pre-declared, and not to be changed after seeing results."""
    return {
        "primary_replication_target": "hotwater-negative support main effect on "
        "steam separation",
        "co_primary_endpoints": [
            "steam_positive_vs_hotwater_negative_pairwise_auc",
            "steam_positive_minus_hotwater_negative_score_margin",
        ],
        "vocabulary": [
            "REPLICATED",
            "DIRECTIONALLY_CONSISTENT_BUT_INCONCLUSIVE",
            "NOT_REPLICATED",
            "EXECUTION_INCOMPLETE",
        ],
        "REPLICATED": [
            "AUC negative-support main effect > 0",
            "margin negative-support main effect > 0",
            "AUC and margin both positive in 3/3 context seeds",
            "both scaler arms positive",
            "building and segment 95% intervals exclude zero on both endpoints",
        ],
        "TABPFN_SPECIFIC_REPLICATED": "the TabPFN-minus-tree gap meets every "
        "REPLICATED condition on both endpoints",
        "DIRECTIONALLY_CONSISTENT_BUT_INCONCLUSIVE": [
            "overall AUC and margin both positive",
            "no arm reversed overall",
            "at least 2/3 seeds positive on both endpoints",
            "but the clustered intervals or the full 3/3 condition did not pass",
        ],
        "NOT_REPLICATED": [
            "any co-primary overall effect <= 0",
            "or any scaler arm reversed overall on any co-primary endpoint",
            "or any co-primary endpoint with at least 2/3 seeds non-positive",
        ],
        "EXECUTION_INCOMPLETE": "coverage, provenance or ensemble validation "
        "incomplete",
        "thresholds_may_not_change_after_seeing_results": True,
        "interval_containing_zero_is_not_proof_of_absence": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--base-commit", required=True)
    args = ap.parse_args()

    audit = query_audit()
    units = state_manifest()

    sources = {}
    for relpath in SOURCE_FILES:
        p = args.repo / relpath
        if not p.exists():
            raise SystemExit(f"source file missing, cannot freeze: {relpath}")
        sources[relpath] = sha256_source(p)

    repeats = [
        {
            "unit_id": u["unit_id"],
            "context_seed": u["context_seed"],
            "cell": u["cell"],
            "scaler_arm": u["scaler_arm"],
            "repeat": r,
            "mode": "same_process_inference_repeat_after_fresh_reload",
            "expected_score_length": 192,
            "expected_artifact": f"{u['unit_id']}/repeats/repeat_{r:03d}.json",
        }
        for u in units
        for r in range(REPEATS)
    ]

    protocol = {
        "schema": "m5_e5_independent_replication_protocol_v1",
        "execution_mode": "E5_INDEPENDENT_REPLICATION",
        "base_commit": args.base_commit,
        "e4_completion_commit": "9a336faa5f80ab7acb68533057d908d0723a90cf",
        "authorization": {
            "execution_status": "HUMAN_AUTHORIZED_FOR_EXECUTION",
            "human_authorization_date": "2026-08-02",
            "authorized_scope": "E5 independent replication only",
        },
        "scientific_role": (
            "does the steam negative-support response E4 established reproduce on "
            "a completely independent, never-scored 192-row query with zero "
            "overlap with the 352-row screening query?"
        ),
        "not_a_rediscovery": [
            "no mechanism re-exploration",
            "no endpoint re-selection",
            "no cell re-selection",
            "no protocol adjustment using results",
        ],
        "pure_rescoring": {
            "refit_prohibited": True,
            "prohibited_calls": [
                "model.fit",
                "fit_from_scratch",
                "context resampling",
                "model seed change",
                "scaler re-selection",
                "tree refit",
            ],
            "permitted_pipeline": "load_fitted_tabpfn_model -> verify state -> "
            "transform the fixed query -> predict_proba",
            "runtime_guard": "scripts/m5_e5_guard.py replaces every fit entry "
            "point with a raising stub before any model is loaded",
        },
        "inherited_from_e4": {
            "scientific_tabpfn_version": "8.0.8",
            "checkpoint": "v3",
            "feature_set": "F4_137",
            "context_n": 20000,
            "model_seed": 42,
            "context_seeds": CONTEXT_SEEDS,
            "cells": {c: CELLS[c] for c in ("11", "10", "01", "00")},
            "scaler_arms": SCALER_ARMS,
            "endpoint_code": "scripts/m5_e4_endpoints.py",
            "factorial_code": "scripts/m5_e4_endpoints.py::factor_effect",
            "clustered_code": "scripts/m5_e4_clustered.py",
        },
        "ensemble": {
            "requested_n_estimators": 8,
            "auto_scale_n_estimators": False,
            "required_effective_n_estimators_": 8,
            "mismatch_policy": "hard_failure",
            "verified_after_every_reload": [
                "tabpfn version 8.0.8",
                "state digest",
                "requested n_estimators == 8",
                "auto_scale_n_estimators is False",
                "effective n_estimators_ == 8",
                "runtime configs/pipelines/pipeline_seeds/"
                "subsample_feature_indices all == 8",
            ],
            "internal_members_are_not_replicates": True,
            "eight_repeats_are_not_eight_fits": True,
        },
        "lifecycle": {
            "one_state_per_fresh_process": True,
            "repeats_per_state": REPEATS,
            "score_vector_length": 192,
            "all_scoring_is_after_fresh_process_reload": True,
            "separate_fresh_process_diagnostic": "not applicable; every E5 score "
            "is already a fresh-process reload",
            "atomic_completion_marker_before_next_state": True,
        },
        "query": audit,
        "scaler_policy": {
            "arm_is_inherited_never_reselected": True,
            "frozen_reference": "load the persisted "
            "scalers/seed{S}_pooled_reference.joblib and verify its digest",
            "cell_specific": "not persisted by E4; rebuild with E4's exact code "
            "from the same 20,000 context rows and verify by transforming that "
            "matrix and comparing to the scaled X_train inside the E4 state",
            "verification_is_mandatory": "no scoring without it",
            "must_verify": [
                "137 feature order",
                "query raw_index order",
                "transformed query shape == (192, 137)",
                "float dtype identical to E4",
            ],
            "transformed_query_cache": "permitted, keyed on the scaler identity; "
            "it may never merge or skip a state",
            "cell_11_arms_still_scored_separately": True,
        },
        "trees": {
            "role": "matched_row_fixed_comparator",
            "refit": False,
            "artificial_replicates": False,
            "one_score_vector_per_unit": True,
            "note": "E4's tree comparators cover the 352-row query only; the "
            "fixed tree must be reloadable to score the 192-row query, and if it "
            "cannot be uniquely located E5 stops before any scientific scoring",
        },
        "endpoints": {
            "co_primary": decision_rules()["co_primary_endpoints"],
            "per_state": "compute the endpoint on each of the 8 repeat score "
            "vectors, then average the 8 endpoint values",
            "forbidden": [
                "averaging the 8 row-probability vectors and scoring once",
                "pairing repeat IDs across cells or states",
            ],
            "secondary_may_not_control_the_decision": [
                "positive_support_main_effect",
                "positive_x_negative_interaction",
                "scaler_arm_interaction",
                "hw01 local manipulation checks",
                "global and within-group rank, kept separate from score",
            ],
            "chilledwater": "absent from this query; no chilledwater endpoint may "
            "be added and no other query may be substituted",
        },
        "decision_rules": decision_rules(),
        "clustered_uncertainty": {
            "estimator": "identical to E4, scripts/m5_e4_clustered.py",
            "master_seed": 20260730,
            "namespace_code": 5005,
            "cluster_code": {"building": 1, "segment": 2},
            "construction": "np.random.SeedSequence([20260730, 5005, "
            "cluster_code[cluster_type], draw_id]) then PCG64 Generator",
            "draws": 1000,
            "draw_ids": [0, 999],
            "interval": "percentile q025 / q975",
            "shared_within_a_draw": [
                "cells",
                "arms",
                "context seeds",
                "repeats",
                "the fixed tree",
            ],
            "repeats_averaged_within_the_state_after_scoring_each": True,
            "scaler_interaction_formed_inside_the_draw": True,
            "three_seed_contrast_averaged_with_equal_weight_inside_the_draw": True,
        },
        "execution_order": {
            "inherited_from_e4": True,
            "realised_order_digest": E4_REALISED_ORDER_DIGEST,
            "reshuffling": "forbidden",
            "units": [
                {"position": u["position"], "unit_id": u["unit_id"]} for u in units
            ],
        },
        "resume_and_fail_closed": {
            "single_gpu_worker": True,
            "process_pool_executor": "forbidden",
            "concurrent_gpu_workers": "forbidden",
            "one_subprocess_per_state": True,
            "on_failure": "write INTERRUPTED_INCOMPLETE and stop the whole run",
            "reload_backfill": "forbidden",
            "silent_retry": "forbidden",
            "cpu_fallback": "prohibited",
        },
        "completion_census": {
            "states_reloaded": 24,
            "same_process_repeats": 192,
            "tabpfn_score_vectors": 192,
            "score_vector_length": 192,
            "tree_score_vectors": 24,
            "effective_n_estimators_": 8,
            "fits_performed": 0,
            "missing": 0,
            "duplicate": 0,
            "tmp_files": 0,
            "stderr_bytes": 0,
            "interrupted": 0,
        },
        "prohibitions": [
            "E6 full test",
            "any refit",
            "Path B",
            "representation ablation",
            "500k",
            "site transfer",
            "tree refit",
            "TabPFN 8.1.0 as science",
            "manuscript change",
            "scoring the 10,137,155-row holdout",
            "adding or removing query rows",
        ],
        "source_digest_method": "sha256 over newline-normalised content",
        "source_digests": sources,
    }

    proto_sha = atomic_json(args.out / "e5_protocol.json", {"protocol": protocol})
    state_sha = atomic_json(args.out / "e5_state_manifest.json", {"states": units})
    rep_sha = atomic_json(args.out / "e5_repeat_manifest.json", {"repeats": repeats})
    audit_sha = atomic_json(args.out / "e5_query_audit.json", audit)

    inputs = {
        "schema": "m5_e5_input_manifest_v1",
        "generated": time.time(),
        "base_commit": args.base_commit,
        "e4_completion_commit": protocol["e4_completion_commit"],
        "protocol_sha256": proto_sha,
        "state_manifest_sha256": state_sha,
        "repeat_manifest_sha256": rep_sha,
        "query_audit_sha256": audit_sha,
        "query": audit,
        "e4_state_digests": {u["unit_id"]: u["state_sha256"] for u in units},
        "tree_comparators": {u["unit_id"]: u["tree_comparator_sha256"] for u in units},
        "frozen_scalers": {
            f"seed{s}": sha256_file(
                FACTORIAL_ROOT / "scalers" / f"seed{s}_pooled_reference.joblib"
            )
            for s in CONTEXT_SEEDS
        },
        "e4_artifacts": {
            name: sha256_file(E4_ROOT / name)
            for name in (
                "e4_protocol.json",
                "e4_fit_manifest.json",
                "e4_summary.json",
                "e4_factorial.json",
                "e4_clustered.json",
                "e4_decision.json",
            )
        },
        "source_digests": sources,
    }
    atomic_json(args.out / "e5_input_manifest.json", inputs)

    print(f"protocol        sha256 = {proto_sha}")
    print(f"state manifest  sha256 = {state_sha}   ({len(units)} states)")
    print(f"repeat manifest sha256 = {rep_sha}   ({len(repeats)} repeats)")
    print(f"query audit     sha256 = {audit_sha}")
    print(f"execution order inherited, digest = {E4_REALISED_ORDER_DIGEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
