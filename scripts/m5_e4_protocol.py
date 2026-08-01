"""Freeze the M5 E4 formal Path A protocol before any fit runs.

Writes four artifacts and nothing else:

* `e4_protocol.json`        -- the frozen design, including the human rulings on
                               the ensemble contract and on clustered
                               uncertainty, recorded verbatim.
* `e4_input_manifest.json`  -- every input the results will depend on, by digest.
* `e4_fit_manifest.json`    -- the 24 external fitted states.
* `e4_repeat_manifest.json` -- the 192 same-process inference repeats.

Nothing here fits, scores, or reads a model. If an input cannot be located or a
digest cannot be computed, this fails rather than writing a partial protocol.

The rulings below were issued by the human operator on 2026-08-02 and are
reproduced as given. Where this file restates a rule in code, the code is the
implementation and the recorded text is the authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

CELLS = {
    "11": "hw_pos_present__hw_neg_present",
    "10": "hw_pos_present__hw_neg_excluded",
    "01": "hw_pos_excluded__hw_neg_present",
    "00": "hw_pos_excluded__hw_neg_excluded",
}
# Canonical policy order, reused unchanged.
CELL_ORDER = ["11", "10", "01", "00"]
CONTEXT_SEEDS = [42, 123, 999]
SCALER_ARMS = ["cell_specific", "frozen_reference"]

ROOT = Path(r"C:\Users\tonykuo\projects\lead-reproduction")
FACTORIAL_ROOT = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"
QUERY_MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "m5_context_stories"
    / "queries"
    / "screening"
    / "manifest.json"
)
QUERY_NPZ = QUERY_MANIFEST.with_name("queries.npz")

# Source files whose content defines the science. Frozen by digest so a later
# edit cannot silently change what "the E4 protocol" meant.
SOURCE_FILES = [
    "scripts/m5_e4_protocol.py",
    "scripts/m5_e4_endpoints.py",
    "scripts/m5_e4_runner.py",
    "scripts/m5_e4_clustered.py",
    "scripts/m5_e3_runner.py",
    "scripts/analyze_m5_hotwater_label_role_factorial.py",
    "scripts/run_m5_hotwater_label_role_factorial.py",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> str:
    """Write JSON and return the digest of what actually landed on disk.

    `newline=""` is load-bearing. Without it Python translates "\\n" to CRLF on
    Windows, so the file would not match the digest computed from the body, and
    freezing the same protocol on Linux and on Windows would produce different
    artifacts for identical content.
    """
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


def unit_id(seed: int, cell: str, arm: str) -> str:
    return f"seed{seed}__cell{cell}__{arm}"


def execution_order() -> list[dict]:
    """The realised order of the 24 fits.

    No schedule randomisation was specified for E4, and none is invented here.
    The order is the frozen context-seed list, then the canonical policy cell
    order, then the scaler-arm list -- every component already fixed elsewhere
    in this protocol. It is fully deterministic and carries no seed.
    """
    order = []
    for seed in CONTEXT_SEEDS:
        for cell in CELL_ORDER:
            for arm in SCALER_ARMS:
                order.append(
                    {
                        "position": len(order),
                        "unit_id": unit_id(seed, cell, arm),
                        "context_seed": seed,
                        "cell": cell,
                        "cell_dir": CELLS[cell],
                        "scaler_arm": arm,
                    }
                )
    return order


def ensemble_contract() -> dict:
    """Human ruling of 2026-08-02 on the TabPFN internal ensemble."""
    return {
        "requested_n_estimators": 8,
        "auto_scale_n_estimators": False,
        "required_effective_n_estimators_": 8,
        "mismatch_policy": "hard_failure",
        "dimension_separation": (
            "n_estimators=8 are ensemble members combined inside a single "
            "predict_proba call. inference repeats=8 are eight separate "
            "predict_proba calls on one fitted state. Internal members are not "
            "fit replicates, not inference replicates, not bootstrap "
            "replicates, and not independent observations."
        ),
        "runner_must_check_after_each_fit": [
            "model.n_estimators_ == 8",
            "len(model.ensemble_configs_) == 8",
            "low-memory executor ensemble_preprocessor configs == 8",
            "low-memory executor ensemble_preprocessor pipelines == 8",
            "low-memory executor ensemble_preprocessor pipeline_seeds == 8",
            "low-memory executor ensemble_preprocessor subsample_feature_indices == 8",
        ],
        "fit_complete_must_record": ["requested", "auto_scale", "effective"],
        "importer_must_reread_effective_from_persisted_state": True,
        "importer_must_not_trust_runner_self_report": True,
        "e3_verification": {
            "verified_on": "2026-08-02",
            "method": (
                "read init_params.json and fitted_attrs.joblib out of each E3 "
                "persisted state, then re-read n_estimators_ and the runtime "
                "ensemble containers after a fresh GPU reload"
            ),
            "effective_n_estimators_": {"00": 8, "01": 8, "10": 8, "11": 8},
            "runtime_containers": "configs/pipelines/pipeline_seeds/"
            "subsample_feature_indices all length 8 in all four cells",
            "auto_scale_would_not_have_fired": (
                "max_features_per_estimator=200 and 137 features give "
                "ceil(137/200)=1, and 8 >= 1, so scaling is a no-op even with "
                "auto_scale_n_estimators=True; MAX_AUTO_SCALED_N_ESTIMATORS=32"
            ),
            "recording_defect_in_e3": (
                "E3 fit_complete.json recorded only the requested value and no "
                "frozen document pinned n_estimators; the setting was correct "
                "but unfrozen. E4 pins it and records the effective value."
            ),
        },
    }


def clustered_uncertainty() -> dict:
    """Human rulings A-E of 2026-08-02, recorded as issued."""
    return {
        "issued": "2026-08-02",
        "authority": "human operator ruling; this text governs the implementation",
        "A_repeats_into_cluster_bootstrap": {
            "per_draw_procedure": [
                "produce one resampled row multiset",
                "apply the same rows to every cell, arm, context seed, TabPFN "
                "repeat and the fixed tree comparator",
                "for each fitted state compute the endpoint separately on each "
                "of its 8 repeat score vectors",
                "average the 8 endpoint values within that fit to obtain the "
                "fit-level draw estimate",
                "form factorial contrasts from the fit-level estimates",
            ],
            "forbidden": [
                "averaging row probabilities before computing AUC",
                "pairing repeat IDs across cells",
            ],
            "note": "repeat IDs have no natural pairing between cells",
        },
        "B_continuous_score_margin": {
            "definition_source": "scripts/m5_e3_runner.py::endpoints",
            "endpoint": "steam_positive_minus_hotwater_negative_score_margin",
            "formula": "mean(score[steam & anomaly==1]) - mean(score[hotwater & anomaly==0])",
            "per_draw": "recompute the margin for every repeat, then average the "
            "8 repeat margins within the fit, then form factorial effects and "
            "the tree gap",
            "invalid_rule": "if any stratum is empty or the result is non-finite, "
            "mark that endpoint/draw invalid",
            "no_imputation": True,
            "no_metric_substitution": True,
        },
        "C_scaler_arm_interaction": {
            "shared_draws": "building and segment draws are shared across both "
            "scaler arms",
            "formula": "scaler_interaction = factorial_effect_frozen_scaler - "
            "factorial_effect_cell_specific",
            "applies_to": [
                "positive_support_main_effect",
                "negative_support_main_effect",
                "positive_x_negative_interaction",
            ],
            "forbidden": "subtracting two independent bootstrap intervals after "
            "the fact",
        },
        "D_context_seed_aggregation": {
            "seeds": CONTEXT_SEEDS,
            "per_seed": "form a seed-specific contrast first",
            "overall_point_estimate": "(contrast_42 + contrast_123 + contrast_999) / 3",
            "within_draw": "form the three seed contrasts inside each draw, then "
            "average with equal weight",
            "overall_interval": "percentile q025/q975 over the 1000 overall draw "
            "values",
            "also_report": [
                "the three per-seed results",
                "range",
                "sample SD",
                "sign consistency (e.g. 3/3 positive)",
            ],
            "forbidden": [
                "resampling the three seeds",
                "random-effects model",
                "n=3 t-test for general inference",
                "selecting the best seed",
                "flattening all repeats into independent samples",
            ],
            "interpretation": "overall means the equal-weight average of the "
            "three pre-specified seeds 42, 123, 999 and nothing more",
        },
        "E_addressable_seed_mapping": {
            "master_seed": 20260730,
            "namespace_code": 4004,
            "cluster_code": {"building": 1, "segment": 2},
            "construction": "np.random.SeedSequence([20260730, 4004, "
            "cluster_code[cluster_type], draw_id]) then PCG64 Generator",
            "draw_ids": [0, 999],
            "draws": 1000,
            "shared_across": [
                "3 context seeds",
                "4 cells",
                "2 scaler arms",
                "8 repeats",
                "TabPFN and tree",
                "all endpoints",
            ],
            "forbidden": "depending on loop order or a single continuous rng stream",
        },
        "cluster_definitions": {
            "source": "scripts/analyze_m5_hotwater_label_role_factorial.py",
            "building": "query building_id as string",
            "segment": "anomaly rows grouped into contiguous runs by raw_index "
            "gap > 1; non-anomaly rows each form their own singleton cluster "
            "named normal_<raw_index>",
            "resample": "cluster names drawn with replacement, count preserved; "
            "member row indices concatenated",
            "interval": "percentile q025 / q975",
        },
    }


def factorial_contracts() -> dict:
    """Ruling F: reuse the repository's exact factor_effect formulas."""
    return {
        "source": "scripts/analyze_m5_hotwater_label_role_factorial.py::factor_effect",
        "cell_coding": {
            "00": {
                "hotwater_positive_present": False,
                "hotwater_negative_present": False,
            },
            "01": {
                "hotwater_positive_present": False,
                "hotwater_negative_present": True,
            },
            "10": {
                "hotwater_positive_present": True,
                "hotwater_negative_present": False,
            },
            "11": {
                "hotwater_positive_present": True,
                "hotwater_negative_present": True,
            },
        },
        "formulas": {
            "positive_support_main_effect": "(y10 + y11 - y00 - y01) / 2",
            "negative_support_main_effect": "(y01 + y11 - y00 - y10) / 2",
            "positive_x_negative_interaction": "y11 - y10 - y01 + y00",
        },
        "sign_and_coding_unchanged": True,
        "per_endpoint_outputs": [
            "positive main effect",
            "negative main effect",
            "positive x negative interaction",
            "scaler-arm interaction of each of the three",
            "TabPFN effect",
            "tree effect",
            "TabPFN-minus-tree effect",
        ],
    }


def uncertainty_interpretation() -> dict:
    """Ruling G: what the intervals may and may not be said to cover."""
    return {
        "reported_separately": [
            "inference variation: the 8 repeats of one fit",
            "context-seed variation: 42 / 123 / 999",
            "building-clustered uncertainty",
            "segment-clustered uncertainty",
        ],
        "conditioning": "clustered intervals are conditional on the fixed 24 "
        "fitted states and the three specified context seeds",
        "must_not_claim": "that the intervals contain model-seed or fresh-fit "
        "population variance, which was not executed",
    }


def build_fit_manifest() -> list[dict]:
    units = []
    for entry in execution_order():
        seed, cell, arm = entry["context_seed"], entry["cell"], entry["scaler_arm"]
        mpath = FACTORIAL_ROOT / "manifests" / f"seed{seed}" / f"{CELLS[cell]}.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if manifest["context_seed"] != seed or manifest["model_seed"] != 42:
            raise SystemExit(f"{mpath}: seed contract violated")
        if manifest["context_rows"] != 20_000 or manifest["feature_count"] != 137:
            raise SystemExit(f"{mpath}: N or feature contract violated")
        tree = (
            FACTORIAL_ROOT
            / "predictions"
            / "trees"
            / f"seed{seed}"
            / CELLS[cell]
            / arm
            / "predictions.npz"
        )
        if not tree.exists():
            raise SystemExit(f"missing fixed tree comparator: {tree}")
        scaler = FACTORIAL_ROOT / "scalers" / f"seed{seed}_pooled_reference.joblib"
        if not scaler.exists():
            raise SystemExit(f"missing frozen scaler: {scaler}")
        units.append(
            {
                **entry,
                "fits": 1,
                "repeats": 8,
                "context_manifest": str(mpath.relative_to(ROOT)),
                "context_manifest_sha256": sha256_file(mpath),
                "context_raw_index_sha256": manifest["raw_index_sha256"],
                "pooled_reference_raw_index_sha256": manifest[
                    "pooled_reference_raw_index_sha256"
                ],
                "label_counts": manifest["label_counts"],
                "tree_comparator": str(tree.relative_to(ROOT)),
                "tree_comparator_sha256": sha256_file(tree),
                "frozen_scaler": str(scaler.relative_to(ROOT)),
                "frozen_scaler_sha256": sha256_file(scaler),
                # For cell 11 the frozen scaler is fitted on cell 11's own rows,
                # so the two arms are the same transform and this unit is not an
                # independent second observation of the scaler axis.
                "arms_are_identical_by_construction": cell == "11",
            }
        )
    return units


def build_repeat_manifest(units: list[dict]) -> list[dict]:
    return [
        {
            "unit_id": u["unit_id"],
            "context_seed": u["context_seed"],
            "cell": u["cell"],
            "scaler_arm": u["scaler_arm"],
            "repeat": r,
            "mode": "same_process_inference_repeat",
            "expected_artifact": f"{u['unit_id']}/repeats/repeat_{r:03d}.json",
        }
        for u in units
        for r in range(8)
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--base-commit", required=True)
    args = ap.parse_args()

    qmanifest = json.loads(QUERY_MANIFEST.read_text(encoding="utf-8"))
    if qmanifest["query_rows"] != 352:
        raise SystemExit("the screening query is not the 352-row artifact")

    units = build_fit_manifest()
    if len(units) != 24:
        raise SystemExit(f"expected 24 fits, built {len(units)}")
    repeats = build_repeat_manifest(units)
    if len(repeats) != 192:
        raise SystemExit(f"expected 192 repeats, built {len(repeats)}")

    sources = {}
    for rel in SOURCE_FILES:
        p = args.repo / rel
        if not p.exists():
            raise SystemExit(f"source file missing, cannot freeze: {rel}")
        sources[rel] = sha256_file(p)

    protocol = {
        "schema": "m5_e4_formal_path_a_protocol_v1",
        "execution_mode": "E4_FORMAL_PATH_A",
        "base_commit": args.base_commit,
        "authorization": {
            "execution_status": "HUMAN_AUTHORIZED_FOR_EXECUTION",
            "human_authorization_date": "2026-08-02",
            "authorized_scope": "E4 formal Path A only",
        },
        "design": {
            "fits": 24,
            "decomposition": "3 context seeds x 4 cells x 2 scaler arms",
            "fits_per_unit": 1,
            "repeats_per_fit": 8,
            "same_process_repeats_total": 192,
            "external_fitted_states": 24,
            "not_192_fits": (
                "r is repeated inference on one fitted state, per the canonical "
                "policy estimand; it is not a fit replicate"
            ),
            "model_seed_factor_added": False,
        },
        "inherited": {
            "scientific_tabpfn_version": "8.0.8",
            "checkpoint": "v3",
            "feature_set": "F4_137",
            "context_n": 20000,
            "model_seed": 42,
            "query": "original_352_row_screening_query",
            "estimand": "Y[c,s,a,r] = mu[c,s,a] + epsilon[c,s,a,r]",
        },
        "ensemble": ensemble_contract(),
        "context_seeds": CONTEXT_SEEDS,
        "cells": {c: CELLS[c] for c in CELL_ORDER},
        "scaler_arms": {
            "cell_specific": {
                "rule": "StandardScaler().fit(this cell's own 20,000 context rows)",
                "source": "scripts/run_m5_hotwater_label_role_factorial.py",
            },
            "frozen_reference": {
                "rule": "StandardScaler().fit(the hw_pos_present__hw_neg_present "
                "pooled-reference 20,000x137 matrix of the same context seed)",
                "persisted": "data/processed/m5_hotwater_label_factorial/scalers/"
                "seed{S}_pooled_reference.joblib",
                "applied_to": "both the context matrix and the query matrix, "
                "cast to float32",
                "note": "for cell 11 the two arms fit on the same rows by construction",
            },
        },
        "query": {
            "manifest": str(QUERY_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": sha256_file(QUERY_MANIFEST),
            "npz_sha256": sha256_file(QUERY_NPZ),
            "raw_index_sha256": qmanifest["raw_index_sha256"],
            "rows": 352,
        },
        "trees": {
            "role": "matched_row_fixed_comparator",
            "refit": False,
            "tuned": False,
            "artificial_replicates": False,
            "gap_rule": "per repeat r: compute the TabPFN endpoint independently, "
            "then gap = TabPFN endpoint - the fixed tree endpoint of the same unit",
        },
        "execution_order": execution_order(),
        "execution_order_note": (
            "deterministic; no schedule randomisation was specified for E4 and "
            "none was invented"
        ),
        "factorial": factorial_contracts(),
        "clustered_uncertainty": clustered_uncertainty(),
        "uncertainty_interpretation": uncertainty_interpretation(),
        "readouts": {
            "primary_steam": [
                "steam_positive_vs_hotwater_negative_pairwise_auc",
                "steam_positive_minus_hotwater_negative_score_margin",
                "tabpfn_minus_tree_factorial_response",
            ],
            "auc_and_margin_reported_together": True,
            "saturation_warning": (
                "E3 cell 01 saturated the steam AUC at 1.0; an AUC half-width of "
                "zero is saturation, not precision, and the margin is retained as "
                "a co-primary for that reason"
            ),
            "secondary": [
                "steam_positive_global_rank",
                "steam_positive_within_meter_rank",
                "chilledwater_positive_vs_chilledwater_negative_pairwise_auc",
                "chilledwater_positive_minus_chilledwater_negative_score_margin",
                "chilledwater_within_meter_pr_auc",
                "chilledwater_within_meter_roc_auc",
                "chilledwater_positive_within_meter_rank",
                "chilledwater_positive_global_rank",
            ],
            "score_and_rank_never_pooled": True,
            "hotwater_local_readouts": "manipulation_checks_only",
            "retained_labels": {
                "chilledwater_vs_hotwater_negative": "RESOLUTION_LIMITED_DIAGNOSTIC",
                "onset_middle_recovery": "UNRESOLVED_NOT_EXECUTED",
            },
            "labels_must_not_carry_mechanism_conclusions": True,
            "chilledwater_not_pooled_into_the_steam_mechanism_claim": True,
        },
        "output_schema": {
            "per_unit_directory": "{unit_id}/",
            "fit_start": "fit_start.json",
            "fit_complete": "FIT_COMPLETE.json",
            "state": "model.tabpfn_fit",
            "repeats": "repeats/repeat_RRR.json",
            "repeat_record_fields": [
                "unit_id",
                "context_seed",
                "cell",
                "scaler_arm",
                "repeat",
                "endpoints",
                "score_sha256",
                "seconds",
                "state_sha256",
                "process_uuid",
            ],
        },
        "cell_11_arm_degeneracy": {
            "fact": (
                "the frozen_reference scaler of a context seed is fitted on that "
                "seed's hw_pos_present__hw_neg_present matrix, which IS cell 11, "
                "so for cell 11 the two scaler arms are the same transform"
            ),
            "evidence": (
                "the 24 fixed tree comparators carry only 21 distinct digests; the "
                "three collisions are exactly the cell-11 arm pairs, one per seed"
            ),
            "consequences": [
                "cell 11 contributes no information to the scaler axis",
                "the tree scaler-arm interaction is identically zero at cell 11",
                "the two cell-11 units receive byte-identical fit inputs, so their "
                "persisted states may coincide; that is expected, not a defect",
            ],
            "units_still_executed": "all 24; the design is not silently reduced",
            "census_effect": "distinct state identities are required to be >= 21, "
            "and any collision must be a cell-11 arm pair of the same context seed",
        },
        "completion_census": {
            "fits": 24,
            "same_process_repeats": 192,
            "distinct_state_identities_minimum": 21,
            "permitted_state_collisions": "cell-11 arm pairs within one context "
            "seed only; any other collision is a hard failure",
            "missing": 0,
            "duplicate": 0,
            "tmp_files": 0,
            "stderr_bytes": 0,
            "interrupted": 0,
            "provenance_mismatch": 0,
        },
        "resume_and_fail_closed": {
            "atomic_writes": "temp file then os.replace, per unit",
            "resume": "a unit whose FIT_COMPLETE.json exists is skipped",
            "interrupted_marker": "INTERRUPTED_INCOMPLETE",
            "reload_backfill_of_same_process_repeats": "forbidden",
            "process_death": "mark the unit INTERRUPTED_INCOMPLETE and stop that "
            "unit; never continue its repeats from a reloaded state",
            "cpu_fallback": "prohibited",
            "parallel_gpu_workers": 1,
            "process_pool_executor": "forbidden",
        },
        "feature_matrix_cache": {
            "allowed": True,
            "rule": "build the raw feature matrix once per (context seed, cell), "
            "verify by digest, then let both scaler arms transform it",
            "must_not": [
                "reuse across a different context seed or cell",
                "change any scientific input",
            ],
            "requires": ["manifest", "content digest"],
        },
        "prohibitions": [
            "E5 frozen 192-row query",
            "E6 complete other-half full test",
            "Path B",
            "representation ablation",
            "500k",
            "site transfer",
            "tree refit",
            "TabPFN 8.1.0 as science",
            "manuscript change",
            "adding a model-seed factor",
            "changing N, cells, seeds or arms",
            "treating internal ensemble members as replicates",
        ],
        "source_digests": sources,
    }

    proto_sha = atomic_json(args.out / "e4_protocol.json", {"protocol": protocol})
    fit_sha = atomic_json(args.out / "e4_fit_manifest.json", {"fits": units})
    rep_sha = atomic_json(args.out / "e4_repeat_manifest.json", {"repeats": repeats})

    inputs = {
        "schema": "m5_e4_input_manifest_v1",
        "generated": time.time(),
        "base_commit": args.base_commit,
        "protocol_sha256": proto_sha,
        "fit_manifest_sha256": fit_sha,
        "repeat_manifest_sha256": rep_sha,
        "query": protocol["query"],
        "context_manifests": {
            u["unit_id"]: u["context_manifest_sha256"] for u in units
        },
        "tree_comparators": {u["unit_id"]: u["tree_comparator_sha256"] for u in units},
        "frozen_scalers": {
            f"seed{s}": sha256_file(
                FACTORIAL_ROOT / "scalers" / f"seed{s}_pooled_reference.joblib"
            )
            for s in CONTEXT_SEEDS
        },
        "source_digests": sources,
    }
    atomic_json(args.out / "e4_input_manifest.json", inputs)

    print(f"protocol        sha256 = {proto_sha}")
    print(f"fit manifest    sha256 = {fit_sha}   ({len(units)} fits)")
    print(f"repeat manifest sha256 = {rep_sha}   ({len(repeats)} repeats)")
    print(
        f"realised order  = {[u['unit_id'] for u in units[:3]]} ... "
        f"{[u['unit_id'] for u in units[-1:]]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
