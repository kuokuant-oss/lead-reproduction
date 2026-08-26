"""Frozen-input and three-phase queue contract for M5 building-count V5."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lead import ROOT
from prepare_m5_building_count_v5_fixed_50k import (
    BUILDING_DRAW_SEEDS,
    CLASS_RATIO_POLICY,
    CONTEXT_ROWS,
    EVALUATION_METER_IDS,
    EXCLUDED_EVALUATION_METER_IDS,
    EXPERIMENT_VERSION,
    FULL_SUPPORT_ID,
    NESTED_BUILDING_BUDGETS,
    ROW_DRAW_SEEDS,
    ROWS_PER_CLASS,
    SAMPLING_PROFILE,
    SCHEDULED_BUDGETS,
    TRAINING_CONTEXT_POLICY,
    byte_array_sha256,
    int_array_sha256,
    planned_contexts,
    resolve_recorded_path,
    sha256_file,
)


OPERATIONAL_FULL_BUILDING_SEED = 725
BUDGETS = (50, 100, 200, 400, 725)
PAIR_ORDER_POLICY = (
    "full725_rows01_then_seeds0to2_k400_k200_k100_k50_then_"
    "seeds3to4_k400_k200_k100_k50"
)
VALIDATION_CONTEXTS = ((725, 0, 725), (0, 0, 50))
DEFAULT_AUDIT_ROOT = ROOT / "experiments/m5_building_count_v5_fixed_50k/audit"
DEFAULT_HOLDOUT_ARTIFACT = (
    ROOT
    / "data/processed/m5_building_curve/v5_fixed_50k/"
    "canonical_holdout_non_electric.npz"
)


@dataclass(frozen=True)
class FixedContext:
    manifest_path: Path
    manifest: dict[str, Any]
    source_manifest_path: Path
    source_manifest: dict[str, Any]
    artifact_path: Path
    building_seed: int
    building_draw_seed: int | None
    support_id: str
    row_seed: int
    budget: int
    raw_index: np.ndarray
    anomaly: np.ndarray
    building_id: np.ndarray
    meter: np.ndarray
    selected_buildings: np.ndarray


def queued_contexts() -> list[tuple[int, int, int]]:
    return [
        (
            OPERATIONAL_FULL_BUILDING_SEED
            if context.building_seed is None
            else context.building_seed,
            context.row_seed,
            context.budget,
        )
        for context in planned_contexts()
    ]


def context_manifest_path(
    audit_root: Path, building_seed: int, row_seed: int, budget: int
) -> Path:
    building = "full" if building_seed == OPERATIONAL_FULL_BUILDING_SEED else f"b{building_seed}"
    return audit_root / "contexts" / f"k{budget}_{building}_r{row_seed}.json"


def phase_for_context(building_seed: int, row_seed: int, budget: int) -> int:
    del row_seed, budget
    if building_seed == OPERATIONAL_FULL_BUILDING_SEED:
        return 1
    return 2 if building_seed in (0, 1, 2) else 3


def phase_final_contexts() -> dict[int, tuple[int, int, int]]:
    contexts = queued_contexts()
    result: dict[int, tuple[int, int, int]] = {}
    for context in contexts:
        result[phase_for_context(*context)] = context
    return result


def verify_training_context_gate(audit_root: Path) -> dict[str, Any]:
    path = audit_root / "training_context_gate.json"
    if not path.is_file():
        raise ValueError(f"V5 training-context gate is missing: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "experiment_version": gate.get("experiment_version") == EXPERIMENT_VERSION,
        "passed": gate.get("passed") is True,
        "status": gate.get("status") == "PASSED",
        "budgets": tuple(map(int, gate.get("scheduled_budgets", ())))
        == SCHEDULED_BUDGETS,
        "building_seeds": tuple(
            map(int, gate.get("building_seeds_for_k50_to_k400", ()))
        )
        == BUILDING_DRAW_SEEDS,
        "row_seeds": tuple(map(int, gate.get("row_seeds", ()))) == ROW_DRAW_SEEDS,
        "context_rows": int(gate.get("context_rows", -1)) == CONTEXT_ROWS,
        "rows_per_class": int(gate.get("rows_per_class", -1)) == ROWS_PER_CLASS,
        "candidate_buildings": int(gate.get("candidate_buildings", -1)) == 725,
        "expected_context_cells": int(gate.get("expected_context_cells", -1)) == 42,
        "checked_context_cells": int(gate.get("checked_context_cells", -1)) == 42,
        "expected_model_cells": int(gate.get("expected_model_cells", -1)) == 84,
        "evaluation_meters": tuple(
            map(int, gate.get("evaluation_meter_ids", ()))
        )
        == EVALUATION_METER_IDS,
        "excluded_meter": tuple(
            map(int, gate.get("excluded_evaluation_meter_ids", ()))
        )
        == EXCLUDED_EVALUATION_METER_IDS,
        "queue": gate.get("context_keys_in_run_order")
        == [context.key for context in planned_contexts()],
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"V5 training-context gate failed: {failures}")
    evaluation = gate.get("evaluation_holdout", {})
    artifact = resolve_recorded_path(evaluation.get("artifact", ""))
    if not artifact.is_file() or sha256_file(artifact) != evaluation.get("artifact_sha256"):
        raise ValueError("V5 evaluation holdout artifact digest drifted")
    with np.load(artifact) as payload:
        raw_index = np.asarray(payload["validation_raw_index"], dtype="int64")
    if int_array_sha256(raw_index) != evaluation.get("validation_raw_index_sha256"):
        raise ValueError("V5 evaluation holdout row identity drifted")
    return gate


def canonical_holdout_row_sha256(audit_root: Path = DEFAULT_AUDIT_ROOT) -> str:
    return str(
        verify_training_context_gate(audit_root)["evaluation_holdout"][
            "validation_raw_index_sha256"
        ]
    )


def load_fixed_context(manifest_path: Path, budget: int) -> FixedContext:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest.get("identity", {})
    if (
        manifest.get("status") != "FROZEN"
        or identity.get("experiment_version") != EXPERIMENT_VERSION
        or identity.get("sampling_profile") != SAMPLING_PROFILE
        or int(identity.get("budget", -1)) != int(budget)
        or int(identity.get("context_rows", -1)) != CONTEXT_ROWS
        or int(identity.get("rows_per_class", -1)) != ROWS_PER_CLASS
    ):
        raise ValueError(f"V5 context identity drifted: {manifest_path}")
    building_draw_seed = identity.get("building_seed")
    building_seed = (
        OPERATIONAL_FULL_BUILDING_SEED
        if building_draw_seed is None
        else int(building_draw_seed)
    )
    row_seed = int(identity["row_seed"])
    if building_seed not in (*BUILDING_DRAW_SEEDS, OPERATIONAL_FULL_BUILDING_SEED):
        raise ValueError("unsupported V5 building identity")
    if row_seed not in ROW_DRAW_SEEDS:
        raise ValueError("unsupported V5 row seed")
    if budget not in BUDGETS:
        raise ValueError("unsupported V5 building budget")
    if (budget == 725) != (building_seed == OPERATIONAL_FULL_BUILDING_SEED):
        raise ValueError("K725 must use exactly the canonical full support")

    artifact_path = resolve_recorded_path(manifest["context_artifact"])
    if sha256_file(artifact_path) != manifest["context_artifact_sha256"]:
        raise ValueError(f"V5 context artifact digest drift: {artifact_path}")
    with np.load(artifact_path) as payload:
        raw_index = np.asarray(payload["raw_index"], dtype="int64")
        anomaly = np.asarray(payload["anomaly"], dtype="int8")
        building_id = np.asarray(payload["building_id"], dtype="int64")
        meter = np.asarray(payload["meter"], dtype="int8")
        selected_buildings = np.asarray(payload["selected_buildings"], dtype="int64")
    if len(raw_index) != CONTEXT_ROWS or len(np.unique(raw_index)) != CONTEXT_ROWS:
        raise ValueError("V5 context size/uniqueness drifted")
    if int(anomaly.sum()) != ROWS_PER_CLASS or int((anomaly == 0).sum()) != ROWS_PER_CLASS:
        raise ValueError("V5 context is not exactly 50:50")
    if int_array_sha256(raw_index) != manifest["raw_index_sha256"]:
        raise ValueError("V5 context row digest drifted")
    if byte_array_sha256(anomaly) != manifest["anomaly_sha256"]:
        raise ValueError("V5 context label digest drifted")
    if int_array_sha256(selected_buildings) != manifest["selected_building_sha256"]:
        raise ValueError("V5 selected-building digest drifted")
    if not np.isin(building_id, selected_buildings).all() or np.any(building_id % 2):
        raise ValueError("V5 context escaped its building support")

    source_path = resolve_recorded_path(manifest["source_manifest"])
    if sha256_file(source_path) != manifest["source_manifest_sha256"]:
        raise ValueError("V5 source-building manifest digest drifted")
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source_cell = source_manifest.get("cells", {}).get(str(budget))
    if source_cell is None:
        raise ValueError("V5 source-building manifest lacks its K cell")
    source_selected = np.asarray(source_cell["available_buildings"], dtype="int64")
    if not np.array_equal(source_selected, selected_buildings):
        raise ValueError("V5 source-building support differs from context")

    runtime_manifest = copy.deepcopy(manifest)
    runtime_manifest["balance_seed"] = row_seed
    runtime_manifest["building_seed"] = building_seed
    runtime_manifest["row_seed"] = row_seed
    runtime_manifest["sampling_profile"] = SAMPLING_PROFILE
    runtime_manifest["cells"] = {
        str(budget): {
            "raw_index_sha256": manifest["raw_index_sha256"],
            "selected_buildings": selected_buildings.tolist(),
        }
    }
    return FixedContext(
        manifest_path=manifest_path,
        manifest=runtime_manifest,
        source_manifest_path=source_path,
        source_manifest=source_manifest,
        artifact_path=artifact_path,
        building_seed=building_seed,
        building_draw_seed=(
            None if building_draw_seed is None else int(building_draw_seed)
        ),
        support_id=str(identity["support_id"]),
        row_seed=row_seed,
        budget=int(budget),
        raw_index=raw_index,
        anomaly=anomaly,
        building_id=building_id,
        meter=meter,
        selected_buildings=selected_buildings,
    )


def verify_context_against_frame(context: FixedContext, frame: Any) -> None:
    rows = context.raw_index
    observed_anomaly = frame.loc[rows, "anomaly"].to_numpy(dtype="int8")
    observed_building = frame.loc[rows, "building_id"].to_numpy(dtype="int64")
    observed_meter = frame.loc[rows, "meter"].to_numpy(dtype="int8")
    if not np.array_equal(observed_anomaly, context.anomaly):
        raise ValueError("V5 context anomaly identity differs from raw frame")
    if not np.array_equal(observed_building, context.building_id):
        raise ValueError("V5 context building identity differs from raw frame")
    if not np.array_equal(observed_meter, context.meter):
        raise ValueError("V5 context meter identity differs from raw frame")
    if np.any(observed_building % 2):
        raise ValueError("V5 context contains odd holdout buildings")
