"""Strict input and scheduling contract for M5 building-count V3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lead import ROOT
from m5_building_curve_protocol import int_array_sha256
from prepare_m5_building_count_v3_balanced_contexts import (
    BALANCE_SEED,
    BUDGETS,
    BUILDING_SEEDS,
    CONTEXT_ROWS,
    EXPERIMENT_VERSION,
    SAMPLING_PROFILE,
    file_sha256,
    validate_seed_artifact,
)

SEED_GROUPS = (tuple(range(42, 47)), tuple(range(47, 52)))
TRAINING_CONTEXT_POLICY = "exact_frozen_global_label_50_50_no_resampling"
CLASS_RATIO_POLICY = "exact_global_50_50"
CANONICAL_HOLDOUT_SHA256 = (
    "6cfebd1cb2bb818f69806c0f14d66a84b81c53d37a716badd48c17b86210d893"
)


@dataclass(frozen=True)
class BalancedContext:
    manifest_path: Path
    manifest: dict[str, Any]
    source_manifest_path: Path
    source_manifest: dict[str, Any]
    artifact_path: Path
    building_seed: int
    budget: int
    raw_index: np.ndarray
    anomaly: np.ndarray
    building_id: np.ndarray
    meter: np.ndarray
    selected_buildings: np.ndarray


def resolve_recorded_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def grouped_budget_major_pairs() -> list[tuple[int, int]]:
    """Complete seeds 42--46 before starting the 47--51 group."""
    return [
        (seed, budget) for group in SEED_GROUPS for budget in BUDGETS for seed in group
    ]


def verify_training_context_gate(audit_root: Path) -> dict[str, Any]:
    path = audit_root / "training_context_gate.json"
    if not path.is_file():
        raise ValueError(f"V3 training-context gate is missing: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    expected = len(BUILDING_SEEDS) * len(BUDGETS)
    checks = {
        "experiment_version": gate.get("experiment_version") == EXPERIMENT_VERSION,
        "passed": gate.get("passed") is True,
        "status": gate.get("status") == "PASSED",
        "expected_cells": int(gate.get("expected_cells", -1)) == expected,
        "checked_cells": int(gate.get("checked_cells", -1)) == expected,
        "building_seeds": tuple(map(int, gate.get("building_seeds", ())))
        == BUILDING_SEEDS,
        "budgets": tuple(map(int, gate.get("budgets", ()))) == BUDGETS,
        "balance_seed": int(gate.get("balance_seed", -1)) == BALANCE_SEED,
        "sampling_profile": gate.get("sampling_profile") == SAMPLING_PROFILE,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"V3 training-context gate failed: {failures}")
    return gate


def load_balanced_context(manifest_path: Path, budget: int) -> BalancedContext:
    manifest_path = manifest_path.resolve()
    if int(budget) not in BUDGETS:
        raise ValueError(f"unsupported V3 building budget: {budget}")
    if validate_seed_artifact(manifest_path) != len(BUDGETS):
        raise ValueError(f"V3 context manifest did not validate all K: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sampling_profile") != SAMPLING_PROFILE:
        raise ValueError("V3 context sampling profile drifted")
    building_seed = int(manifest["building_seed"])
    if building_seed not in BUILDING_SEEDS:
        raise ValueError(f"unsupported V3 building seed: {building_seed}")

    source_path = resolve_recorded_path(manifest["source_building_manifest"])
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    if file_sha256(source_path) != manifest["source_building_manifest_sha256"]:
        raise ValueError(f"V3 source manifest digest drift: {source_path}")
    artifact_path = resolve_recorded_path(manifest["context_artifact"])
    if file_sha256(artifact_path) != manifest["context_artifact_sha256"]:
        raise ValueError(f"V3 context artifact digest drift: {artifact_path}")

    cell = manifest["cells"][str(int(budget))]
    length = int(cell["context_rows"])
    if length != CONTEXT_ROWS[int(budget)]:
        raise ValueError(f"V3 context row target drifted for K={budget}")
    with np.load(artifact_path) as payload:
        raw_index = np.asarray(payload["raw_index"][:length], dtype="int64")
        anomaly = np.asarray(payload["anomaly"][:length], dtype="int8")
        building_id = np.asarray(payload["building_id"][:length], dtype="int64")
        meter = np.asarray(payload["meter"][:length], dtype="int8")
    selected_buildings = np.asarray(cell["selected_buildings"], dtype="int64")
    if len(selected_buildings) != int(budget):
        raise ValueError(f"V3 selected-building count drifted for K={budget}")
    if len(np.unique(raw_index)) != length:
        raise ValueError(f"V3 context repeats rows for K={budget}")
    if int_array_sha256(raw_index) != cell["raw_index_sha256"]:
        raise ValueError(f"V3 context row digest drifted for K={budget}")
    if int(anomaly.sum()) != length // 2 or int((anomaly == 0).sum()) != length // 2:
        raise ValueError(f"V3 context is not exactly 50:50 for K={budget}")
    if not np.isin(building_id, selected_buildings).all():
        raise ValueError(f"V3 context escaped building support for K={budget}")
    return BalancedContext(
        manifest_path=manifest_path,
        manifest=manifest,
        source_manifest_path=source_path,
        source_manifest=source_manifest,
        artifact_path=artifact_path,
        building_seed=building_seed,
        budget=int(budget),
        raw_index=raw_index,
        anomaly=anomaly,
        building_id=building_id,
        meter=meter,
        selected_buildings=selected_buildings,
    )


def verify_context_against_frame(context: BalancedContext, frame: Any) -> None:
    rows = context.raw_index
    observed_anomaly = frame.loc[rows, "anomaly"].to_numpy(dtype="int8")
    observed_building = frame.loc[rows, "building_id"].to_numpy(dtype="int64")
    observed_meter = frame.loc[rows, "meter"].to_numpy(dtype="int8")
    if not np.array_equal(observed_anomaly, context.anomaly):
        raise ValueError("V3 context anomaly identity differs from raw frame")
    if not np.array_equal(observed_building, context.building_id):
        raise ValueError("V3 context building identity differs from raw frame")
    if not np.array_equal(observed_meter, context.meter):
        raise ValueError("V3 context meter identity differs from raw frame")
    if np.any(observed_building % 2):
        raise ValueError("V3 context contains odd holdout buildings")
