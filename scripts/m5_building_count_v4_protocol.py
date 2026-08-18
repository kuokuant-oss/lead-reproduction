"""Strict frozen-input and K-major scheduling contract for M5 V4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lead import ROOT
from m5_building_curve_protocol import int_array_sha256
from prepare_m5_building_count_v4_fixed_10k import (
    BUDGETS,
    BUILDING_DRAW_SEEDS,
    CONTEXT_ROWS,
    EXPERIMENT_VERSION,
    RNG_ALGORITHM,
    ROW_DRAW_SEEDS,
    ROWS_PER_CLASS,
    SAMPLING_PROFILE,
    file_sha256,
    validate_context,
)

TRAINING_CONTEXT_POLICY = "frozen_unique_global_label_50_50_without_replacement"
CLASS_RATIO_POLICY = "exact_global_5000_anomaly_5000_normal"
CANONICAL_HOLDOUT_SHA256 = (
    "6cfebd1cb2bb818f69806c0f14d66a84b81c53d37a716badd48c17b86210d893"
)
VALIDATION_CONTEXTS = ((0, 0, 50), (4, 1, 400))


@dataclass(frozen=True)
class FixedContext:
    manifest_path: Path
    manifest: dict[str, Any]
    source_manifest_path: Path
    source_manifest: dict[str, Any]
    artifact_path: Path
    building_seed: int
    row_seed: int
    budget: int
    raw_index: np.ndarray
    anomaly: np.ndarray
    building_id: np.ndarray
    meter: np.ndarray
    selected_buildings: np.ndarray


def resolve_recorded_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def context_manifest_path(audit_root: Path, building_seed: int, row_seed: int) -> Path:
    return audit_root / (
        f"fixed_context_building_seed{building_seed}_row_seed{row_seed}.json"
    )


def k_major_contexts() -> list[tuple[int, int, int]]:
    """Return (building_seed, row_seed, K), completing every K before the next."""
    return [
        (building_seed, row_seed, budget)
        for budget in BUDGETS
        for building_seed in BUILDING_DRAW_SEEDS
        for row_seed in ROW_DRAW_SEEDS
    ]


def verify_training_context_gate(audit_root: Path) -> dict[str, Any]:
    path = audit_root / "training_context_gate.json"
    if not path.is_file():
        raise ValueError(f"V4 training-context gate is missing: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    expected_ladders = len(BUILDING_DRAW_SEEDS) * len(BUDGETS)
    expected_contexts = len(BUILDING_DRAW_SEEDS) * len(ROW_DRAW_SEEDS) * len(BUDGETS)
    checks = {
        "experiment_version": gate.get("experiment_version") == EXPERIMENT_VERSION,
        "passed": gate.get("passed") is True,
        "status": gate.get("status") == "PASSED",
        "building_seeds": tuple(map(int, gate.get("building_seeds", ())))
        == BUILDING_DRAW_SEEDS,
        "row_seeds": tuple(map(int, gate.get("row_seeds", ()))) == ROW_DRAW_SEEDS,
        "budgets": tuple(map(int, gate.get("budgets", ()))) == BUDGETS,
        "context_rows": int(gate.get("context_rows", -1)) == CONTEXT_ROWS,
        "rows_per_class": int(gate.get("rows_per_class", -1)) == ROWS_PER_CLASS,
        "rng_algorithm": gate.get("rng_algorithm") == RNG_ALGORITHM,
        "expected_ladder_cells": int(gate.get("expected_ladder_cells", -1))
        == expected_ladders,
        "checked_ladder_cells": int(gate.get("checked_ladder_cells", -1))
        == expected_ladders,
        "expected_context_cells": int(gate.get("expected_context_cells", -1))
        == expected_contexts,
        "checked_context_cells": int(gate.get("checked_context_cells", -1))
        == expected_contexts,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"V4 training-context gate failed: {failures}")
    return gate


def load_fixed_context(manifest_path: Path, budget: int) -> FixedContext:
    manifest_path = manifest_path.resolve()
    if int(budget) not in BUDGETS:
        raise ValueError(f"unsupported V4 building budget: {budget}")
    if validate_context(manifest_path) != len(BUDGETS):
        raise ValueError(f"V4 context manifest did not validate all K: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sampling_profile") != SAMPLING_PROFILE:
        raise ValueError("V4 context sampling profile drifted")
    building_seed = int(manifest["building_seed"])
    row_seed = int(manifest["row_seed"])
    if building_seed not in BUILDING_DRAW_SEEDS or row_seed not in ROW_DRAW_SEEDS:
        raise ValueError("unsupported V4 building/row seed identity")
    source_path = resolve_recorded_path(manifest["source_building_manifest"])
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    if file_sha256(source_path) != manifest["source_building_manifest_sha256"]:
        raise ValueError(f"V4 source manifest digest drift: {source_path}")
    artifact_path = resolve_recorded_path(manifest["context_artifact"])
    if file_sha256(artifact_path) != manifest["context_artifact_sha256"]:
        raise ValueError(f"V4 context artifact digest drift: {artifact_path}")
    with np.load(artifact_path) as payload:
        raw_index = np.asarray(payload[f"raw_index_k{budget}"], dtype="int64")
        anomaly = np.asarray(payload[f"anomaly_k{budget}"], dtype="int8")
        building_id = np.asarray(payload[f"building_id_k{budget}"], dtype="int64")
        meter = np.asarray(payload[f"meter_k{budget}"], dtype="int8")
    cell = manifest["cells"][str(int(budget))]
    selected_buildings = np.asarray(cell["selected_buildings"], dtype="int64")
    if len(raw_index) != CONTEXT_ROWS or len(np.unique(raw_index)) != CONTEXT_ROWS:
        raise ValueError(f"V4 context size/uniqueness drift at K={budget}")
    if int_array_sha256(raw_index) != cell["raw_index_sha256"]:
        raise ValueError(f"V4 context row digest drift at K={budget}")
    if (
        int(anomaly.sum()) != ROWS_PER_CLASS
        or int((anomaly == 0).sum()) != ROWS_PER_CLASS
    ):
        raise ValueError(f"V4 context is not exactly 50:50 at K={budget}")
    if not np.isin(building_id, selected_buildings).all():
        raise ValueError(f"V4 context escaped building support at K={budget}")
    return FixedContext(
        manifest_path=manifest_path,
        manifest=manifest,
        source_manifest_path=source_path,
        source_manifest=source_manifest,
        artifact_path=artifact_path,
        building_seed=building_seed,
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
        raise ValueError("V4 context anomaly identity differs from raw frame")
    if not np.array_equal(observed_building, context.building_id):
        raise ValueError("V4 context building identity differs from raw frame")
    if not np.array_equal(observed_meter, context.meter):
        raise ValueError("V4 context meter identity differs from raw frame")
    if np.any(observed_building % 2):
        raise ValueError("V4 context contains odd holdout buildings")
