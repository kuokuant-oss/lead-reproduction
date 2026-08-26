"""Prepare frozen 50K contexts for the M5 building-count V5 experiment.

The K=50/100/200/400 building supports are reused byte-for-byte from the
validated V4 building ladders. K=725 is one canonical support containing every
even (training) building. Rows are newly drawn for V5, uniformly without
replacement within the binary anomaly classes, with exactly 25K rows per
class. Meter is never used to select training rows.

The formal order is 725, 400, 200, 100, 50. K=725 has two row draws and no
building-draw replicate; every smaller K has five building draws and two row
draws. This program only prepares/validates inputs and never launches a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from lead import ROOT


SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "m5_building_count_v5_fixed_50k"
SOURCE_EXPERIMENT_VERSION = "m5_building_count_v4_fixed_10k"
SAMPLING_PROFILE = "reused_v4_nested_buildings_new_fixed_50k_global_50_50"
TRAINING_CONTEXT_POLICY = "frozen_unique_global_label_50_50_without_replacement"
CLASS_RATIO_POLICY = "exact_global_25000_anomaly_25000_normal"
RNG_ALGORITHM = "numpy.random.PCG64"
ROW_RNG_DOMAIN = 5_050_725

SCHEDULED_BUDGETS = (725, 400, 200, 100, 50)
NESTED_BUILDING_BUDGETS = (50, 100, 200, 400)
BUILDING_DRAW_SEEDS = tuple(range(5))
ROW_DRAW_SEEDS = (0, 1)
CONTEXT_ROWS = 50_000
ROWS_PER_CLASS = 25_000
EXPECTED_CANDIDATE_BUILDINGS = 725
FULL_SUPPORT_ID = "full_725"
EVALUATION_METER_IDS = (1, 2, 3)
EXCLUDED_EVALUATION_METER_IDS = (0,)

DEFAULT_RAW_ROOT = ROOT / "data/raw/m3"
DEFAULT_SOURCE_AUDIT_ROOT = (
    ROOT / "experiments/m5_building_count_v4_fixed_10k/audit"
)
DEFAULT_AUDIT_ROOT = ROOT / "experiments/m5_building_count_v5_fixed_50k/audit"
DEFAULT_CONTEXT_ROOT = (
    ROOT / "data/processed/m5_building_curve/v5_fixed_50k/training_contexts"
)
DEFAULT_HOLDOUT_ARTIFACT = (
    ROOT / "data/processed/m5_building_curve/v5_fixed_50k/"
    "canonical_holdout_non_electric.npz"
)
SOURCE_CANONICAL_HOLDOUT = (
    ROOT / "data/processed/m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"
)


@dataclass(frozen=True)
class ContextIdentity:
    budget: int
    building_seed: int | None
    row_seed: int
    support_id: str

    @property
    def key(self) -> str:
        building = "full" if self.building_seed is None else f"b{self.building_seed}"
        return f"k{self.budget}_{building}_r{self.row_seed}"


@dataclass(frozen=True)
class SourceSupport:
    selected_buildings: np.ndarray
    source_manifest_path: Path
    source_manifest_sha256: str
    source_cell_selected_sha256: str


def planned_contexts() -> list[ContextIdentity]:
    contexts = [
        ContextIdentity(725, None, row_seed, FULL_SUPPORT_ID)
        for row_seed in ROW_DRAW_SEEDS
    ]
    for seed_group in ((0, 1, 2), (3, 4)):
        contexts.extend(
            ContextIdentity(
                budget,
                building_seed,
                row_seed,
                f"building_seed{building_seed}",
            )
            for budget in SCHEDULED_BUDGETS[1:]
            for building_seed in seed_group
            for row_seed in ROW_DRAW_SEEDS
        )
    return contexts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def int_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="int64"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def byte_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.tobytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_recorded_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, lineterminator="\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("formal context preparation requires clean tracked sources")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_raw_identifiers(
    raw_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    train_path = raw_root / "train.csv"
    label_path = raw_root / "bad_meter_readings.csv"
    if not train_path.is_file() or not label_path.is_file():
        raise ValueError(f"M3 raw identity files are missing under {raw_root}")
    train = pd.read_csv(
        train_path,
        usecols=["building_id", "meter"],
        dtype={"building_id": "int32", "meter": "int8"},
    )
    labels = pd.read_csv(label_path)
    if tuple(labels.columns) != ("is_bad_meter_reading",):
        raise ValueError("bad_meter_readings.csv has an unexpected schema")
    if len(train) != len(labels):
        raise ValueError("M3 train rows and anomaly labels are not aligned")
    anomaly = labels["is_bad_meter_reading"].to_numpy(dtype="int8", copy=True)
    if not np.isin(anomaly, (0, 1)).all():
        raise ValueError("M3 anomaly labels must be binary")
    return (
        train["building_id"].to_numpy(dtype="int32", copy=True),
        train["meter"].to_numpy(dtype="int8", copy=True),
        anomaly,
        {
            "train_csv": display_path(train_path),
            "train_csv_sha256": sha256_file(train_path),
            "bad_meter_readings_csv": display_path(label_path),
            "bad_meter_readings_csv_sha256": sha256_file(label_path),
        },
    )


def load_source_supports(
    source_audit_root: Path,
    building: np.ndarray,
    raw_inputs: dict[str, str],
) -> tuple[dict[tuple[int, int], SourceSupport], SourceSupport, dict[str, Any]]:
    gate_path = source_audit_root / "training_context_gate.json"
    if not gate_path.is_file():
        raise ValueError(f"source V4 gate is missing: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("experiment_version") != SOURCE_EXPERIMENT_VERSION
        or gate.get("status") != "PASSED"
        or gate.get("passed") is not True
    ):
        raise ValueError("source V4 training-context gate is not PASSED")
    if int(gate.get("candidate_buildings", -1)) != EXPECTED_CANDIDATE_BUILDINGS:
        raise ValueError("source V4 candidate-building count drifted")
    for name in ("train_csv_sha256", "bad_meter_readings_csv_sha256"):
        if gate.get("raw_inputs", {}).get(name) != raw_inputs[name]:
            raise ValueError(f"source V4 raw input digest drifted: {name}")

    even_candidates = np.sort(np.unique(building[building % 2 == 0])).astype("int64")
    if len(even_candidates) != EXPECTED_CANDIDATE_BUILDINGS:
        raise ValueError("current even-building population is not exactly 725")
    candidate_digest = int_array_sha256(even_candidates)
    if gate.get("candidate_building_sha256") != candidate_digest:
        raise ValueError("source V4 candidate-building identity drifted")

    supports: dict[tuple[int, int], SourceSupport] = {}
    for building_seed in BUILDING_DRAW_SEEDS:
        path = source_audit_root / f"building_ladder_seed{building_seed}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("experiment_version") != SOURCE_EXPERIMENT_VERSION:
            raise ValueError(f"source ladder experiment drifted: {path}")
        candidates = np.asarray(manifest["candidate_buildings"], dtype="int64")
        if not np.array_equal(np.sort(candidates), even_candidates):
            raise ValueError(f"source ladder candidate identity drifted: {path}")
        previous = np.empty(0, dtype="int64")
        manifest_sha = sha256_file(path)
        for budget in NESTED_BUILDING_BUDGETS:
            cell = manifest["cells"][str(budget)]
            selected = np.asarray(cell["available_buildings"], dtype="int64")
            if len(selected) != budget or len(np.unique(selected)) != budget:
                raise ValueError(f"invalid source support seed={building_seed} K={budget}")
            if len(previous) and not np.array_equal(selected[: len(previous)], previous):
                raise ValueError(f"source ladder is not nested seed={building_seed}")
            if np.any(selected % 2) or not np.isin(selected, even_candidates).all():
                raise ValueError("source ladder escaped the even-building population")
            digest = int_array_sha256(selected)
            if digest != cell["available_building_sha256"]:
                raise ValueError("source selected-building digest drifted")
            supports[(building_seed, budget)] = SourceSupport(
                selected_buildings=selected,
                source_manifest_path=path.resolve(),
                source_manifest_sha256=manifest_sha,
                source_cell_selected_sha256=digest,
            )
            previous = selected

    full_support = SourceSupport(
        selected_buildings=even_candidates,
        source_manifest_path=gate_path.resolve(),
        source_manifest_sha256=sha256_file(gate_path),
        source_cell_selected_sha256=candidate_digest,
    )
    source = {
        "source_gate": display_path(gate_path),
        "source_gate_sha256": sha256_file(gate_path),
        "candidate_buildings": len(even_candidates),
        "candidate_building_sha256": candidate_digest,
    }
    return supports, full_support, source


def prepare_full_support_manifest(
    full_support: SourceSupport,
    *,
    audit_root: Path,
    raw_inputs: dict[str, str],
    preparation_commit: str,
) -> SourceSupport:
    path = audit_root / "building_support_full_725.json"
    selected = full_support.selected_buildings
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "m5_building_count_v5_full_building_support",
        "experiment": "m5_building_count_v5_full_support",
        "experiment_version": EXPERIMENT_VERSION,
        "status": "FROZEN",
        "sampling_profile": SAMPLING_PROFILE,
        "support_id": FULL_SUPPORT_ID,
        "building_seed": 725,
        "building_draw_seed": None,
        "operational_building_seed_sentinel": 725,
        "selection_rule": "all_even_training_buildings",
        "candidate_buildings": selected.tolist(),
        "candidate_building_sha256": int_array_sha256(selected),
        "raw_inputs": raw_inputs,
        "preparation_source_commit": preparation_commit,
        "cells": {
            "725": {
                "K": 725,
                "constraint_pass": True,
                "available_buildings": selected.tolist(),
                "available_building_sha256": int_array_sha256(selected),
                "tree_fit_buildings": selected.tolist(),
                "tree_early_stop_buildings": [],
            }
        },
    }
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"full-support manifest provenance mismatch: {path}")
    else:
        atomic_json(path, payload)
    return SourceSupport(
        selected_buildings=selected,
        source_manifest_path=path.resolve(),
        source_manifest_sha256=sha256_file(path),
        source_cell_selected_sha256=int_array_sha256(selected),
    )


def resolve_source_canonical_holdout() -> Path:
    if SOURCE_CANONICAL_HOLDOUT.is_file():
        return SOURCE_CANONICAL_HOLDOUT
    fallback = (
        ROOT.parent
        / "lead-reproduction"
        / "data/processed"
        / SOURCE_CANONICAL_HOLDOUT.name
    )
    if not fallback.is_file():
        raise ValueError("canonical M5 holdout artifact is missing")
    return fallback


def prepare_evaluation_holdout(
    *,
    building: np.ndarray,
    meter: np.ndarray,
    anomaly: np.ndarray,
    artifact_path: Path,
    audit_root: Path,
    preparation_commit: str,
) -> dict[str, Any]:
    source_path = resolve_source_canonical_holdout()
    with np.load(source_path) as payload:
        required = {"validation_raw_index", "anomaly", "site_id", "building_id"}
        if not required.issubset(payload.files):
            raise ValueError("source canonical holdout schema drifted")
        raw_index = np.asarray(payload["validation_raw_index"], dtype="int64")
        source_anomaly = np.asarray(payload["anomaly"], dtype="int8")
        site_id = np.asarray(payload["site_id"], dtype="int8")
        source_building = np.asarray(payload["building_id"], dtype="int16")
    if (
        not np.array_equal(source_anomaly, anomaly[raw_index])
        or not np.array_equal(source_building, building[raw_index])
    ):
        raise ValueError("source canonical holdout differs from raw frame identity")
    row_meter = meter[raw_index]
    selected_mask = np.isin(row_meter, EVALUATION_METER_IDS)
    arrays = {
        "validation_raw_index": raw_index[selected_mask],
        "anomaly": source_anomaly[selected_mask],
        "site_id": site_id[selected_mask],
        "building_id": source_building[selected_mask],
    }
    if np.any(building[arrays["validation_raw_index"]] % 2 == 0):
        raise ValueError("V5 evaluation holdout contains an even training building")
    if set(map(int, np.unique(meter[arrays["validation_raw_index"]]))) != set(
        EVALUATION_METER_IDS
    ):
        raise ValueError("V5 evaluation holdout meter identity drifted")
    if artifact_path.is_file():
        with np.load(artifact_path) as existing:
            if set(existing.files) != set(arrays) or any(
                not np.array_equal(np.asarray(existing[name]), value)
                for name, value in arrays.items()
            ):
                raise ValueError("existing V5 evaluation holdout is incompatible")
    else:
        atomic_savez(artifact_path, **arrays)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "m5_building_count_v5_evaluation_holdout",
        "experiment_version": EXPERIMENT_VERSION,
        "status": "FROZEN",
        "preparation_source_commit": preparation_commit,
        "source_canonical_holdout": display_path(source_path),
        "source_canonical_holdout_sha256": sha256_file(source_path),
        "artifact": display_path(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "validation_raw_index_sha256": int_array_sha256(
            arrays["validation_raw_index"]
        ),
        "rows": int(len(arrays["validation_raw_index"])),
        "anomalies": int(arrays["anomaly"].sum()),
        "meter_ids": list(EVALUATION_METER_IDS),
        "excluded_meter_ids": list(EXCLUDED_EVALUATION_METER_IDS),
        "meter_row_counts": {
            str(value): int(
                np.sum(meter[arrays["validation_raw_index"]] == value)
            )
            for value in EVALUATION_METER_IDS
        },
    }
    manifest_path = audit_root / "evaluation_holdout.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("V5 evaluation-holdout manifest provenance mismatch")
    else:
        atomic_json(manifest_path, manifest)
    return manifest


def row_seed_material(identity: ContextIdentity, label: int) -> tuple[int, ...]:
    support_code = 725 if identity.building_seed is None else identity.building_seed
    return (
        SCHEMA_VERSION,
        ROW_RNG_DOMAIN,
        identity.budget,
        support_code,
        identity.row_seed,
        int(label),
    )


def draw_context(
    building: np.ndarray,
    anomaly: np.ndarray,
    selected_buildings: np.ndarray,
    identity: ContextIdentity,
) -> tuple[np.ndarray, dict[str, int]]:
    member = np.isin(building, selected_buildings)
    raw_index = np.arange(len(building), dtype="int64")
    selected: dict[int, np.ndarray] = {}
    support: dict[int, int] = {}
    for label in (1, 0):
        candidates = raw_index[member & (anomaly == label)]
        support[label] = int(len(candidates))
        if len(candidates) < ROWS_PER_CLASS:
            raise ValueError(
                "insufficient unique class support without replacement: "
                f"context={identity.key}, label={label}, available={len(candidates)}, "
                f"required={ROWS_PER_CLASS}"
            )
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(row_seed_material(identity, label)))
        )
        selected[label] = rng.choice(
            candidates, size=ROWS_PER_CLASS, replace=False, shuffle=True
        ).astype("int64", copy=False)
    rows = np.empty(CONTEXT_ROWS, dtype="int64")
    rows[0::2] = selected[1]
    rows[1::2] = selected[0]
    if len(np.unique(rows)) != CONTEXT_ROWS:
        raise AssertionError("V5 context repeats raw rows")
    return rows, {"full_anomalies": support[1], "full_normals": support[0]}


def meter_counts(meter: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in range(4):
        mask = meter == value
        rows = int(mask.sum())
        positives = int(labels[mask].sum())
        result[str(value)] = {
            "rows": rows,
            "anomalies": positives,
            "normals": rows - positives,
            "anomaly_rate": positives / rows if rows else None,
        }
    return result


def context_paths(
    identity: ContextIdentity, audit_root: Path, context_root: Path
) -> tuple[Path, Path]:
    manifest_path = audit_root / "contexts" / f"{identity.key}.json"
    support = FULL_SUPPORT_ID if identity.building_seed is None else identity.support_id
    artifact_path = (
        context_root
        / support
        / f"row_seed{identity.row_seed}"
        / f"fixed_50k_k{identity.budget}.npz"
    )
    return manifest_path, artifact_path


def expected_identity(
    identity: ContextIdentity,
    support: SourceSupport,
    raw_inputs: dict[str, str],
    preparation_commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "sampling_profile": SAMPLING_PROFILE,
        "budget": identity.budget,
        "building_seed": identity.building_seed,
        "row_seed": identity.row_seed,
        "support_id": identity.support_id,
        "context_rows": CONTEXT_ROWS,
        "rows_per_class": ROWS_PER_CLASS,
        "rng_algorithm": RNG_ALGORITHM,
        "row_rng_domain": ROW_RNG_DOMAIN,
        "selected_building_sha256": support.source_cell_selected_sha256,
        "source_manifest_sha256": support.source_manifest_sha256,
        "train_csv_sha256": raw_inputs["train_csv_sha256"],
        "bad_meter_readings_csv_sha256": raw_inputs[
            "bad_meter_readings_csv_sha256"
        ],
        "preparation_source_commit": preparation_commit,
    }


def validate_context(
    manifest_path: Path,
    *,
    expected: dict[str, Any],
    support: SourceSupport,
    building: np.ndarray,
    meter: np.ndarray,
    anomaly: np.ndarray,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN" or manifest.get("identity") != expected:
        raise ValueError(f"context identity/provenance mismatch: {manifest_path}")
    artifact_path = resolve_recorded_path(manifest["context_artifact"])
    if not artifact_path.is_file():
        raise ValueError(f"context artifact is missing: {artifact_path}")
    if sha256_file(artifact_path) != manifest["context_artifact_sha256"]:
        raise ValueError(f"context artifact digest drift: {artifact_path}")
    with np.load(artifact_path) as payload:
        required = {"raw_index", "anomaly", "building_id", "meter", "selected_buildings"}
        if set(payload.files) != required:
            raise ValueError(f"context artifact schema drift: {artifact_path}")
        rows = np.asarray(payload["raw_index"], dtype="int64")
        labels = np.asarray(payload["anomaly"], dtype="int8")
        row_building = np.asarray(payload["building_id"], dtype="int64")
        row_meter = np.asarray(payload["meter"], dtype="int8")
        selected = np.asarray(payload["selected_buildings"], dtype="int64")
    if len(rows) != CONTEXT_ROWS or len(np.unique(rows)) != CONTEXT_ROWS:
        raise ValueError(f"context size/uniqueness drift: {manifest_path}")
    if int(labels.sum()) != ROWS_PER_CLASS or int((labels == 0).sum()) != ROWS_PER_CLASS:
        raise ValueError(f"context class balance drift: {manifest_path}")
    if not np.array_equal(selected, support.selected_buildings):
        raise ValueError(f"selected-building identity drift: {manifest_path}")
    if not np.isin(row_building, selected).all() or np.any(row_building % 2):
        raise ValueError(f"context escaped its even-building support: {manifest_path}")
    if (
        not np.array_equal(labels, anomaly[rows])
        or not np.array_equal(row_building, building[rows])
        or not np.array_equal(row_meter, meter[rows])
    ):
        raise ValueError(f"context raw-row identity drift: {manifest_path}")
    identity = ContextIdentity(
        budget=int(expected["budget"]),
        building_seed=expected["building_seed"],
        row_seed=int(expected["row_seed"]),
        support_id=str(expected["support_id"]),
    )
    deterministic_rows, _ = draw_context(building, anomaly, selected, identity)
    if not np.array_equal(rows, deterministic_rows):
        raise ValueError(f"context differs from deterministic row draw: {manifest_path}")
    checks = {
        "raw_index_sha256": int_array_sha256(rows),
        "anomaly_sha256": byte_array_sha256(labels),
        "building_id_sha256": int_array_sha256(row_building),
        "meter_sha256": byte_array_sha256(row_meter),
    }
    for name, value in checks.items():
        if manifest.get(name) != value:
            raise ValueError(f"context recorded {name} drift: {manifest_path}")
    return manifest


def prepare_context(
    identity: ContextIdentity,
    *,
    support: SourceSupport,
    building: np.ndarray,
    meter: np.ndarray,
    anomaly: np.ndarray,
    raw_inputs: dict[str, str],
    preparation_commit: str,
    audit_root: Path,
    context_root: Path,
) -> tuple[dict[str, Any], bool]:
    manifest_path, artifact_path = context_paths(identity, audit_root, context_root)
    expected = expected_identity(identity, support, raw_inputs, preparation_commit)
    if manifest_path.is_file():
        return (
            validate_context(
                manifest_path,
                expected=expected,
                support=support,
                building=building,
                meter=meter,
                anomaly=anomaly,
            ),
            True,
        )

    rows, capacity = draw_context(
        building, anomaly, support.selected_buildings, identity
    )
    labels = anomaly[rows].astype("int8", copy=False)
    row_building = building[rows].astype("int32", copy=False)
    row_meter = meter[rows].astype("int8", copy=False)
    arrays = {
        "raw_index": rows,
        "anomaly": labels,
        "building_id": row_building,
        "meter": row_meter,
        "selected_buildings": support.selected_buildings.astype("int32", copy=False),
    }
    if artifact_path.is_file():
        with np.load(artifact_path) as existing:
            if set(existing.files) != set(arrays) or any(
                not np.array_equal(np.asarray(existing[name]), value)
                for name, value in arrays.items()
            ):
                raise ValueError(
                    f"orphan context artifact conflicts with deterministic draw: {artifact_path}"
                )
    else:
        atomic_savez(artifact_path, **arrays)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "m5_building_count_v5_fixed_50k_context",
        "status": "FROZEN",
        "identity": expected,
        "training_sampling": TRAINING_CONTEXT_POLICY,
        "class_ratio_policy": CLASS_RATIO_POLICY,
        "sampling_inputs": ["selected_building_membership", "anomaly"],
        "ignored_for_sampling": ["meter", "site_id", "building_row_quota"],
        "sampling_without_replacement": True,
        "retry_or_repair": False,
        "source_manifest": display_path(support.source_manifest_path),
        "source_manifest_sha256": support.source_manifest_sha256,
        "selected_buildings": support.selected_buildings.tolist(),
        "selected_building_count": int(len(support.selected_buildings)),
        "selected_building_sha256": int_array_sha256(support.selected_buildings),
        "context_artifact": display_path(artifact_path),
        "context_artifact_sha256": sha256_file(artifact_path),
        "raw_index_sha256": int_array_sha256(rows),
        "anomaly_sha256": byte_array_sha256(labels),
        "building_id_sha256": int_array_sha256(row_building),
        "meter_sha256": byte_array_sha256(row_meter),
        "row_rng_seed_material": {
            str(label): list(row_seed_material(identity, label)) for label in (1, 0)
        },
        "full_anomalies": capacity["full_anomalies"],
        "full_normals": capacity["full_normals"],
        "observed_building_count": int(len(np.unique(row_building))),
        "meter_composition": meter_counts(row_meter, labels),
        "evaluation_meter_ids": list(EVALUATION_METER_IDS),
        "excluded_evaluation_meter_ids": list(EXCLUDED_EVALUATION_METER_IDS),
    }
    atomic_json(manifest_path, manifest)
    validate_context(
        manifest_path,
        expected=expected,
        support=support,
        building=building,
        meter=meter,
        anomaly=anomaly,
    )
    return manifest, False


def support_for(
    identity: ContextIdentity,
    supports: dict[tuple[int, int], SourceSupport],
    full_support: SourceSupport,
) -> SourceSupport:
    if identity.building_seed is None:
        return full_support
    return supports[(identity.building_seed, identity.budget)]


def finalize_gate(
    records: Iterable[dict[str, Any]],
    *,
    contexts: list[ContextIdentity],
    raw_inputs: dict[str, str],
    source: dict[str, Any],
    evaluation_holdout: dict[str, Any],
    preparation_commit: str,
    audit_root: Path,
    mode: str,
) -> dict[str, Any]:
    rows = list(records)
    if len(rows) != len(contexts):
        raise ValueError(
            f"refusing finalization with missing contexts: {len(rows)}/{len(contexts)}"
        )
    by_key = {row["identity"]["support_id"] + f"|{row['identity']['budget']}": {} for row in rows}
    for row in rows:
        key = row["identity"]["support_id"] + f"|{row['identity']['budget']}"
        by_key[key][int(row["identity"]["row_seed"])] = row["raw_index_sha256"]
    if any(set(pair) != set(ROW_DRAW_SEEDS) for pair in by_key.values()):
        raise ValueError("row-seed pair census is incomplete")
    if any(pair[0] == pair[1] for pair in by_key.values()):
        raise ValueError("row seeds produced identical ordered contexts")
    expected = 2 + 4 * 5 * 2
    if len(contexts) != expected:
        raise AssertionError("V5 context census is not 42")
    gate = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "m5_building_count_v5_training_context_gate",
        "experiment_version": EXPERIMENT_VERSION,
        "status": "PASSED",
        "passed": True,
        "mode": mode,
        "preparation_source_commit": preparation_commit,
        "sampling_profile": SAMPLING_PROFILE,
        "training_sampling": TRAINING_CONTEXT_POLICY,
        "class_ratio_policy": CLASS_RATIO_POLICY,
        "scheduled_budgets": list(SCHEDULED_BUDGETS),
        "building_seeds_for_k50_to_k400": list(BUILDING_DRAW_SEEDS),
        "k725_building_seed": None,
        "row_seeds": list(ROW_DRAW_SEEDS),
        "context_rows": CONTEXT_ROWS,
        "rows_per_class": ROWS_PER_CLASS,
        "expected_context_cells": expected,
        "checked_context_cells": len(rows),
        "expected_model_cells": expected * 2,
        "candidate_buildings": source["candidate_buildings"],
        "candidate_building_sha256": source["candidate_building_sha256"],
        "source_gate": source["source_gate"],
        "source_gate_sha256": source["source_gate_sha256"],
        "raw_inputs": raw_inputs,
        "rng_algorithm": RNG_ALGORITHM,
        "row_rng_domain": ROW_RNG_DOMAIN,
        "training_meter_ids": [0, 1, 2, 3],
        "evaluation_meter_ids": list(EVALUATION_METER_IDS),
        "excluded_evaluation_meter_ids": list(EXCLUDED_EVALUATION_METER_IDS),
        "evaluation_holdout": evaluation_holdout,
        "phase_plan": [
            {
                "phase": 1,
                "description": "K725 full support, row seeds 0 and 1",
                "contexts": 2,
                "model_cells": 4,
            },
            {
                "phase": 2,
                "description": "building seeds 0-2, K400 to K50",
                "contexts": 24,
                "model_cells": 48,
            },
            {
                "phase": 3,
                "description": "building seeds 3-4, K400 to K50",
                "contexts": 16,
                "model_cells": 32,
            },
        ],
        "context_keys_in_run_order": [identity.key for identity in contexts],
    }
    atomic_json(audit_root / "training_context_gate.json", gate)
    atomic_json(audit_root / "PREPARATION_COMPLETE.json", gate)
    return gate


def prepare_all(args: argparse.Namespace, *, mode: str) -> dict[str, Any]:
    failure_path = args.audit_root / "FAILED.json"
    if failure_path.exists():
        raise ValueError(f"preparation is blocked by existing failure marker: {failure_path}")
    commit = source_commit()
    building, meter, anomaly, raw_inputs = load_raw_identifiers(args.raw_root)
    supports, full_support, source = load_source_supports(
        args.source_audit_root, building, raw_inputs
    )
    full_support = prepare_full_support_manifest(
        full_support,
        audit_root=args.audit_root,
        raw_inputs=raw_inputs,
        preparation_commit=commit,
    )
    evaluation_holdout = prepare_evaluation_holdout(
        building=building,
        meter=meter,
        anomaly=anomaly,
        artifact_path=args.holdout_artifact,
        audit_root=args.audit_root,
        preparation_commit=commit,
    )
    contexts = planned_contexts()
    if mode == "validation":
        contexts = contexts[: args.max_units]
    records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        for index, identity in enumerate(contexts):
            atomic_json(
                args.audit_root / "heartbeat.json",
                {
                    "status": "running",
                    "completed_units": index,
                    "total_units": len(contexts),
                    "current_unit": identity.key,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            record, reused = prepare_context(
                identity,
                support=support_for(identity, supports, full_support),
                building=building,
                meter=meter,
                anomaly=anomaly,
                raw_inputs=raw_inputs,
                preparation_commit=commit,
                audit_root=args.audit_root,
                context_root=args.context_root,
            )
            records.append(record)
            audit_rows.append(
                {
                    **asdict(identity),
                    "context_rows": CONTEXT_ROWS,
                    "anomalies": ROWS_PER_CLASS,
                    "normals": ROWS_PER_CLASS,
                    "selected_buildings": record["selected_building_count"],
                    "observed_buildings": record["observed_building_count"],
                    "full_anomalies": record["full_anomalies"],
                    "full_normals": record["full_normals"],
                    "raw_index_sha256": record["raw_index_sha256"],
                    "reused": reused,
                }
            )
            elapsed = time.monotonic() - started
            rate = elapsed / (index + 1)
            print(
                f"prepared {index + 1}/{len(contexts)} {identity.key} "
                f"reused={reused} seconds_per_unit={rate:.2f}",
                flush=True,
            )
        atomic_csv(args.audit_root / "context_census.csv", pd.DataFrame(audit_rows))
        if mode == "prepare":
            gate = finalize_gate(
                records,
                contexts=contexts,
                raw_inputs=raw_inputs,
                source=source,
                evaluation_holdout=evaluation_holdout,
                preparation_commit=commit,
                audit_root=args.audit_root,
                mode="FORMAL_INPUT_PREPARATION",
            )
        else:
            gate = {
                "schema_version": SCHEMA_VERSION,
                "experiment_version": EXPERIMENT_VERSION,
                "status": "PASSED",
                "passed": True,
                "mode": "NON_SCIENTIFIC_VALIDATION",
                "checked_context_cells": len(records),
                "max_units": args.max_units,
                "preparation_source_commit": commit,
            }
            atomic_json(args.audit_root / "VALIDATION_COMPLETE.json", gate)
        atomic_json(
            args.audit_root / "heartbeat.json",
            {
                "status": "complete",
                "completed_units": len(contexts),
                "total_units": len(contexts),
                "current_unit": None,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        return gate
    except BaseException as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "experiment_version": EXPERIMENT_VERSION,
            "status": "BLOCKED",
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_units": len(records),
            "total_units": len(contexts),
        }
        atomic_json(args.audit_root / "heartbeat.json", failure)
        atomic_json(failure_path, failure)
        raise


def check_all(args: argparse.Namespace) -> dict[str, Any]:
    gate_path = args.audit_root / "training_context_gate.json"
    if not gate_path.is_file():
        raise ValueError(f"training-context gate is missing: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASSED" or gate.get("passed") is not True:
        raise ValueError("training-context gate is not PASSED")
    building, meter, anomaly, raw_inputs = load_raw_identifiers(args.raw_root)
    supports, full_support, source = load_source_supports(
        args.source_audit_root, building, raw_inputs
    )
    full_support = prepare_full_support_manifest(
        full_support,
        audit_root=args.audit_root,
        raw_inputs=raw_inputs,
        preparation_commit=gate["preparation_source_commit"],
    )
    evaluation_holdout = prepare_evaluation_holdout(
        building=building,
        meter=meter,
        anomaly=anomaly,
        artifact_path=args.holdout_artifact,
        audit_root=args.audit_root,
        preparation_commit=gate["preparation_source_commit"],
    )
    contexts = planned_contexts()
    records = []
    for identity in contexts:
        support = support_for(identity, supports, full_support)
        manifest_path, _ = context_paths(identity, args.audit_root, args.context_root)
        records.append(
            validate_context(
                manifest_path,
                expected=expected_identity(
                    identity,
                    support,
                    raw_inputs,
                    gate["preparation_source_commit"],
                ),
                support=support,
                building=building,
                meter=meter,
                anomaly=anomaly,
            )
        )
    checks = {
        "passed": len(records) == 42,
        "checked_context_cells": len(records),
        "expected_context_cells": 42,
        "candidate_building_sha256": source["candidate_building_sha256"],
        "preparation_source_commit": gate["preparation_source_commit"],
        "evaluation_holdout_rows": evaluation_holdout["rows"],
        "evaluation_holdout_row_sha256": evaluation_holdout[
            "validation_raw_index_sha256"
        ],
    }
    if not checks["passed"]:
        raise ValueError("formal V5 context census is incomplete")
    return checks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("plan", "validation", "prepare", "check"), default="plan"
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--source-audit-root", type=Path, default=DEFAULT_SOURCE_AUDIT_ROOT
    )
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--context-root", type=Path, default=DEFAULT_CONTEXT_ROOT)
    parser.add_argument(
        "--holdout-artifact", type=Path, default=DEFAULT_HOLDOUT_ARTIFACT
    )
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--authorize-prepare", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        if not args.authorize_prepare:
            parser.error("formal input preparation requires --authorize-prepare")
        if args.max_units is not None:
            parser.error("formal input preparation cannot use --max-units")
    elif args.authorize_prepare:
        parser.error("--authorize-prepare is only valid in prepare mode")
    if args.mode == "validation":
        if args.max_units is None or not 1 <= args.max_units <= len(planned_contexts()):
            parser.error("validation requires --max-units between 1 and 42")
        if "NON_SCIENTIFIC_VALIDATION" not in str(args.audit_root) or (
            "NON_SCIENTIFIC_VALIDATION" not in str(args.context_root)
        ) or (
            "NON_SCIENTIFIC_VALIDATION" not in str(args.holdout_artifact)
        ):
            parser.error("validation roots must be visibly NON_SCIENTIFIC_VALIDATION")
    elif args.max_units is not None:
        parser.error("--max-units is only valid in validation mode")
    return args


def plan() -> dict[str, Any]:
    contexts = planned_contexts()
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "scheduled_budgets": list(SCHEDULED_BUDGETS),
        "contexts": len(contexts),
        "model_cells_when_launched": len(contexts) * 2,
        "context_rows": CONTEXT_ROWS,
        "rows_per_class": ROWS_PER_CLASS,
        "k725_contexts": 2,
        "k50_to_k400_contexts_per_budget": 10,
        "evaluation_meter_ids": list(EVALUATION_METER_IDS),
        "context_keys_in_run_order": [context.key for context in contexts],
        "launches_models": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "plan":
        result = plan()
    elif args.mode in {"prepare", "validation"}:
        result = prepare_all(args, mode=args.mode)
    else:
        result = check_all(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
