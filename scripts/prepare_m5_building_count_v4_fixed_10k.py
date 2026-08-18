"""Prepare frozen 10K, globally balanced contexts for building-count V4.

Building ladders are new label-blind PCG64 permutations of every even training
building. Rows are sampled without replacement using only membership in the
selected K-building support and the binary anomaly label. No composition
constraint, repair, retry, or redraw is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lead import ROOT

try:
    from m5_building_curve_protocol import atomic_write_json, int_array_sha256
except ModuleNotFoundError:
    from scripts.m5_building_curve_protocol import atomic_write_json, int_array_sha256


SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "m5_building_count_v4_fixed_10k"
SAMPLING_PROFILE = "new_nested_random_buildings_fixed_10k_global_50_50"
BUILDING_DRAW_SEEDS = tuple(range(5))
ROW_DRAW_SEEDS = (0, 1)
BUDGETS = (50, 100, 200, 300, 400)
CONTEXT_ROWS = 10_000
ROWS_PER_CLASS = 5_000
RNG_ALGORITHM = "numpy.random.PCG64"
DEFAULT_RAW_ROOT = ROOT / "data/raw/m3"
DEFAULT_AUDIT_ROOT = ROOT / "experiments/m5_building_count_v4_fixed_10k/audit"
DEFAULT_CONTEXT_ROOT = (
    ROOT / "data/processed/m5_building_curve/v4_fixed_10k/training_contexts"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, lineterminator="\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


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
            "train_csv": _display_path(train_path),
            "train_csv_sha256": file_sha256(train_path),
            "bad_meter_readings_csv": _display_path(label_path),
            "bad_meter_readings_csv_sha256": file_sha256(label_path),
        },
    )


def build_ladder(
    candidates: np.ndarray, *, building_seed: int
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    values = np.asarray(candidates, dtype="int64")
    if len(values) < BUDGETS[-1] or len(np.unique(values)) != len(values):
        raise ValueError("candidate building population is insufficient or duplicated")
    if np.any(values % 2):
        raise ValueError("V4 building candidate population contains odd IDs")
    ordered_candidates = np.sort(values)
    rng = np.random.Generator(np.random.PCG64(building_seed))
    permutation = rng.permutation(ordered_candidates).astype("int64", copy=False)
    cells = {budget: permutation[:budget].copy() for budget in BUDGETS}
    previous = np.empty(0, dtype="int64")
    for budget, selected in cells.items():
        if len(selected) != budget or len(np.unique(selected)) != budget:
            raise AssertionError(f"invalid V4 building prefix at K={budget}")
        if len(previous) and not np.array_equal(selected[: len(previous)], previous):
            raise AssertionError(f"V4 building ladder is not nested at K={budget}")
        previous = selected
    return permutation, cells


def _row_seed_material(
    building_seed: int, row_seed: int, budget: int, label: int
) -> tuple[int, ...]:
    return (SCHEMA_VERSION, int(building_seed), int(row_seed), int(budget), int(label))


def draw_context(
    building: np.ndarray,
    anomaly: np.ndarray,
    selected_buildings: np.ndarray,
    *,
    building_seed: int,
    row_seed: int,
    budget: int,
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
                f"building_seed={building_seed}, row_seed={row_seed}, K={budget}, "
                f"label={label}, available={len(candidates)}, required={ROWS_PER_CLASS}"
            )
        material = _row_seed_material(building_seed, row_seed, budget, label)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(material)))
        selected[label] = rng.choice(
            candidates, size=ROWS_PER_CLASS, replace=False, shuffle=True
        ).astype("int64", copy=False)
    rows = np.empty(CONTEXT_ROWS, dtype="int64")
    rows[0::2] = selected[1]
    rows[1::2] = selected[0]
    if len(np.unique(rows)) != CONTEXT_ROWS:
        raise AssertionError("V4 context repeats raw rows")
    return rows, {"full_anomalies": support[1], "full_normals": support[0]}


def _meter_counts(meter: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in sorted(map(int, np.unique(meter))):
        mask = meter == value
        rows = int(mask.sum())
        positives = int(labels[mask].sum())
        result[str(value)] = {
            "rows": rows,
            "anomalies": positives,
            "anomaly_rate": positives / rows if rows else None,
        }
    return result


def prepare_building_ladder(
    *,
    building_seed: int,
    building: np.ndarray,
    anomaly: np.ndarray,
    candidates: np.ndarray,
    audit_root: Path,
    raw_inputs: dict[str, str],
) -> tuple[Path, dict[int, np.ndarray], list[dict[str, Any]]]:
    permutation, cells = build_ladder(candidates, building_seed=building_seed)
    cell_records: dict[str, Any] = {}
    audit: list[dict[str, Any]] = []
    for budget, selected in cells.items():
        member = np.isin(building, selected)
        labels = anomaly[member]
        record = {
            "K": budget,
            "constraint_pass": True,
            "available_buildings": selected.tolist(),
            "available_building_sha256": int_array_sha256(selected),
            "tree_fit_buildings": selected.tolist(),
            "tree_early_stop_buildings": [],
            "full_rows": int(member.sum()),
            "full_anomalies": int(labels.sum()),
            "full_normals": int(len(labels) - labels.sum()),
        }
        cell_records[str(budget)] = record
        audit.append(
            {
                "building_seed": building_seed,
                "K": budget,
                "full_rows": record["full_rows"],
                "full_anomalies": record["full_anomalies"],
                "full_normals": record["full_normals"],
                "selected_building_sha256": record["available_building_sha256"],
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "m5_building_count_v4_building_ladder",
        "experiment": "m5_building_count_v4_building_sampling",
        "experiment_version": EXPERIMENT_VERSION,
        "status": "FROZEN",
        "sampling_profile": SAMPLING_PROFILE,
        "building_seed": building_seed,
        "building_draw_seed": building_seed,
        "rng_algorithm": RNG_ALGORITHM,
        "candidate_order": "ascending_building_id_before_permutation",
        "selection_rule": "single_permutation_then_strict_K_prefixes",
        "selection_inputs": ["even_building_membership", "building_draw_seed"],
        "selection_constraints": [],
        "retry_or_repair": False,
        "budgets": list(BUDGETS),
        "candidate_buildings": candidates.tolist(),
        "candidate_building_sha256": int_array_sha256(np.sort(candidates)),
        "permutation_sha256": int_array_sha256(permutation),
        "raw_inputs": raw_inputs,
        "cells": cell_records,
    }
    path = audit_root / f"building_ladder_seed{building_seed}.json"
    atomic_write_json(path, manifest)
    return path, cells, audit


def prepare_context_pair(
    *,
    building_seed: int,
    row_seed: int,
    building: np.ndarray,
    meter: np.ndarray,
    anomaly: np.ndarray,
    ladder_path: Path,
    ladder_cells: dict[int, np.ndarray],
    audit_root: Path,
    context_root: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    arrays: dict[str, np.ndarray] = {"budgets": np.asarray(BUDGETS, dtype="int16")}
    cells: dict[str, Any] = {}
    audit: list[dict[str, Any]] = []
    for budget in BUDGETS:
        selected_buildings = ladder_cells[budget]
        rows, support = draw_context(
            building,
            anomaly,
            selected_buildings,
            building_seed=building_seed,
            row_seed=row_seed,
            budget=budget,
        )
        labels = anomaly[rows].astype("int8", copy=False)
        row_buildings = building[rows].astype("int32", copy=False)
        row_meter = meter[rows].astype("int8", copy=False)
        arrays[f"raw_index_k{budget}"] = rows
        arrays[f"anomaly_k{budget}"] = labels
        arrays[f"building_id_k{budget}"] = row_buildings
        arrays[f"meter_k{budget}"] = row_meter
        observed = np.unique(row_buildings)
        cell = {
            "K": budget,
            "context_rows": CONTEXT_ROWS,
            "anomalies": int(labels.sum()),
            "normals": int((labels == 0).sum()),
            "anomaly_rate": float(labels.mean()),
            "raw_index_sha256": int_array_sha256(rows),
            "selected_buildings": selected_buildings.tolist(),
            "selected_building_sha256": int_array_sha256(selected_buildings),
            "observed_building_count": int(len(observed)),
            "full_anomalies": support["full_anomalies"],
            "full_normals": support["full_normals"],
            "meter_composition": _meter_counts(row_meter, labels),
            "row_rng_seed_material": {
                str(label): list(
                    _row_seed_material(building_seed, row_seed, budget, label)
                )
                for label in (1, 0)
            },
        }
        cells[str(budget)] = cell
        audit.append(
            {
                "building_seed": building_seed,
                "row_seed": row_seed,
                "K": budget,
                "context_rows": CONTEXT_ROWS,
                "anomalies": cell["anomalies"],
                "normals": cell["normals"],
                "anomaly_rate": cell["anomaly_rate"],
                "full_anomalies": cell["full_anomalies"],
                "full_normals": cell["full_normals"],
                "observed_building_count": cell["observed_building_count"],
                "raw_index_sha256": cell["raw_index_sha256"],
                "meter_composition_json": json.dumps(
                    cell["meter_composition"], sort_keys=True, separators=(",", ":")
                ),
            }
        )
    artifact_path = (
        context_root
        / f"building_seed{building_seed}"
        / f"row_seed{row_seed}"
        / "fixed_10k_contexts.npz"
    )
    atomic_savez(artifact_path, **arrays)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "m5_building_count_v4_fixed_context",
        "experiment_version": EXPERIMENT_VERSION,
        "status": "TRAINING_CONTEXT_PREPARED",
        "sampling_profile": SAMPLING_PROFILE,
        "building_seed": building_seed,
        "building_draw_seed": building_seed,
        "row_seed": row_seed,
        "row_draw_seed": row_seed,
        "rng_algorithm": RNG_ALGORITHM,
        "sampling_inputs": ["selected_building_membership", "anomaly"],
        "ignored_for_sampling": ["meter", "site_id", "building_row_quota"],
        "sampling_without_replacement": True,
        "per_building_minimum": None,
        "per_building_quota": None,
        "per_meter_balance": False,
        "per_site_balance": False,
        "retry_or_repair": False,
        "source_building_manifest": _display_path(ladder_path),
        "source_building_manifest_sha256": file_sha256(ladder_path),
        "context_artifact": _display_path(artifact_path),
        "context_artifact_sha256": file_sha256(artifact_path),
        "budgets": list(BUDGETS),
        "context_rows": CONTEXT_ROWS,
        "cells": cells,
    }
    manifest_path = audit_root / (
        f"fixed_context_building_seed{building_seed}_row_seed{row_seed}.json"
    )
    atomic_write_json(manifest_path, manifest)
    return manifest_path, audit


def validate_ladder(path: Path) -> int:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError(f"V4 ladder experiment drift: {path}")
    if manifest.get("selection_constraints") != [] or manifest.get("retry_or_repair"):
        raise ValueError(f"V4 ladder contains a forbidden constraint/repair: {path}")
    candidates = np.asarray(manifest["candidate_buildings"], dtype="int64")
    if int_array_sha256(np.sort(candidates)) != manifest["candidate_building_sha256"]:
        raise ValueError(f"V4 candidate digest drift: {path}")
    previous = np.empty(0, dtype="int64")
    checked = 0
    for budget in BUDGETS:
        cell = manifest["cells"][str(budget)]
        selected = np.asarray(cell["available_buildings"], dtype="int64")
        if len(selected) != budget or len(np.unique(selected)) != budget:
            raise ValueError(f"V4 ladder invalid at K={budget}: {path}")
        if len(previous) and not np.array_equal(selected[: len(previous)], previous):
            raise ValueError(f"V4 ladder not nested at K={budget}: {path}")
        if np.any(selected % 2) or not np.isin(selected, candidates).all():
            raise ValueError(f"V4 ladder escaped candidates at K={budget}: {path}")
        if int_array_sha256(selected) != cell["available_building_sha256"]:
            raise ValueError(f"V4 selected-building digest drift at K={budget}")
        previous = selected
        checked += 1
    return checked


def _resolve_recorded_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_context(path: Path) -> int:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError(f"V4 context experiment drift: {path}")
    if manifest.get("retry_or_repair") or manifest.get("per_building_minimum"):
        raise ValueError(f"V4 context contains forbidden repair/minimum: {path}")
    ladder_path = _resolve_recorded_path(manifest["source_building_manifest"])
    if file_sha256(ladder_path) != manifest["source_building_manifest_sha256"]:
        raise ValueError(f"V4 source ladder digest drift: {ladder_path}")
    validate_ladder(ladder_path)
    ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
    artifact_path = _resolve_recorded_path(manifest["context_artifact"])
    if file_sha256(artifact_path) != manifest["context_artifact_sha256"]:
        raise ValueError(f"V4 context artifact digest drift: {artifact_path}")
    checked = 0
    with np.load(artifact_path) as payload:
        if tuple(map(int, payload["budgets"])) != BUDGETS:
            raise ValueError(f"V4 artifact budget drift: {artifact_path}")
        for budget in BUDGETS:
            rows = np.asarray(payload[f"raw_index_k{budget}"], dtype="int64")
            labels = np.asarray(payload[f"anomaly_k{budget}"], dtype="int8")
            row_buildings = np.asarray(payload[f"building_id_k{budget}"], dtype="int64")
            selected = np.asarray(
                ladder["cells"][str(budget)]["available_buildings"], dtype="int64"
            )
            cell = manifest["cells"][str(budget)]
            if len(rows) != CONTEXT_ROWS or len(np.unique(rows)) != CONTEXT_ROWS:
                raise ValueError(f"V4 context size/uniqueness failed at K={budget}")
            if (
                int(labels.sum()) != ROWS_PER_CLASS
                or int((labels == 0).sum()) != ROWS_PER_CLASS
            ):
                raise ValueError(f"V4 context is not exactly 50:50 at K={budget}")
            if not np.isin(row_buildings, selected).all():
                raise ValueError(f"V4 context escaped building support at K={budget}")
            if int_array_sha256(rows) != cell["raw_index_sha256"]:
                raise ValueError(f"V4 row digest drift at K={budget}")
            checked += 1
    return checked


def prepare_all(
    *, raw_root: Path, audit_root: Path, context_root: Path
) -> dict[str, Any]:
    building, meter, anomaly, raw_inputs = load_raw_identifiers(raw_root)
    candidates = np.sort(np.unique(building[(building % 2) == 0])).astype("int64")
    ladder_audit: list[dict[str, Any]] = []
    context_audit: list[dict[str, Any]] = []
    context_paths: list[Path] = []
    for building_seed in BUILDING_DRAW_SEEDS:
        ladder_path, ladder_cells, audit = prepare_building_ladder(
            building_seed=building_seed,
            building=building,
            anomaly=anomaly,
            candidates=candidates,
            audit_root=audit_root,
            raw_inputs=raw_inputs,
        )
        ladder_audit.extend(audit)
        for row_seed in ROW_DRAW_SEEDS:
            context_path, audit = prepare_context_pair(
                building_seed=building_seed,
                row_seed=row_seed,
                building=building,
                meter=meter,
                anomaly=anomaly,
                ladder_path=ladder_path,
                ladder_cells=ladder_cells,
                audit_root=audit_root,
                context_root=context_root,
            )
            context_paths.append(context_path)
            context_audit.extend(audit)
            print(
                f"prepared building_seed={building_seed} row_seed={row_seed}: 5/5",
                flush=True,
            )
    atomic_write_csv(
        audit_root / "building_support_audit.csv", pd.DataFrame(ladder_audit)
    )
    atomic_write_csv(
        audit_root / "context_composition_audit.csv", pd.DataFrame(context_audit)
    )
    checked_ladders = sum(
        validate_ladder(audit_root / f"building_ladder_seed{seed}.json")
        for seed in BUILDING_DRAW_SEEDS
    )
    checked_contexts = sum(validate_context(path) for path in context_paths)
    expected_ladders = len(BUILDING_DRAW_SEEDS) * len(BUDGETS)
    expected_contexts = len(BUILDING_DRAW_SEEDS) * len(ROW_DRAW_SEEDS) * len(BUDGETS)
    passed = (
        checked_ladders == expected_ladders and checked_contexts == expected_contexts
    )
    gate = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "m5_building_count_v4_training_context_gate",
        "experiment_version": EXPERIMENT_VERSION,
        "status": "PASSED" if passed else "FAILED",
        "passed": passed,
        "building_seeds": list(BUILDING_DRAW_SEEDS),
        "row_seeds": list(ROW_DRAW_SEEDS),
        "budgets": list(BUDGETS),
        "context_rows": CONTEXT_ROWS,
        "rows_per_class": ROWS_PER_CLASS,
        "rng_algorithm": RNG_ALGORITHM,
        "candidate_buildings": int(len(candidates)),
        "candidate_building_sha256": int_array_sha256(candidates),
        "expected_ladder_cells": expected_ladders,
        "checked_ladder_cells": checked_ladders,
        "expected_context_cells": expected_contexts,
        "checked_context_cells": checked_contexts,
        "raw_inputs": raw_inputs,
    }
    atomic_write_json(audit_root / "training_context_gate.json", gate)
    atomic_write_json(
        audit_root / "summary.json",
        {
            **gate,
            "raw_root": str(raw_root.resolve()),
            "audit_root": _display_path(audit_root),
            "context_root": _display_path(context_root),
            "formal_model_launch_authorized": False,
        },
    )
    return gate


def check_all(*, audit_root: Path) -> dict[str, Any]:
    ladders = sum(
        validate_ladder(audit_root / f"building_ladder_seed{seed}.json")
        for seed in BUILDING_DRAW_SEEDS
    )
    contexts = sum(
        validate_context(
            audit_root
            / f"fixed_context_building_seed{building_seed}_row_seed{row_seed}.json"
        )
        for building_seed in BUILDING_DRAW_SEEDS
        for row_seed in ROW_DRAW_SEEDS
    )
    expected_ladders = len(BUILDING_DRAW_SEEDS) * len(BUDGETS)
    expected_contexts = len(BUILDING_DRAW_SEEDS) * len(ROW_DRAW_SEEDS) * len(BUDGETS)
    return {
        "expected_ladder_cells": expected_ladders,
        "checked_ladder_cells": ladders,
        "expected_context_cells": expected_contexts,
        "checked_context_cells": contexts,
        "passed": ladders == expected_ladders and contexts == expected_contexts,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "check"), default="prepare")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--context-root", type=Path, default=DEFAULT_CONTEXT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = (
        prepare_all(
            raw_root=args.raw_root,
            audit_root=args.audit_root,
            context_root=args.context_root,
        )
        if args.mode == "prepare"
        else check_all(audit_root=args.audit_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
