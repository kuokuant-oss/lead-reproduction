"""Prepare frozen label-only 50:50 contexts for M5 building-count V3.

This command does not train a model. It reuses the exact V2 building ladders,
then draws unique training rows using only the binary anomaly label and a
frozen seeded pseudo-random priority. Building, site, meter, V2 row identity,
and composition diagnostics never rank, repair, accept, or reject rows.
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
    from m5_building_curve_protocol import (
        atomic_write_json,
        int_array_sha256,
        stable_priority,
    )
except ModuleNotFoundError:  # Package import used by unittest discovery.
    from scripts.m5_building_curve_protocol import (
        atomic_write_json,
        int_array_sha256,
        stable_priority,
    )


SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "m5_building_count_v3_balanced_context"
SAMPLING_PROFILE = "global_label_50_50_seeded_random_without_replacement"
BUILDING_SEEDS = tuple(range(42, 52))
BUDGETS = (10, 20, 50, 100)
CONTEXT_ROWS = {10: 5_000, 20: 10_000, 50: 25_000, 100: 50_000}
BALANCE_SEED = 42
DEFAULT_RAW_ROOT = ROOT / "data" / "raw" / "m3"
DEFAULT_SEED42_46_ROOT = (
    ROOT / "experiments/m5_building_count_v3_balanced_context/source_building_manifests"
)
DEFAULT_SEED47_51_ROOT = ROOT / "experiments/m5_building_count_v2_seed47_51/audit"
DEFAULT_AUDIT_ROOT = ROOT / "experiments/m5_building_count_v3_balanced_context/audit"
DEFAULT_CONTEXT_ROOT = (
    ROOT / "data/processed/m5_building_curve/v3_balanced_context/training_contexts"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def source_manifest_path(
    seed: int, *, seed42_46_root: Path, seed47_51_root: Path
) -> Path:
    root = seed42_46_root if seed <= 46 else seed47_51_root
    return root / f"building_ladder_seed{seed}.json"


def validate_source_manifest(
    path: Path, *, expected_seed: int
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    if not path.is_file():
        raise ValueError(f"source building manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("building_seed", -1)) != expected_seed:
        raise ValueError(f"source manifest seed mismatch: {path}")
    if tuple(map(int, manifest.get("budgets", ()))) != BUDGETS:
        raise ValueError(f"source manifest budget mismatch: {path}")

    ladder_csv = path.with_suffix(".csv")
    expected_ladder_sha = manifest.get("ladder_csv_sha256")
    if expected_ladder_sha:
        if not ladder_csv.is_file():
            raise ValueError(f"source ladder CSV is missing: {ladder_csv}")
        observed_ladder_sha = file_sha256(ladder_csv)
        if observed_ladder_sha != expected_ladder_sha:
            raise ValueError(f"source ladder CSV digest mismatch: {ladder_csv}")

    buildings: dict[int, np.ndarray] = {}
    previous = np.empty(0, dtype="int64")
    for budget in BUDGETS:
        try:
            values = np.asarray(
                manifest["cells"][str(budget)]["available_buildings"],
                dtype="int64",
            )
        except KeyError as error:
            raise ValueError(f"source manifest lacks K={budget}: {path}") from error
        if len(values) != budget or len(np.unique(values)) != budget:
            raise ValueError(
                f"source manifest seed={expected_seed} K={budget} identity invalid"
            )
        if np.any(values % 2):
            raise ValueError("source building manifest includes odd holdout building")
        if len(previous) and not np.array_equal(values[: len(previous)], previous):
            raise ValueError(
                f"source building ladder is not an ordered prefix at K={budget}"
            )
        buildings[budget] = values
        previous = values
    return manifest, buildings


def load_raw_identifiers(raw_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_path = raw_root / "train.csv"
    label_path = raw_root / "bad_meter_readings.csv"
    if not train_path.is_file() or not label_path.is_file():
        raise ValueError(f"M3 raw identity files are missing under {raw_root}")

    train = pd.read_csv(
        train_path,
        usecols=["building_id", "meter"],
        dtype={"building_id": "int32", "meter": "int8"},
    )
    label = pd.read_csv(label_path)
    if tuple(label.columns) != ("is_bad_meter_reading",):
        raise ValueError(
            "bad_meter_readings.csv must contain only is_bad_meter_reading"
        )
    if len(train) != len(label):
        raise ValueError("M3 train and anomaly labels are not positionally aligned")
    anomaly = label["is_bad_meter_reading"].to_numpy(dtype="int8", copy=True)
    if not np.isin(anomaly, (0, 1)).all():
        raise ValueError("M3 anomaly labels must be binary")
    building = train["building_id"].to_numpy(dtype="int32", copy=True)
    meter = train["meter"].to_numpy(dtype="int8", copy=True)
    return building, meter, anomaly


def _validate_building_prefixes(buildings: dict[int, np.ndarray]) -> None:
    if tuple(buildings) != BUDGETS:
        raise ValueError("building sets must use K=10/20/50/100 in order")
    previous = np.empty(0, dtype="int64")
    for budget in BUDGETS:
        values = np.asarray(buildings[budget], dtype="int64")
        if len(values) != budget or len(np.unique(values)) != budget:
            raise ValueError(f"K={budget} must contain exactly K unique buildings")
        if len(previous) and not np.array_equal(values[: len(previous)], previous):
            raise ValueError(f"building sets are not strict prefixes at K={budget}")
        previous = values


def build_nested_balanced_contexts(
    building: np.ndarray,
    anomaly: np.ndarray,
    buildings: dict[int, np.ndarray],
    *,
    balance_seed: int = BALANCE_SEED,
    context_rows: dict[int, int] = CONTEXT_ROWS,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, int]]]:
    """Draw label-only balanced row prefixes from frozen building support."""
    building_values = np.asarray(building)
    labels = np.asarray(anomaly)
    if len(building_values) != len(labels) or not len(labels):
        raise ValueError("building and anomaly arrays must be non-empty and aligned")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("anomaly labels must be binary")
    _validate_building_prefixes(buildings)
    if tuple(context_rows) != BUDGETS:
        raise ValueError("context row targets must use K=10/20/50/100 in order")

    row_index = np.arange(len(labels), dtype="int64")
    selected_mask = np.zeros(len(labels), dtype=bool)
    selected = np.empty(0, dtype="int64")
    contexts: dict[int, np.ndarray] = {}
    support: dict[int, dict[str, int]] = {}
    previous_per_class = 0

    for budget in BUDGETS:
        target = int(context_rows[budget])
        if target <= 0 or target % 2:
            raise ValueError(f"K={budget} context target must be positive and even")
        target_per_class = target // 2
        if target_per_class <= previous_per_class:
            raise ValueError("context class targets must increase strictly with K")
        member = np.isin(building_values, buildings[budget])
        class_support: dict[int, int] = {}
        additions: dict[int, np.ndarray] = {}
        needed = target_per_class - previous_per_class
        for class_label in (1, 0):
            candidates = row_index[member & (labels == class_label)]
            class_support[class_label] = int(len(candidates))
            eligible = candidates[~selected_mask[candidates]]
            if len(eligible) < needed:
                raise ValueError(
                    "insufficient unique class support for balanced sampling "
                    f"without replacement: K={budget}, class={class_label}, "
                    f"need_additional={needed}, eligible={len(eligible)}"
                )
            priority = stable_priority(eligible, seed=balance_seed)
            order = np.lexsort((eligible, priority))
            additions[class_label] = eligible[order[:needed]]

        added = np.empty(needed * 2, dtype="int64")
        added[0::2] = additions[1]
        added[1::2] = additions[0]
        selected_mask[added] = True
        selected = np.concatenate((selected, added))

        selected_labels = labels[selected]
        if len(selected) != target:
            raise AssertionError(f"K={budget} context size mismatch")
        if int((selected_labels == 1).sum()) != target_per_class:
            raise AssertionError(f"K={budget} anomaly count mismatch")
        if int((selected_labels == 0).sum()) != target_per_class:
            raise AssertionError(f"K={budget} normal count mismatch")
        if len(np.unique(selected)) != len(selected):
            raise AssertionError(f"K={budget} context repeats rows")
        if not np.isin(building_values[selected], buildings[budget]).all():
            raise AssertionError(f"K={budget} context escapes building support")
        contexts[budget] = selected.copy()
        support[budget] = {
            "full_anomalies": class_support[1],
            "full_normals": class_support[0],
            "selected_anomalies": target_per_class,
            "selected_normals": target_per_class,
        }
        previous_per_class = target_per_class

    return contexts, support


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


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def prepare_seed(
    *,
    seed: int,
    building: np.ndarray,
    meter: np.ndarray,
    anomaly: np.ndarray,
    source_path: Path,
    audit_root: Path,
    context_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _, building_sets = validate_source_manifest(source_path, expected_seed=seed)
    contexts, support = build_nested_balanced_contexts(
        building, anomaly, building_sets, balance_seed=BALANCE_SEED
    )
    maximum_rows = contexts[BUDGETS[-1]]
    artifact_path = context_root / f"building_seed{seed}" / "balanced_context_rows.npz"
    atomic_savez(
        artifact_path,
        raw_index=maximum_rows.astype("int64", copy=False),
        anomaly=anomaly[maximum_rows].astype("int8", copy=False),
        building_id=building[maximum_rows].astype("int32", copy=False),
        meter=meter[maximum_rows].astype("int8", copy=False),
        budgets=np.asarray(BUDGETS, dtype="int16"),
        context_rows=np.asarray([CONTEXT_ROWS[k] for k in BUDGETS], dtype="int32"),
    )

    cells: dict[str, Any] = {}
    context_rows_audit: list[dict[str, Any]] = []
    building_audit: list[dict[str, Any]] = []
    for budget in BUDGETS:
        rows = contexts[budget]
        row_labels = anomaly[rows]
        row_buildings = building[rows]
        row_meter = meter[rows]
        selected_buildings = building_sets[budget]
        meter_counts = _meter_counts(row_meter, row_labels)
        observed_buildings = np.unique(row_buildings)
        cell = {
            "K": budget,
            "context_rows": int(len(rows)),
            "anomalies": int(row_labels.sum()),
            "normals": int(len(rows) - row_labels.sum()),
            "anomaly_rate": float(row_labels.mean()),
            "raw_index_sha256": int_array_sha256(rows),
            "selected_buildings": selected_buildings.tolist(),
            "selected_building_sha256": int_array_sha256(selected_buildings),
            "observed_building_count": int(len(observed_buildings)),
            "full_anomalies": support[budget]["full_anomalies"],
            "full_normals": support[budget]["full_normals"],
            "meter_composition": meter_counts,
        }
        cells[str(budget)] = cell
        context_rows_audit.append(
            {
                "building_seed": seed,
                "K": budget,
                "context_rows": cell["context_rows"],
                "anomalies": cell["anomalies"],
                "normals": cell["normals"],
                "anomaly_rate": cell["anomaly_rate"],
                "full_anomalies": cell["full_anomalies"],
                "full_normals": cell["full_normals"],
                "observed_building_count": cell["observed_building_count"],
                "raw_index_sha256": cell["raw_index_sha256"],
                "selected_building_sha256": cell["selected_building_sha256"],
                "meter_composition_json": json.dumps(
                    meter_counts, sort_keys=True, separators=(",", ":")
                ),
            }
        )
        for building_id in selected_buildings:
            mask = row_buildings == int(building_id)
            positives = int(row_labels[mask].sum())
            selected_count = int(mask.sum())
            building_audit.append(
                {
                    "building_seed": seed,
                    "K": budget,
                    "building_id": int(building_id),
                    "selected_rows": selected_count,
                    "selected_anomalies": positives,
                    "selected_normals": selected_count - positives,
                    "selected_anomaly_rate": (
                        positives / selected_count if selected_count else None
                    ),
                }
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "m5_building_count_v3_balanced_context",
        "experiment_version": EXPERIMENT_VERSION,
        "status": "TRAINING_CONTEXT_PREPARED",
        "building_seed": seed,
        "balance_seed": BALANCE_SEED,
        "sampling_profile": SAMPLING_PROFILE,
        "sampling_inputs": ["raw_index", "anomaly"],
        "eligibility_inputs": ["frozen_selected_building_membership"],
        "ignored_for_priority": [
            "V2_row_membership",
            "building_id",
            "site_id",
            "meter",
            "anomaly_severity",
            "composition_diagnostics",
        ],
        "sampling_without_replacement": True,
        "per_building_minimum": None,
        "per_building_quota": None,
        "per_meter_balance": False,
        "per_site_balance": False,
        "source_building_manifest": _display_path(source_path),
        "source_building_manifest_sha256": file_sha256(source_path),
        "context_artifact": _display_path(artifact_path),
        "context_artifact_sha256": file_sha256(artifact_path),
        "budgets": list(BUDGETS),
        "context_rows": [CONTEXT_ROWS[k] for k in BUDGETS],
        "cells": cells,
    }
    manifest_path = audit_root / f"balanced_context_seed{seed}.json"
    atomic_write_json(manifest_path, manifest)
    return manifest, context_rows_audit, building_audit


def validate_seed_artifact(
    manifest_path: Path,
) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError(f"V3 experiment version mismatch: {manifest_path}")
    source_path = Path(manifest["source_building_manifest"])
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    if file_sha256(source_path) != manifest["source_building_manifest_sha256"]:
        raise ValueError(f"source manifest drift: {source_path}")
    _, building_sets = validate_source_manifest(
        source_path, expected_seed=int(manifest["building_seed"])
    )

    artifact_path = Path(manifest["context_artifact"])
    if not artifact_path.is_absolute():
        artifact_path = ROOT / artifact_path
    if file_sha256(artifact_path) != manifest["context_artifact_sha256"]:
        raise ValueError(f"balanced context artifact drift: {artifact_path}")
    with np.load(artifact_path) as payload:
        raw_index = np.asarray(payload["raw_index"], dtype="int64")
        labels = np.asarray(payload["anomaly"], dtype="int8")
        building = np.asarray(payload["building_id"], dtype="int64")
        meter = np.asarray(payload["meter"], dtype="int8")
        budgets = tuple(map(int, payload["budgets"]))
        lengths = tuple(map(int, payload["context_rows"]))
    if not (len(raw_index) == len(labels) == len(building) == len(meter)):
        raise ValueError("balanced context arrays differ in length")
    if budgets != BUDGETS or lengths != tuple(CONTEXT_ROWS[k] for k in BUDGETS):
        raise ValueError("balanced context prefix metadata mismatch")

    checked = 0
    for budget, length in zip(budgets, lengths, strict=True):
        rows = raw_index[:length]
        y = labels[:length]
        row_buildings = building[:length]
        cell = manifest["cells"][str(budget)]
        if len(np.unique(rows)) != length:
            raise ValueError(
                f"seed={manifest['building_seed']} K={budget} repeats rows"
            )
        if int(y.sum()) != length // 2 or int((y == 0).sum()) != length // 2:
            raise ValueError(f"seed={manifest['building_seed']} K={budget} not 50:50")
        if not np.isin(row_buildings, building_sets[budget]).all():
            raise ValueError(
                f"seed={manifest['building_seed']} K={budget} building escape"
            )
        if int_array_sha256(rows) != cell["raw_index_sha256"]:
            raise ValueError(
                f"seed={manifest['building_seed']} K={budget} row digest mismatch"
            )
        checked += 1
    return checked


def prepare_all(
    *,
    seeds: tuple[int, ...],
    raw_root: Path,
    seed42_46_root: Path,
    seed47_51_root: Path,
    audit_root: Path,
    context_root: Path,
) -> dict[str, Any]:
    unexpected = sorted(set(seeds) - set(BUILDING_SEEDS))
    if unexpected:
        raise ValueError(f"unsupported building seeds: {unexpected}")
    building, meter, anomaly = load_raw_identifiers(raw_root)
    context_audit: list[dict[str, Any]] = []
    building_audit: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    for seed in seeds:
        source_path = source_manifest_path(
            seed,
            seed42_46_root=seed42_46_root,
            seed47_51_root=seed47_51_root,
        )
        registry.append(
            {
                "building_seed": seed,
                "source_manifest": _display_path(source_path),
                "source_manifest_sha256": file_sha256(source_path),
            }
        )
        _, context_rows, building_rows = prepare_seed(
            seed=seed,
            building=building,
            meter=meter,
            anomaly=anomaly,
            source_path=source_path,
            audit_root=audit_root,
            context_root=context_root,
        )
        context_audit.extend(context_rows)
        building_audit.extend(building_rows)
        print(f"prepared seed={seed}: 4/4 contexts", flush=True)

    atomic_write_csv(
        audit_root / "source_manifest_registry.csv", pd.DataFrame(registry)
    )
    atomic_write_csv(
        audit_root / "context_composition_audit.csv", pd.DataFrame(context_audit)
    )
    atomic_write_csv(
        audit_root / "per_building_contribution_audit.csv", pd.DataFrame(building_audit)
    )

    checked = 0
    for seed in seeds:
        checked += validate_seed_artifact(
            audit_root / f"balanced_context_seed{seed}.json"
        )
    expected = len(seeds) * len(BUDGETS)
    gate = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "m5_building_count_v3_training_context_gate",
        "experiment_version": EXPERIMENT_VERSION,
        "status": "PASSED" if checked == expected else "FAILED",
        "expected_cells": expected,
        "checked_cells": checked,
        "passed": checked == expected,
        "building_seeds": list(seeds),
        "budgets": list(BUDGETS),
        "balance_seed": BALANCE_SEED,
        "sampling_profile": SAMPLING_PROFILE,
    }
    atomic_write_json(audit_root / "training_context_gate.json", gate)
    summary = {
        **gate,
        "status": "TRAINING_CONTEXTS_PREPARED" if gate["passed"] else "FAILED",
        "raw_root": str(raw_root.resolve()),
        "audit_root": _display_path(audit_root),
        "context_root": _display_path(context_root),
        "formal_model_launch_authorized": False,
    }
    atomic_write_json(audit_root / "summary.json", summary)
    return summary


def check_all(*, seeds: tuple[int, ...], audit_root: Path) -> dict[str, Any]:
    checked = sum(
        validate_seed_artifact(audit_root / f"balanced_context_seed{seed}.json")
        for seed in seeds
    )
    expected = len(seeds) * len(BUDGETS)
    return {
        "expected_cells": expected,
        "checked_cells": checked,
        "passed": checked == expected,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "check"), default="prepare")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(BUILDING_SEEDS))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--seed42-46-root", type=Path, default=DEFAULT_SEED42_46_ROOT)
    parser.add_argument("--seed47-51-root", type=Path, default=DEFAULT_SEED47_51_ROOT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--context-root", type=Path, default=DEFAULT_CONTEXT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = tuple(map(int, args.seeds))
    if len(set(seeds)) != len(seeds):
        raise ValueError("building seeds must be unique")
    if args.mode == "prepare":
        result = prepare_all(
            seeds=seeds,
            raw_root=args.raw_root,
            seed42_46_root=args.seed42_46_root,
            seed47_51_root=args.seed47_51_root,
            audit_root=args.audit_root,
            context_root=args.context_root,
        )
    else:
        result = check_all(seeds=seeds, audit_root=args.audit_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
