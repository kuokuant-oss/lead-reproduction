"""Deterministic building-budget protocol for the additive M5 curve.

The row-context curve answers how models scale with labelled rows.  This module
answers a different question: how they scale when labels come from more source
buildings.  It produces one deterministic building order; requested budgets are
strict prefixes, and every fifth position is reserved for tree early stopping.

Nothing in this module reads the odd-building canonical test labels.  Building
profiles and selection targets are computed from the even-building candidate
half only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
PROFILES = ("representative", "site_balanced", "meter_balanced", "anomaly_balanced")
METER_IDS = (0, 1, 2, 3)
ROLE_FIT = "fit"
ROLE_EARLY_STOP = "early_stop"


def int_array_sha256(values: np.ndarray | list[int]) -> str:
    array = np.asarray(values, dtype="int64").astype("<i8", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def stable_priority(values: np.ndarray, *, seed: int) -> np.ndarray:
    """Return deterministic uint64 priorities without depending on row order."""
    raw = np.asarray(values, dtype="uint64")
    state = raw ^ np.uint64(seed & ((1 << 64) - 1))
    state ^= state >> np.uint64(30)
    state *= np.uint64(0xBF58476D1CE4E5B9)
    state ^= state >> np.uint64(27)
    state *= np.uint64(0x94D049BB133111EB)
    state ^= state >> np.uint64(31)
    return state


def _positive_rate_bins(rates: pd.Series) -> pd.Series:
    """Four stable anomaly strata: zero plus three positive-rate quantiles."""
    result = pd.Series("zero", index=rates.index, dtype="object")
    positive = rates > 0
    if not positive.any():
        return result
    ranked = rates.loc[positive].rank(method="first", pct=True)
    result.loc[positive & (ranked <= 1 / 3)] = "positive_low"
    result.loc[positive & (ranked > 1 / 3) & (ranked <= 2 / 3)] = "positive_mid"
    result.loc[positive & (ranked > 2 / 3)] = "positive_high"
    return result


def _quantile_bins(values: pd.Series, labels: tuple[str, ...]) -> pd.Series:
    ranked = values.rank(method="first", pct=True)
    edges = np.linspace(0.0, 1.0, len(labels) + 1)
    output = pd.Series(index=values.index, dtype="object")
    for index, label in enumerate(labels):
        lower, upper = edges[index], edges[index + 1]
        mask = (ranked > lower) & (ranked <= upper)
        output.loc[mask] = label
    return output.fillna(labels[0])


def build_building_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one auditable profile row per even-building candidate."""
    required = {"building_id", "site_id", "meter", "anomaly"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"building profile frame missing {sorted(missing)}")
    if frame.empty:
        raise ValueError("building profile frame is empty")
    if frame["building_id"].mod(2).any():
        raise ValueError("building profiles must be built from the even-building half")

    grouped = frame.groupby("building_id", sort=True, observed=True)
    profile = grouped.agg(
        site_id=("site_id", "first"),
        rows=("anomaly", "size"),
        anomalies=("anomaly", "sum"),
        site_count=("site_id", "nunique"),
    ).reset_index()
    if (profile["site_count"] != 1).any():
        raise ValueError("a building maps to more than one site")
    profile = profile.drop(columns="site_count")
    profile["anomaly_rate"] = profile["anomalies"] / profile["rows"]

    for meter in METER_IDS:
        subset = frame[frame["meter"] == meter]
        counts = subset.groupby("building_id", observed=True)["anomaly"].agg(
            ["size", "sum"]
        )
        profile[f"meter_{meter}_rows"] = (
            profile["building_id"].map(counts["size"]).fillna(0).astype("int64")
        )
        profile[f"meter_{meter}_anomalies"] = (
            profile["building_id"].map(counts["sum"]).fillna(0).astype("int64")
        )
        profile[f"meter_{meter}_present"] = (profile[f"meter_{meter}_rows"] > 0).astype(
            "int8"
        )
        profile[f"meter_{meter}_row_share"] = (
            profile[f"meter_{meter}_rows"] / profile["rows"]
        )
        denominator = profile[f"meter_{meter}_rows"].replace(0, np.nan)
        profile[f"meter_{meter}_anomaly_rate"] = (
            profile[f"meter_{meter}_anomalies"] / denominator
        ).fillna(0.0)

    profile["meter_count"] = profile[
        [f"meter_{meter}_present" for meter in METER_IDS]
    ].sum(axis=1)
    profile["anomaly_meter_count"] = sum(
        (profile[f"meter_{meter}_anomalies"] > 0).astype("int8") for meter in METER_IDS
    )
    profile["zero_anomaly"] = (profile["anomalies"] == 0).astype("int8")
    profile["anomaly_bin"] = _positive_rate_bins(profile["anomaly_rate"])
    profile["size_bin"] = _quantile_bins(
        np.log1p(profile["rows"]), ("size_q1", "size_q2", "size_q3", "size_q4")
    )
    if "primary_use" in frame.columns:
        primary = grouped["primary_use"].first()
        profile["primary_use"] = profile["building_id"].map(primary).astype("string")
    return profile.sort_values("building_id").reset_index(drop=True)


def _design_matrix(
    profiles: pd.DataFrame, *, sampling_profile: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, float]]:
    if sampling_profile not in PROFILES:
        raise ValueError(f"unknown sampling profile {sampling_profile!r}")

    blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    names: list[str] = []

    def add_categorical(
        column: str, prefix: str, weight: float, balanced: bool
    ) -> None:
        levels = sorted(profiles[column].astype(str).unique())
        matrix = np.column_stack(
            [
                (profiles[column].astype(str) == level).to_numpy(float)
                for level in levels
            ]
        )
        target = (
            np.full(len(levels), 1 / len(levels)) if balanced else matrix.mean(axis=0)
        )
        blocks.append(matrix)
        targets.append(target)
        weights.append(np.full(len(levels), weight / len(levels)))
        names.extend(f"{prefix}:{level}" for level in levels)

    add_categorical(
        "site_id",
        "site",
        4.0 if sampling_profile == "site_balanced" else 2.0,
        sampling_profile == "site_balanced",
    )
    add_categorical(
        "anomaly_bin",
        "anomaly_bin",
        4.0 if sampling_profile == "anomaly_balanced" else 2.0,
        sampling_profile == "anomaly_balanced",
    )
    add_categorical("size_bin", "size_bin", 1.0, False)

    meter_presence = profiles[
        [f"meter_{meter}_present" for meter in METER_IDS]
    ].to_numpy(float)
    meter_share = profiles[
        [f"meter_{meter}_row_share" for meter in METER_IDS]
    ].to_numpy(float)
    meter_weight = 4.0 if sampling_profile == "meter_balanced" else 2.0
    blocks.extend((meter_presence, meter_share))
    targets.extend(
        (
            meter_presence.mean(axis=0),
            np.full(len(METER_IDS), 1 / len(METER_IDS))
            if sampling_profile == "meter_balanced"
            else meter_share.mean(axis=0),
        )
    )
    weights.extend(
        (
            np.full(len(METER_IDS), meter_weight / len(METER_IDS)),
            np.full(len(METER_IDS), meter_weight / len(METER_IDS)),
        )
    )
    names.extend(f"meter_presence:{meter}" for meter in METER_IDS)
    names.extend(f"meter_row_share:{meter}" for meter in METER_IDS)

    numeric = profiles[["anomaly_rate", "zero_anomaly"]].to_numpy(float)
    blocks.append(numeric)
    targets.append(numeric.mean(axis=0))
    weights.append(
        np.asarray(
            [
                2.0 if sampling_profile == "anomaly_balanced" else 1.0,
                2.0 if sampling_profile == "anomaly_balanced" else 1.0,
            ]
        )
    )
    names.extend(("mean_anomaly_rate", "zero_anomaly_share"))

    matrix = np.column_stack(blocks).astype("float64")
    target = np.concatenate(targets).astype("float64")
    weight = np.concatenate(weights).astype("float64")
    return matrix, target, weight, names, dict(zip(names, target, strict=True))


def build_nested_building_ladder(
    profiles: pd.DataFrame,
    budgets: list[int] | tuple[int, ...],
    *,
    seed: int = 42,
    sampling_profile: str = "representative",
    early_stop_every: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Greedily order buildings while balancing every requested prefix.

    The overall prefix objective is always active.  For an early-stop position,
    a second objective keeps the validation-role prefix representative as well.
    Role assignments never change as K grows.
    """
    ordered_budgets = sorted(set(int(value) for value in budgets))
    if not ordered_budgets or any(value <= 0 for value in ordered_budgets):
        raise ValueError("building budgets must be positive")
    if early_stop_every < 2:
        raise ValueError("early_stop_every must be at least 2")
    if any(value % early_stop_every for value in ordered_budgets):
        raise ValueError("building budgets must be multiples of early_stop_every")
    if ordered_budgets[-1] > len(profiles):
        raise ValueError("largest building budget exceeds candidate buildings")
    if profiles["building_id"].duplicated().any():
        raise ValueError("building profiles contain duplicate building IDs")

    matrix, target, weight, dimensions, target_map = _design_matrix(
        profiles, sampling_profile=sampling_profile
    )
    building_ids = profiles["building_id"].to_numpy(dtype="int64")
    priority = stable_priority(building_ids, seed=seed)
    available = np.ones(len(profiles), dtype=bool)
    total_sum = np.zeros(matrix.shape[1], dtype="float64")
    role_sums = {
        ROLE_FIT: np.zeros(matrix.shape[1], dtype="float64"),
        ROLE_EARLY_STOP: np.zeros(matrix.shape[1], dtype="float64"),
    }
    role_counts = {ROLE_FIT: 0, ROLE_EARLY_STOP: 0}
    rows: list[dict[str, Any]] = []

    for position in range(ordered_budgets[-1]):
        role = ROLE_EARLY_STOP if (position + 1) % early_stop_every == 0 else ROLE_FIT
        candidates = np.flatnonzero(available)
        overall_mean = (total_sum + matrix[candidates]) / (position + 1)
        overall_error = ((overall_mean - target) ** 2 * weight).sum(axis=1)
        role_mean = (role_sums[role] + matrix[candidates]) / (role_counts[role] + 1)
        role_error = ((role_mean - target) ** 2 * weight).sum(axis=1)
        score = overall_error + 0.35 * role_error
        order = np.lexsort((building_ids[candidates], priority[candidates], score))
        chosen = int(candidates[order[0]])
        available[chosen] = False
        total_sum += matrix[chosen]
        role_sums[role] += matrix[chosen]
        role_counts[role] += 1
        rows.append(
            {
                "position": position + 1,
                "building_id": int(building_ids[chosen]),
                "role": role,
                "site_id": int(profiles.iloc[chosen]["site_id"]),
                "stable_priority": int(priority[chosen]),
                "selection_score": float(score[order[0]]),
            }
        )

    ladder = pd.DataFrame(rows)
    cells: dict[str, Any] = {}
    for budget in ordered_budgets:
        prefix = ladder.iloc[:budget]
        available_ids = prefix["building_id"].to_numpy(dtype="int64")
        fit_ids = prefix.loc[prefix["role"] == ROLE_FIT, "building_id"].to_numpy(
            dtype="int64"
        )
        es_ids = prefix.loc[prefix["role"] == ROLE_EARLY_STOP, "building_id"].to_numpy(
            dtype="int64"
        )
        cells[str(budget)] = {
            "available_buildings": [int(value) for value in available_ids],
            "tree_fit_buildings": [int(value) for value in fit_ids],
            "tree_early_stop_buildings": [int(value) for value in es_ids],
            "available_building_sha256": int_array_sha256(available_ids),
            "tree_fit_building_sha256": int_array_sha256(fit_ids),
            "tree_early_stop_building_sha256": int_array_sha256(es_ids),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "m5_building_ladder",
        "sampling_profile": sampling_profile,
        "seed": int(seed),
        "early_stop_every": int(early_stop_every),
        "budgets": ordered_budgets,
        "candidate_buildings": int(len(profiles)),
        "candidate_building_sha256": int_array_sha256(building_ids),
        "selection_dimensions": dimensions,
        "selection_targets": target_map,
        "role_objective_weight": 0.35,
        "cells": cells,
    }
    validate_ladder(ladder, manifest)
    return ladder, manifest


def validate_ladder(ladder: pd.DataFrame, manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported building ladder schema")
    budgets = [int(value) for value in manifest["budgets"]]
    if len(ladder) != max(budgets):
        raise AssertionError("building ladder length differs from largest budget")
    if ladder["building_id"].duplicated().any():
        raise AssertionError("building ladder repeats a building")
    if ladder["building_id"].mod(2).any():
        raise AssertionError("building ladder contains an odd holdout building")
    previous: set[int] = set()
    for budget in budgets:
        cell = manifest["cells"][str(budget)]
        available = np.asarray(cell["available_buildings"], dtype="int64")
        fit = np.asarray(cell["tree_fit_buildings"], dtype="int64")
        early_stop = np.asarray(cell["tree_early_stop_buildings"], dtype="int64")
        expected = ladder.iloc[:budget]["building_id"].to_numpy(dtype="int64")
        if not np.array_equal(available, expected):
            raise AssertionError(f"K={budget} is not the ladder prefix")
        if previous and not previous < set(map(int, available)):
            raise AssertionError("building budgets are not strict nested supersets")
        previous = set(map(int, available))
        if set(map(int, fit)) & set(map(int, early_stop)):
            raise AssertionError("tree fit and early-stop buildings overlap")
        if set(map(int, fit)) | set(map(int, early_stop)) != previous:
            raise AssertionError("tree roles do not partition the available buildings")
        for name, values in (
            ("available_building", available),
            ("tree_fit_building", fit),
            ("tree_early_stop_building", early_stop),
        ):
            if int_array_sha256(values) != cell[f"{name}_sha256"]:
                raise AssertionError(f"K={budget} {name} digest mismatch")


def cell_indices(
    frame: pd.DataFrame, manifest: dict[str, Any], budget: int
) -> dict[str, np.ndarray]:
    """Resolve a ladder cell to row identities and enforce split/class gates."""
    cell = manifest["cells"].get(str(int(budget)))
    if cell is None:
        raise ValueError(f"building ladder has no K={budget} cell")
    building = frame["building_id"].to_numpy(dtype="int64")
    raw = frame.index.to_numpy(dtype="int64")
    output: dict[str, np.ndarray] = {}
    mapping = {
        "available": cell["available_buildings"],
        "tree_fit": cell["tree_fit_buildings"],
        "tree_early_stop": cell["tree_early_stop_buildings"],
    }
    for name, buildings in mapping.items():
        ids = np.asarray(buildings, dtype="int64")
        output[f"{name}_buildings"] = ids
        output[f"{name}_rows"] = raw[np.isin(building, ids)]
    if set(output["tree_fit_rows"]) & set(output["tree_early_stop_rows"]):
        raise AssertionError("tree fit and early-stop rows overlap")
    if set(output["available_rows"]) != (
        set(output["tree_fit_rows"]) | set(output["tree_early_stop_rows"])
    ):
        raise AssertionError("tree role rows do not partition available rows")
    for name in ("available", "tree_fit", "tree_early_stop"):
        labels = frame.loc[output[f"{name}_rows"], "anomaly"].to_numpy()
        if len(np.unique(labels)) != 2:
            raise ValueError(f"K={budget} {name} rows do not contain both classes")
    return output


def add_cell_composition(
    frame: pd.DataFrame, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Attach row/meter/site/class census without altering ladder identity."""
    result = json.loads(json.dumps(manifest))
    for budget in result["budgets"]:
        resolved = cell_indices(frame, result, int(budget))
        cell = result["cells"][str(budget)]
        for name in ("available", "tree_fit", "tree_early_stop"):
            rows = resolved[f"{name}_rows"]
            values = frame.loc[rows]
            cell[f"{name}_rows"] = int(len(rows))
            cell[f"{name}_row_sha256"] = int_array_sha256(rows)
            cell[f"{name}_anomalies"] = int(values["anomaly"].sum())
            cell[f"{name}_anomaly_rate"] = float(values["anomaly"].mean())
            cell[f"{name}_site_counts"] = {
                str(key): int(value)
                for key, value in values["site_id"].value_counts().sort_index().items()
            }
            cell[f"{name}_meter_counts"] = {
                str(key): int(value)
                for key, value in values["meter"].value_counts().sort_index().items()
            }
    return result


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
