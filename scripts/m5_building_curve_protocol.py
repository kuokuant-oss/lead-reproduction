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
DEFAULT_DIVERSIFICATION_TOP_N = 4
DEFAULT_DIVERSIFICATION_RELATIVE_TOLERANCE = 0.02
DEFAULT_DIVERSIFICATION_ABSOLUTE_TOLERANCE = 1e-12
DEFAULT_PREFIX_QUALITY_MAX_RATIO = 1.50
DEFAULT_PREFIX_QUALITY_MAX_ABSOLUTE_DEGRADATION = 0.003


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
    add_categorical(
        "anomaly_meter_count",
        "anomaly_meter_count",
        2.0 if sampling_profile == "anomaly_balanced" else 1.0,
        False,
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


def prefix_composition_discrepancy(
    profiles: pd.DataFrame,
    building_ids: np.ndarray | list[int],
    *,
    sampling_profile: str = "representative",
) -> float:
    """Measure one building prefix against the candidate-pool target."""
    matrix, target, weight, _, _ = _design_matrix(
        profiles, sampling_profile=sampling_profile
    )
    lookup = {
        int(building_id): index
        for index, building_id in enumerate(
            profiles["building_id"].to_numpy(dtype="int64")
        )
    }
    try:
        indices = np.asarray(
            [lookup[int(building_id)] for building_id in building_ids], dtype="int64"
        )
    except KeyError as error:
        raise ValueError(
            f"unknown building in discrepancy prefix: {error.args[0]}"
        ) from error
    if not len(indices):
        raise ValueError("discrepancy prefix must contain at least one building")
    prefix_mean = matrix[indices].mean(axis=0)
    return float(((prefix_mean - target) ** 2 * weight).sum())


def manifest_building_seed(manifest: dict[str, Any]) -> int | None:
    """Resolve explicit pilot provenance while accepting historical manifests."""
    for key in ("building_seed", "building_selection_seed", "seed"):
        if manifest.get(key) is not None:
            return int(manifest[key])
    return None


def build_nested_building_ladder(
    profiles: pd.DataFrame,
    budgets: list[int] | tuple[int, ...],
    *,
    seed: int = 42,
    sampling_profile: str = "representative",
    early_stop_every: int = 5,
    diversify_candidates: bool = False,
    acceptable_candidate_top_n: int = DEFAULT_DIVERSIFICATION_TOP_N,
    acceptable_candidate_relative_tolerance: float = (
        DEFAULT_DIVERSIFICATION_RELATIVE_TOLERANCE
    ),
    acceptable_candidate_absolute_tolerance: float = (
        DEFAULT_DIVERSIFICATION_ABSOLUTE_TOLERANCE
    ),
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
    if acceptable_candidate_top_n < 1:
        raise ValueError("acceptable candidate top-N must be positive")
    if acceptable_candidate_relative_tolerance < 0:
        raise ValueError("acceptable candidate relative tolerance cannot be negative")
    if acceptable_candidate_absolute_tolerance < 0:
        raise ValueError("acceptable candidate absolute tolerance cannot be negative")
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
        canonical_order = np.lexsort(
            (building_ids[candidates], priority[candidates], score)
        )
        score_order = np.lexsort((building_ids[candidates], score))
        best_score = float(score[canonical_order[0]])
        acceptable_limit = (
            best_score * (1.0 + acceptable_candidate_relative_tolerance)
            + acceptable_candidate_absolute_tolerance
        )
        acceptable = score_order[score[score_order] <= acceptable_limit]
        acceptable = acceptable[:acceptable_candidate_top_n]
        if diversify_candidates:
            diversified_order = np.lexsort(
                (
                    building_ids[candidates][acceptable],
                    priority[candidates][acceptable],
                )
            )
            chosen_local = int(acceptable[diversified_order[0]])
        else:
            chosen_local = int(canonical_order[0])
            acceptable = np.asarray([chosen_local], dtype="int64")
            acceptable_limit = best_score
        chosen = int(candidates[chosen_local])
        if position:
            error_before = float(((total_sum / position - target) ** 2 * weight).sum())
            gap = np.abs(total_sum / position - target) * weight
        else:
            error_before = float((target**2 * weight).sum())
            gap = np.abs(target) * weight
        primary_need = dimensions[int(np.argmax(gap))]
        error_after = float(overall_error[chosen_local])
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
                "seed_priority": int(priority[chosen]),
                "selection_score": float(score[chosen_local]),
                "best_selection_score": best_score,
                "selection_score_ratio_to_best": (
                    float(score[chosen_local] / best_score)
                    if best_score > 0
                    else 1.0
                ),
                "selection_rank": int(
                    np.flatnonzero(score_order == chosen_local)[0] + 1
                ),
                "acceptable_candidate_rank": int(
                    np.flatnonzero(acceptable == chosen_local)[0] + 1
                ),
                "acceptable_candidate_count": int(len(acceptable)),
                "acceptable_score_limit": float(acceptable_limit),
                "overall_error_before": error_before,
                "overall_error_after": error_after,
                "marginal_error_reduction": error_before - error_after,
                "primary_balance_need_addressed": primary_need,
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
        "building_seed": int(seed),
        "early_stop_every": int(early_stop_every),
        "role_policy": f"every_{early_stop_every}th_position_is_early_stop",
        "role_seed": None,
        "budgets": ordered_budgets,
        "candidate_buildings": int(len(profiles)),
        "candidate_building_sha256": int_array_sha256(building_ids),
        "selection_dimensions": dimensions,
        "selection_targets": target_map,
        "role_objective_weight": 0.35,
        "candidate_diversification": {
            "enabled": bool(diversify_candidates),
            "acceptable_candidate_top_n": int(acceptable_candidate_top_n),
            "relative_score_tolerance": float(
                acceptable_candidate_relative_tolerance
            ),
            "absolute_score_tolerance": float(
                acceptable_candidate_absolute_tolerance
            ),
            "acceptable_set": (
                "top-N candidates whose score is within "
                "best*(1+relative_tolerance)+absolute_tolerance"
            ),
            "selector": "minimum seed-controlled stable hash, then building_id",
            "uncontrolled_rng": False,
        },
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


def add_proportional_row_quotas(
    frame: pd.DataFrame, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Allocate fixed per-building quotas with each requested prefix mean <= limit."""
    result = json.loads(json.dumps(manifest))
    average_limit = int(result["average_rows_per_building_limit"])
    total_limit = int(result["max_context_rows"])
    counts = frame["building_id"].value_counts()
    quotas: dict[str, int] = {}
    previous_budget = 0
    allocated_total = 0
    for budget in result["budgets"]:
        ids = np.asarray(
            result["cells"][str(budget)]["available_buildings"][previous_budget:],
            dtype="int64",
        )
        capacity = min(
            (int(budget) - previous_budget) * average_limit,
            total_limit - allocated_total,
        )
        available = np.asarray([int(counts.loc[value]) for value in ids], dtype="int64")
        if capacity < len(ids):
            raise ValueError("row capacity cannot give every building at least one row")
        allocation = np.ones(len(ids), dtype="int64")
        remaining = capacity - len(ids)
        room = available - 1
        if remaining and room.sum():
            ideal = remaining * room / room.sum()
            addition = np.minimum(np.floor(ideal).astype("int64"), room)
            allocation += addition
        while int(allocation.sum()) < capacity:
            eligible = allocation < available
            if not eligible.any():
                break
            target = capacity * available / available.sum()
            deficit = np.where(eligible, target - allocation, -np.inf)
            chosen = int(np.lexsort((ids, -deficit))[0])
            allocation[chosen] += 1
        if int(allocation.sum()) > capacity:
            raise AssertionError("proportional allocation exceeds block capacity")
        for building_id, quota in zip(ids, allocation, strict=True):
            quotas[str(int(building_id))] = int(quota)
        allocated_total += int(allocation.sum())
        cell = result["cells"][str(budget)]
        cell["allocated_row_upper_bound"] = int(budget) * average_limit
        cell["allocated_rows"] = allocated_total
        cell["average_allocated_rows_per_building"] = allocated_total / int(budget)
        previous_budget = int(budget)
    result["building_row_quotas"] = quotas
    return result


def average_building_capped_indices(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    budget: int,
    *,
    average_rows_per_building: int = 500,
    max_total_rows: int = 50_000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Select each building's fixed proportional quota without using labels."""
    cell = manifest["cells"].get(str(int(budget)))
    if cell is None:
        raise ValueError(f"building ladder has no K={budget} cell")
    if average_rows_per_building <= 0 or max_total_rows <= 0:
        raise ValueError("row caps must be positive")
    if average_rows_per_building * int(budget) > max_total_rows:
        raise ValueError("average cap would exceed the total context limit")
    building_values = frame["building_id"].to_numpy(dtype="int64")
    raw_values = frame.index.to_numpy(dtype="int64")
    selected: list[np.ndarray] = []
    for building_id in cell["available_buildings"]:
        rows = raw_values[building_values == int(building_id)]
        quota = int(manifest["building_row_quotas"][str(int(building_id))])
        if quota > len(rows):
            raise AssertionError("building quota exceeds available rows")
        priority = stable_priority(rows, seed=seed)
        order = np.lexsort((rows, priority))
        selected.append(rows[order[:quota]])
    available_rows = np.concatenate(selected).astype("int64", copy=False)
    if len(available_rows) > max_total_rows:
        raise AssertionError("capped row selection exceeds the total context limit")
    if len(np.unique(available_rows)) != len(available_rows):
        raise AssertionError("capped row selection repeats rows")
    fit_buildings = np.asarray(cell["tree_fit_buildings"], dtype="int64")
    es_buildings = np.asarray(cell["tree_early_stop_buildings"], dtype="int64")
    row_buildings = frame.loc[available_rows, "building_id"].to_numpy(dtype="int64")
    output = {
        "available_buildings": np.asarray(cell["available_buildings"], dtype="int64"),
        "tree_fit_buildings": fit_buildings,
        "tree_early_stop_buildings": es_buildings,
        "available_rows": available_rows,
        "tree_fit_rows": available_rows[np.isin(row_buildings, fit_buildings)],
        "tree_early_stop_rows": available_rows[np.isin(row_buildings, es_buildings)],
    }
    if set(output["tree_fit_rows"]) & set(output["tree_early_stop_rows"]):
        raise AssertionError("tree fit and early-stop rows overlap")
    if set(output["available_rows"]) != (
        set(output["tree_fit_rows"]) | set(output["tree_early_stop_rows"])
    ):
        raise AssertionError("tree roles do not partition capped rows")
    for name in ("available", "tree_fit", "tree_early_stop"):
        labels = frame.loc[output[f"{name}_rows"], "anomaly"].to_numpy()
        if len(np.unique(labels)) != 2:
            raise ValueError(f"K={budget} {name} rows do not contain both classes")
    return output


def resolve_cell_indices(
    frame: pd.DataFrame, manifest: dict[str, Any], budget: int
) -> dict[str, np.ndarray]:
    """Resolve either the full-row baseline or fixed-context K protocol."""
    policy = manifest.get("row_policy", "all_rows")
    if policy == "all_rows":
        return cell_indices(frame, manifest, budget)
    if policy == "average_building_cap":
        return average_building_capped_indices(
            frame,
            manifest,
            budget,
            average_rows_per_building=int(manifest["average_rows_per_building_limit"]),
            max_total_rows=int(manifest["max_context_rows"]),
            seed=int(manifest["row_selection_seed"]),
        )
    raise ValueError(f"unsupported row policy {policy!r}")


def add_cell_composition(
    frame: pd.DataFrame, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Attach row/meter/site/class census without altering ladder identity."""
    result = json.loads(json.dumps(manifest))
    previous_rows: set[int] = set()
    for budget in result["budgets"]:
        resolved = resolve_cell_indices(frame, result, int(budget))
        current_rows = set(map(int, resolved["available_rows"]))
        if previous_rows and not previous_rows < current_rows:
            raise AssertionError("row budgets are not strict nested supersets")
        previous_rows = current_rows
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
        pair_census = (
            frame.loc[resolved["available_rows"]]
            .groupby(["building_id", "meter"], observed=True)["anomaly"]
            .agg(["size", "sum"])
        )
        cell["available_building_meter_pairs"] = int(len(pair_census))
        cell["available_anomalous_building_meter_pairs"] = int(
            (pair_census["sum"] > 0).sum()
        )
        cell["available_anomalous_building_meter_rate"] = float(
            (pair_census["sum"] > 0).mean()
        )
    return result


def add_building_audit(
    frame: pd.DataFrame,
    ladder: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Attach transparent source, meter, class and selection-reason records."""
    result = json.loads(json.dumps(manifest))
    for budget in result["budgets"]:
        resolved = resolve_cell_indices(frame, result, int(budget))
        selected_rows = set(map(int, resolved["available_rows"]))
        prefix = ladder.iloc[: int(budget)]
        records: list[dict[str, Any]] = []
        for row in prefix.itertuples(index=False):
            available = frame.loc[frame["building_id"].eq(int(row.building_id))]
            chosen = available.loc[available.index.isin(selected_rows)]
            record = {
                "position": int(row.position),
                "building_id": int(row.building_id),
                "site_id": int(row.site_id),
                "primary_use": (
                    str(available["primary_use"].iloc[0])
                    if "primary_use" in available.columns
                    else None
                ),
                "role": str(row.role),
                "selection_reason": str(row.primary_balance_need_addressed),
                "selection_score": float(row.selection_score),
                "marginal_error_reduction": float(row.marginal_error_reduction),
                "available_rows": int(len(available)),
                "available_anomalies": int(available["anomaly"].sum()),
                "available_anomaly_rate": float(available["anomaly"].mean()),
                "selected_rows": int(len(chosen)),
                "allocated_row_quota": int(
                    result.get("building_row_quotas", {}).get(
                        str(int(row.building_id)), len(chosen)
                    )
                ),
                "row_allocation_reason": (
                    "proportional_to_available_rows_within_incremental_K_block"
                    if result.get("row_policy") == "average_building_cap"
                    else "all_available_rows"
                ),
                "selected_anomalies": int(chosen["anomaly"].sum()),
                "selected_anomaly_rate": float(chosen["anomaly"].mean()),
                "selected_row_sha256": int_array_sha256(chosen.index.to_numpy()),
            }
            meter_details: list[dict[str, Any]] = []
            for meter in sorted(map(int, available["meter"].unique())):
                available_meter = available.loc[available["meter"].eq(meter)]
                selected_meter = chosen.loc[chosen["meter"].eq(meter)]
                meter_details.append(
                    {
                        "meter": meter,
                        "available_rows": int(len(available_meter)),
                        "available_anomalies": int(available_meter["anomaly"].sum()),
                        "available_anomaly_rate": float(
                            available_meter["anomaly"].mean()
                        ),
                        "selected_rows": int(len(selected_meter)),
                        "selected_anomalies": int(selected_meter["anomaly"].sum()),
                        "selected_anomaly_rate": (
                            float(selected_meter["anomaly"].mean())
                            if len(selected_meter)
                            else None
                        ),
                    }
                )
            record["meter_types"] = [item["meter"] for item in meter_details]
            record["meter_details"] = meter_details
            records.append(record)
        result["cells"][str(budget)]["selected_building_audit"] = records
    return result


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
