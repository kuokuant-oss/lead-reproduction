"""Seeded site-stratified source-building sampling for the additive M5 curve.

The building-count curve measures how models scale when labels come from more
source buildings. Source identity is sampled without replacement from the
even-building training half. Within-site permutations are genuinely random
under an explicit PCG64 seed, while a proportional site schedule interleaves
the strata. Requested budgets are strict prefixes of one accepted ladder.

Meter presence is used only as a predeclared feasibility gate on the completed
random ladder. Labels, anomaly statistics, row counts, meter row shares and
tree roles never rank, weight, repair or otherwise choose source identities.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = 2
SAMPLING_PROFILE = "site_stratified_random"
PROFILES = (SAMPLING_PROFILE,)
METER_IDS = (0, 1, 2, 3)
ROLE_FIT = "fit"
ROLE_EARLY_STOP = "early_stop"
DEFAULT_METER_MIN_SOURCE_BUILDINGS = 2
DEFAULT_METER_GROWTH_PER_TRANSITION = 1
DEFAULT_MAX_LADDER_ATTEMPTS = 10_000
RNG_ALGORITHM = "numpy.random.PCG64"


class LadderInfeasibilityError(ValueError):
    """No random ladder satisfied the declared meter constraints."""


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


def _sampling_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Return the only columns allowed to influence building identity."""
    required = {
        "building_id",
        "site_id",
        *(f"meter_{meter}_present" for meter in METER_IDS),
    }
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"sampling profile missing {sorted(missing)}")
    sampling = profiles.loc[:, sorted(required)].copy()
    if sampling.empty:
        raise ValueError("sampling profile is empty")
    if sampling["building_id"].duplicated().any():
        raise ValueError("sampling profile contains duplicate building IDs")
    if sampling["building_id"].mod(2).any():
        raise ValueError("sampling profile contains odd holdout buildings")
    if sampling["site_id"].isna().any():
        raise ValueError("sampling profile contains missing site IDs")
    for meter in METER_IDS:
        column = f"meter_{meter}_present"
        if not sampling[column].isin((0, 1, False, True)).all():
            raise ValueError(f"{column} must contain only binary presence values")
        sampling[column] = sampling[column].astype("int8")
    return sampling.sort_values("building_id").reset_index(drop=True)


def _rng_for_attempt(seed: int, attempt: int) -> np.random.Generator:
    """Create one reproducible PCG64 redraw stream."""
    unsigned_seed = int(seed) & ((1 << 64) - 1)
    seed_sequence = np.random.SeedSequence(
        [unsigned_seed & 0xFFFFFFFF, unsigned_seed >> 32, int(attempt)]
    )
    return np.random.Generator(np.random.PCG64(seed_sequence))


def _site_stratified_random_draw(
    sampling: pd.DataFrame,
    *,
    length: int,
    seed: int,
    attempt: int,
) -> pd.DataFrame:
    """Randomly permute each site and interleave sites near pool proportions."""
    rng = _rng_for_attempt(seed, attempt)
    sites = np.asarray(sorted(map(int, sampling["site_id"].unique())), dtype="int64")
    site_buildings: dict[int, np.ndarray] = {}
    populations = np.zeros(len(sites), dtype="int64")
    for index, site in enumerate(sites):
        building_ids = (
            sampling.loc[sampling["site_id"].eq(site), "building_id"]
            .sort_values()
            .to_numpy(dtype="int64")
        )
        site_buildings[int(site)] = rng.permutation(building_ids)
        populations[index] = len(building_ids)

    # This priority affects only exact site-schedule ties. Building identities
    # remain random permutations within each site.
    site_tie_priority = rng.permutation(len(sites))
    selected_by_site = np.zeros(len(sites), dtype="int64")
    next_index = np.zeros(len(sites), dtype="int64")
    rows: list[dict[str, Any]] = []
    total_candidates = int(populations.sum())

    for position in range(1, length + 1):
        eligible = selected_by_site < populations
        proportional_deficit = (
            position * populations.astype("float64") / total_candidates
            - selected_by_site
        )
        proportional_deficit[~eligible] = -np.inf
        best_deficit = float(proportional_deficit.max())
        tied = np.flatnonzero(
            np.isclose(proportional_deficit, best_deficit, rtol=0.0, atol=1e-12)
        )
        chosen_site_index = int(
            min(
                tied,
                key=lambda index: (
                    int(site_tie_priority[index]),
                    int(sites[index]),
                ),
            )
        )
        site = int(sites[chosen_site_index])
        draw_rank = int(next_index[chosen_site_index])
        building_id = int(site_buildings[site][draw_rank])
        rows.append(
            {
                "position": position,
                "building_id": building_id,
                "site_id": site,
                "sampling_attempt": int(attempt),
                "site_draw_rank": draw_rank + 1,
                "site_candidate_count": int(populations[chosen_site_index]),
                "site_tie_priority": int(site_tie_priority[chosen_site_index]),
            }
        )
        selected_by_site[chosen_site_index] += 1
        next_index[chosen_site_index] += 1
    return pd.DataFrame(rows)


def _meter_constraint_audit(
    sampling: pd.DataFrame,
    ladder: pd.DataFrame,
    budgets: list[int],
    *,
    meter_min_source_buildings: int,
    meter_growth_per_transition: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, int]]]:
    indexed = sampling.set_index("building_id")
    cell_audits: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, int]] = []
    previous_counts: dict[int, int] | None = None

    for budget_index, budget in enumerate(budgets):
        prefix_ids = ladder.iloc[:budget]["building_id"].to_numpy(dtype="int64")
        prefix = indexed.loc[prefix_ids]
        counts = {
            meter: int(prefix[f"meter_{meter}_present"].sum()) for meter in METER_IDS
        }
        constraints: dict[str, dict[str, Any]] = {}
        for meter in METER_IDS:
            if previous_counts is None:
                required = int(meter_min_source_buildings)
                rule = f"count_{meter}({budget}) >= {required}"
            else:
                required = int(previous_counts[meter] + meter_growth_per_transition)
                rule = (
                    f"count_{meter}({budget}) >= count_{meter}"
                    f"({budgets[budget_index - 1]})"
                    f" + {meter_growth_per_transition}"
                )
            passed = counts[meter] >= required
            constraints[str(meter)] = {
                "source_building_count": counts[meter],
                "required_minimum": required,
                "rule": rule,
                "passed": bool(passed),
            }
            if not passed:
                failures.append(
                    {
                        "meter": int(meter),
                        "K": int(budget),
                        "observed": counts[meter],
                        "required": required,
                    }
                )
        cell_audits[str(budget)] = {
            "meter_source_building_counts": {
                str(meter): count for meter, count in counts.items()
            },
            "meter_constraints": constraints,
            "meter_constraint_pass": all(
                item["passed"] for item in constraints.values()
            ),
        }
        previous_counts = counts
    return cell_audits, failures


def _preflight_meter_capacity(
    sampling: pd.DataFrame,
    budgets: list[int],
    *,
    meter_min_source_buildings: int,
    meter_growth_per_transition: int,
) -> None:
    final_required = meter_min_source_buildings + meter_growth_per_transition * (
        len(budgets) - 1
    )
    failures: list[str] = []
    for meter in METER_IDS:
        available = int(sampling[f"meter_{meter}_present"].sum())
        if available < final_required:
            failures.append(
                f"meter={meter} K={budgets[-1]} available={available} "
                f"required_at_least={final_required}"
            )
    if failures:
        raise LadderInfeasibilityError(
            "meter coverage is globally infeasible: " + "; ".join(failures)
        )


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
    early_stop_every: int = 5,
    meter_min_source_buildings: int = DEFAULT_METER_MIN_SOURCE_BUILDINGS,
    meter_growth_per_transition: int = DEFAULT_METER_GROWTH_PER_TRANSITION,
    max_sampling_attempts: int = DEFAULT_MAX_LADDER_ATTEMPTS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Draw one feasible site-stratified random ladder.

    Each attempt independently permutes buildings within every site with PCG64,
    then interleaves the site streams by candidate-pool proportion. Meter
    presence is checked only after the complete ladder is drawn. Failed ladders
    are discarded whole; no building is greedily corrected or swapped.
    """
    ordered_budgets = sorted(set(int(value) for value in budgets))
    if not ordered_budgets or any(value <= 0 for value in ordered_budgets):
        raise ValueError("building budgets must be positive")
    if early_stop_every < 2:
        raise ValueError("early_stop_every must be at least 2")
    if any(value % early_stop_every for value in ordered_budgets):
        raise ValueError("building budgets must be multiples of early_stop_every")
    if meter_min_source_buildings < 1:
        raise ValueError("meter minimum source-building coverage must be positive")
    if meter_growth_per_transition < 1:
        raise ValueError("meter growth per transition must be positive")
    if max_sampling_attempts < 1:
        raise ValueError("maximum sampling attempts must be positive")

    sampling = _sampling_profiles(profiles)
    if ordered_budgets[-1] > len(sampling):
        raise ValueError("largest building budget exceeds candidate buildings")
    _preflight_meter_capacity(
        sampling,
        ordered_budgets,
        meter_min_source_buildings=meter_min_source_buildings,
        meter_growth_per_transition=meter_growth_per_transition,
    )

    accepted_ladder: pd.DataFrame | None = None
    accepted_audits: dict[str, dict[str, Any]] | None = None
    best_observed: dict[tuple[int, int], int] = {}
    last_failures: list[dict[str, int]] = []
    accepted_attempt = -1
    for attempt in range(max_sampling_attempts):
        candidate_ladder = _site_stratified_random_draw(
            sampling,
            length=ordered_budgets[-1],
            seed=seed,
            attempt=attempt,
        )
        cell_audits, failures = _meter_constraint_audit(
            sampling,
            candidate_ladder,
            ordered_budgets,
            meter_min_source_buildings=meter_min_source_buildings,
            meter_growth_per_transition=meter_growth_per_transition,
        )
        if not failures:
            accepted_ladder = candidate_ladder
            accepted_audits = cell_audits
            accepted_attempt = attempt
            break
        last_failures = failures
        for failure in failures:
            key = (failure["K"], failure["meter"])
            best_observed[key] = max(best_observed.get(key, -1), failure["observed"])

    if accepted_ladder is None or accepted_audits is None:
        details = []
        for failure in last_failures:
            key = (failure["K"], failure["meter"])
            details.append(
                f"meter={failure['meter']} K={failure['K']} "
                f"best_observed={best_observed.get(key, failure['observed'])} "
                f"last_required={failure['required']}"
            )
        raise LadderInfeasibilityError(
            f"no feasible ladder for building_seed={seed} after "
            f"{max_sampling_attempts} attempts: " + "; ".join(details)
        )

    ladder = accepted_ladder
    ladder["role"] = np.where(
        ladder["position"].mod(early_stop_every).eq(0),
        ROLE_EARLY_STOP,
        ROLE_FIT,
    )

    building_ids = sampling["building_id"].to_numpy(dtype="int64")
    pool_site_counts = {
        str(int(site)): int(count)
        for site, count in sampling["site_id"].value_counts().sort_index().items()
    }
    pool_site_shares = {
        site: count / len(sampling) for site, count in pool_site_counts.items()
    }
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
        site_counts = {
            str(int(site)): int(count)
            for site, count in prefix["site_id"].value_counts().sort_index().items()
        }
        all_sites = set(pool_site_counts)
        max_site_count_deviation = max(
            abs(site_counts.get(site, 0) - budget * pool_site_shares[site])
            for site in all_sites
        )
        cell = {
            "available_buildings": [int(value) for value in available_ids],
            "tree_fit_buildings": [int(value) for value in fit_ids],
            "tree_early_stop_buildings": [int(value) for value in es_ids],
            "available_building_sha256": int_array_sha256(available_ids),
            "tree_fit_building_sha256": int_array_sha256(fit_ids),
            "tree_early_stop_building_sha256": int_array_sha256(es_ids),
            "reproducibility_digest": int_array_sha256(available_ids),
            "site_counts": site_counts,
            "site_max_absolute_count_deviation": float(max_site_count_deviation),
            "site_stratified_sampling_applied": True,
            **accepted_audits[str(budget)],
        }
        cell["constraint_pass"] = bool(cell["meter_constraint_pass"])
        cells[str(budget)] = cell

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "m5_building_ladder",
        "sampling_profile": SAMPLING_PROFILE,
        "sampling_method": "seeded_site_stratified_random_without_replacement",
        "seed": int(seed),
        "building_seed": int(seed),
        "sampling_attempt": int(accepted_attempt),
        "attempts_used": int(accepted_attempt + 1),
        "max_sampling_attempts": int(max_sampling_attempts),
        "rng": {
            "algorithm": RNG_ALGORITHM,
            "seeded": True,
            "building_seed": int(seed),
            "redraw_stream": "SeedSequence([seed_low32, seed_high32, attempt])",
            "within_site_operation": "random permutation without replacement",
        },
        "site_stratification": {
            "pool_site_counts": pool_site_counts,
            "pool_site_shares": pool_site_shares,
            "interleaving": (
                "largest proportional deficit at each position; seeded random "
                "priority resolves exact site ties"
            ),
        },
        "meter_feasibility": {
            "evaluation_meters": list(METER_IDS),
            "minimum_at_smallest_budget": int(meter_min_source_buildings),
            "minimum_growth_per_budget_transition": int(meter_growth_per_transition),
            "gate": "whole-ladder deterministic rejection sampling",
            "single_building_swap_or_greedy_correction": False,
        },
        "identity_selection_inputs": ["building_id", "site_id"],
        "feasibility_only_inputs": [f"meter_{meter}_present" for meter in METER_IDS],
        "excluded_from_identity_selection": [
            "anomaly",
            "anomaly_rate",
            "anomaly_bin",
            "zero_anomaly",
            "anomaly_meter_count",
            "rows",
            "size_bin",
            "meter_row_share",
            "candidate_pool_discrepancy",
            "role_prefix_discrepancy",
            "tree_role",
        ],
        "early_stop_every": int(early_stop_every),
        "role_policy": (
            f"roles assigned after ladder acceptance; every_{early_stop_every}"
            "th_position_is_early_stop"
        ),
        "role_seed": None,
        "budgets": ordered_budgets,
        "candidate_buildings": int(len(sampling)),
        "candidate_building_sha256": int_array_sha256(building_ids),
        "cells": cells,
    }
    validate_ladder(ladder, manifest)
    return ladder, manifest


def validate_ladder(ladder: pd.DataFrame, manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported building ladder schema")
    if manifest.get("sampling_profile") != SAMPLING_PROFILE:
        raise ValueError("unsupported building sampling profile")
    budgets = [int(value) for value in manifest["budgets"]]
    if len(ladder) != max(budgets):
        raise AssertionError("building ladder length differs from largest budget")
    expected_positions = np.arange(1, len(ladder) + 1, dtype="int64")
    if not np.array_equal(
        ladder["position"].to_numpy(dtype="int64"), expected_positions
    ):
        raise AssertionError("building ladder positions are not consecutive")
    if ladder["building_id"].duplicated().any():
        raise AssertionError("building ladder repeats a building")
    if ladder["building_id"].mod(2).any():
        raise AssertionError("building ladder contains an odd holdout building")
    early_stop_every = int(manifest["early_stop_every"])
    expected_roles = np.where(
        ladder["position"].mod(early_stop_every).eq(0),
        ROLE_EARLY_STOP,
        ROLE_FIT,
    )
    if not np.array_equal(ladder["role"].to_numpy(), expected_roles):
        raise AssertionError("tree roles do not follow the fixed position rule")

    previous: set[int] = set()
    previous_meter_counts: dict[str, int] | None = None
    meter_growth = int(
        manifest["meter_feasibility"]["minimum_growth_per_budget_transition"]
    )
    meter_minimum = int(manifest["meter_feasibility"]["minimum_at_smallest_budget"])
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
        if cell["reproducibility_digest"] != cell["available_building_sha256"]:
            raise AssertionError(f"K={budget} reproducibility digest mismatch")
        if sum(map(int, cell["site_counts"].values())) != budget:
            raise AssertionError(f"K={budget} site counts do not sum to budget")
        if not cell.get("site_stratified_sampling_applied"):
            raise AssertionError(f"K={budget} lacks site-stratified sampling gate")

        meter_counts = {
            str(meter): int(cell["meter_source_building_counts"][str(meter)])
            for meter in METER_IDS
        }
        for meter in METER_IDS:
            required = (
                meter_minimum
                if previous_meter_counts is None
                else previous_meter_counts[str(meter)] + meter_growth
            )
            detail = cell["meter_constraints"][str(meter)]
            if meter_counts[str(meter)] < required or not detail["passed"]:
                raise AssertionError(
                    f"K={budget} meter={meter} source-building coverage failed"
                )
            if int(detail["required_minimum"]) != required:
                raise AssertionError(
                    f"K={budget} meter={meter} stored requirement mismatch"
                )
        if not cell["meter_constraint_pass"] or not cell["constraint_pass"]:
            raise AssertionError(f"K={budget} meter feasibility gate failed")
        previous_meter_counts = meter_counts


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
                "sampling_method": "site_stratified_random_without_replacement",
                "sampling_attempt": int(row.sampling_attempt),
                "site_draw_rank": int(row.site_draw_rank),
                "site_candidate_count": int(row.site_candidate_count),
                "site_tie_priority": int(row.site_tie_priority),
                "labels_used_for_identity_selection": False,
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
