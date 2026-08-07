"""Audit representative building-ladder sensitivity without fitting models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lead import PROC, ROOT, load_m3_frame
from m5_building_curve_protocol import (
    DEFAULT_DIVERSIFICATION_ABSOLUTE_TOLERANCE,
    DEFAULT_DIVERSIFICATION_RELATIVE_TOLERANCE,
    DEFAULT_DIVERSIFICATION_TOP_N,
    DEFAULT_PREFIX_QUALITY_MAX_ABSOLUTE_DEGRADATION,
    DEFAULT_PREFIX_QUALITY_MAX_RATIO,
    atomic_write_json,
    build_building_profiles,
    build_nested_building_ladder,
    int_array_sha256,
    prefix_composition_discrepancy,
    validate_ladder,
)

DEFAULT_BUILDING_SEEDS = (42, 43, 44)
DEFAULT_BUDGETS = (10, 20, 50, 100)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _counts_and_shares(values: pd.Series, denominator: int) -> dict[str, Any]:
    counts = values.astype(str).value_counts().sort_index()
    return {
        "counts": {str(key): int(value) for key, value in counts.items()},
        "shares": {
            str(key): float(value / denominator) for key, value in counts.items()
        },
    }


def _composition(
    profiles: pd.DataFrame,
    building_ids: list[int],
    *,
    sampling_profile: str,
) -> dict[str, Any]:
    selected = profiles.set_index("building_id").loc[building_ids].reset_index()
    total_rows = int(selected["rows"].sum())
    total_anomalies = int(selected["anomalies"].sum())
    meter_presence: dict[str, Any] = {"counts": {}, "shares": {}}
    meter_row_share: dict[str, float] = {}
    for meter in range(4):
        present = int(selected[f"meter_{meter}_present"].sum())
        meter_presence["counts"][str(meter)] = present
        meter_presence["shares"][str(meter)] = float(present / len(selected))
        meter_row_share[str(meter)] = float(
            selected[f"meter_{meter}_rows"].sum() / total_rows
        )
    return {
        "site_distribution": _counts_and_shares(selected["site_id"], len(selected)),
        "primary_use_distribution": _counts_and_shares(
            selected["primary_use"].fillna("Unknown"), len(selected)
        ),
        "meter_presence": meter_presence,
        "meter_row_share": meter_row_share,
        "anomaly_rate_bin_distribution": _counts_and_shares(
            selected["anomaly_bin"], len(selected)
        ),
        "anomaly_bearing_meter_count_distribution": _counts_and_shares(
            selected["anomaly_meter_count"], len(selected)
        ),
        "building_size_bin_distribution": _counts_and_shares(
            selected["size_bin"], len(selected)
        ),
        "total_available_rows": total_rows,
        "total_anomaly_rows": total_anomalies,
        "natural_anomaly_prevalence": float(total_anomalies / total_rows),
        "candidate_target_discrepancy": prefix_composition_discrepancy(
            profiles, building_ids, sampling_profile=sampling_profile
        ),
    }


def _attach_primary_use(profiles: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    result = profiles.copy()
    if "primary_use" not in result:
        metadata = pd.read_csv(
            metadata_path, usecols=["building_id", "primary_use"]
        ).drop_duplicates("building_id")
        result = result.merge(metadata, on="building_id", how="left", validate="one_to_one")
    result["primary_use"] = result["primary_use"].fillna("Unknown").astype("string")
    return result.sort_values("building_id").reset_index(drop=True)


def _validate_profiles(profiles: pd.DataFrame) -> None:
    required = {
        "building_id",
        "site_id",
        "primary_use",
        "rows",
        "anomalies",
        "anomaly_rate",
        "anomaly_bin",
        "anomaly_meter_count",
        "size_bin",
        *(f"meter_{meter}_rows" for meter in range(4)),
        *(f"meter_{meter}_present" for meter in range(4)),
    }
    missing = required - set(profiles)
    if missing:
        raise ValueError(f"candidate profile missing {sorted(missing)}")
    if profiles["building_id"].duplicated().any():
        raise ValueError("candidate profile repeats a building")
    if profiles["building_id"].mod(2).any():
        raise ValueError("candidate profile contains odd holdout buildings")


def profiles_from_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Profile only even training buildings; odd labels are never passed onward."""
    candidate = frame.loc[frame["building_id"].mod(2).eq(0)].copy()
    return build_building_profiles(candidate)


def _add_profile_row_quotas(
    profiles: pd.DataFrame, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Reproduce formal incremental-block quotas from profile row counts."""
    result = json.loads(json.dumps(manifest))
    counts = profiles.set_index("building_id")["rows"]
    average_limit = int(result["average_rows_per_building_limit"])
    total_limit = int(result["max_context_rows"])
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
        allocation = np.ones(len(ids), dtype="int64")
        remaining = capacity - len(ids)
        room = available - 1
        if remaining and room.sum():
            ideal = remaining * room / room.sum()
            allocation += np.minimum(np.floor(ideal).astype("int64"), room)
        while int(allocation.sum()) < capacity:
            eligible = allocation < available
            if not eligible.any():
                break
            target = capacity * available / available.sum()
            deficit = np.where(eligible, target - allocation, -np.inf)
            chosen = int(np.lexsort((ids, -deficit))[0])
            allocation[chosen] += 1
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


def build_sensitivity_audit(
    profiles: pd.DataFrame,
    out_root: Path,
    *,
    building_seeds: tuple[int, ...] = DEFAULT_BUILDING_SEEDS,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    row_seed: int = 42,
    model_seed: int = 42,
    sampling_profile: str = "representative",
    quality_max_ratio: float = DEFAULT_PREFIX_QUALITY_MAX_RATIO,
    quality_max_absolute_degradation: float = (
        DEFAULT_PREFIX_QUALITY_MAX_ABSOLUTE_DEGRADATION
    ),
) -> dict[str, Any]:
    """Build all pilot artifacts from an already even-only candidate profile."""
    profiles = profiles.copy().sort_values("building_id").reset_index(drop=True)
    _validate_profiles(profiles)
    seeds = tuple(int(seed) for seed in building_seeds)
    ordered_budgets = tuple(sorted(set(int(budget) for budget in budgets)))
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("building seeds must contain at least two unique values")
    if quality_max_ratio < 1:
        raise ValueError("quality max ratio must be at least one")
    if quality_max_absolute_degradation < 0:
        raise ValueError("quality absolute degradation cannot be negative")

    canonical, canonical_manifest = build_nested_building_ladder(
        profiles,
        ordered_budgets,
        seed=42,
        sampling_profile=sampling_profile,
        diversify_candidates=False,
    )
    _atomic_csv(profiles, out_root / "candidate_building_profiles.csv")

    ladders: dict[int, pd.DataFrame] = {}
    manifests: dict[int, dict[str, Any]] = {}
    compositions: list[dict[str, Any]] = []
    ladder_files: dict[str, dict[str, str]] = {}
    all_quality_pass = True

    for seed in seeds:
        ladder, manifest = build_nested_building_ladder(
            profiles,
            ordered_budgets,
            seed=seed,
            sampling_profile=sampling_profile,
            diversify_candidates=True,
        )
        validate_ladder(ladder, manifest)
        manifest.update(
            {
                "experiment": "m5_building_candidate_sensitivity_pilot",
                "pilot_status": "selection_audit_only_no_models_fitted",
                "building_seed": seed,
                "row_seed": int(row_seed),
                "row_selection_seed": int(row_seed),
                "role_seed": None,
                "model_seed": int(model_seed),
                "row_policy": "average_building_cap",
                "average_rows_per_building_limit": 500,
                "max_context_rows": 50_000,
                "split": {
                    "candidate": "building_id % 2 == 0",
                    "canonical_test": "building_id % 2 == 1",
                    "odd_labels_used_for_selection": False,
                },
                "quality_gate": {
                    "metric": "representative prefix composition discrepancy",
                    "reference": "canonical exact-best greedy prefix",
                    "maximum_ratio": float(quality_max_ratio),
                    "maximum_absolute_degradation": float(
                        quality_max_absolute_degradation
                    ),
                    "absolute_epsilon": 1e-12,
                },
            }
        )
        manifest = _add_profile_row_quotas(profiles, manifest)
        for budget in ordered_budgets:
            selected_ids = list(
                map(int, manifest["cells"][str(budget)]["available_buildings"])
            )
            canonical_ids = list(
                map(
                    int,
                    canonical_manifest["cells"][str(budget)]["available_buildings"],
                )
            )
            composition = _composition(
                profiles, selected_ids, sampling_profile=sampling_profile
            )
            canonical_discrepancy = prefix_composition_discrepancy(
                profiles, canonical_ids, sampling_profile=sampling_profile
            )
            discrepancy = float(composition["candidate_target_discrepancy"])
            ratio = (
                discrepancy / canonical_discrepancy
                if canonical_discrepancy > 0
                else (1.0 if discrepancy <= 1e-12 else float("inf"))
            )
            absolute_degradation = max(0.0, discrepancy - canonical_discrepancy)
            passed = bool(
                discrepancy <= canonical_discrepancy * quality_max_ratio + 1e-12
                and absolute_degradation
                <= quality_max_absolute_degradation + 1e-12
            )
            all_quality_pass = all_quality_pass and passed
            manifest["cells"][str(budget)]["selection_quality"] = {
                "seeded_prefix_discrepancy": discrepancy,
                "canonical_best_greedy_prefix_discrepancy": canonical_discrepancy,
                "degradation_ratio": ratio,
                "maximum_ratio": float(quality_max_ratio),
                "absolute_degradation": absolute_degradation,
                "maximum_absolute_degradation": float(
                    quality_max_absolute_degradation
                ),
                "passed": passed,
            }
            compositions.append(
                {
                    "building_seed": seed,
                    "K": budget,
                    "prefix_discrepancy": discrepancy,
                    "canonical_best_greedy_discrepancy": canonical_discrepancy,
                    "degradation_ratio": ratio,
                    "quality_gate_max_ratio": float(quality_max_ratio),
                    "absolute_degradation": absolute_degradation,
                    "quality_gate_max_absolute_degradation": float(
                        quality_max_absolute_degradation
                    ),
                    "quality_gate_pass": passed,
                    "total_available_rows": composition["total_available_rows"],
                    "total_anomaly_rows": composition["total_anomaly_rows"],
                    "natural_anomaly_prevalence": composition[
                        "natural_anomaly_prevalence"
                    ],
                    "site_distribution_json": json.dumps(
                        composition["site_distribution"], sort_keys=True
                    ),
                    "primary_use_distribution_json": json.dumps(
                        composition["primary_use_distribution"], sort_keys=True
                    ),
                    "meter_presence_json": json.dumps(
                        composition["meter_presence"], sort_keys=True
                    ),
                    "meter_row_share_json": json.dumps(
                        composition["meter_row_share"], sort_keys=True
                    ),
                    "anomaly_rate_bin_distribution_json": json.dumps(
                        composition["anomaly_rate_bin_distribution"], sort_keys=True
                    ),
                    "anomaly_bearing_meter_count_distribution_json": json.dumps(
                        composition["anomaly_bearing_meter_count_distribution"],
                        sort_keys=True,
                    ),
                    "building_size_bin_distribution_json": json.dumps(
                        composition["building_size_bin_distribution"], sort_keys=True
                    ),
                }
            )

        enriched = ladder.rename(columns={"role": "tree_role"}).merge(
            profiles.drop(columns=["site_id"]),
            on="building_id",
            how="left",
            validate="one_to_one",
        )
        enriched.insert(0, "building_seed", seed)
        enriched.insert(1, "row_seed", int(row_seed))
        enriched.insert(2, "role_seed", pd.NA)
        enriched.insert(3, "model_seed", int(model_seed))
        enriched["meter_presence"] = enriched.apply(
            lambda row: ",".join(
                str(meter)
                for meter in range(4)
                if int(row[f"meter_{meter}_present"])
            ),
            axis=1,
        )
        enriched["first_included_K"] = enriched["position"].map(
            lambda position: next(
                (
                    budget
                    for budget in ordered_budgets
                    if int(position) <= int(budget)
                ),
                None,
            )
        )
        csv_path = out_root / f"building_ladder_seed{seed}.csv"
        json_path = out_root / f"building_ladder_seed{seed}.json"
        _atomic_csv(enriched, csv_path)
        manifest["ladder_csv"] = csv_path.name
        manifest["ladder_csv_sha256"] = _sha256_file(csv_path)
        manifest["candidate_profile_csv"] = "candidate_building_profiles.csv"
        atomic_write_json(json_path, manifest)
        ladder_files[str(seed)] = {
            "csv": csv_path.name,
            "csv_sha256": manifest["ladder_csv_sha256"],
            "json": json_path.name,
        }
        ladders[seed] = ladder
        manifests[seed] = manifest

    overlaps: list[dict[str, Any]] = []
    for budget in ordered_budgets:
        for left_index, left_seed in enumerate(seeds):
            for right_seed in seeds[left_index + 1 :]:
                left = set(
                    map(
                        int,
                        manifests[left_seed]["cells"][str(budget)][
                            "available_buildings"
                        ],
                    )
                )
                right = set(
                    map(
                        int,
                        manifests[right_seed]["cells"][str(budget)][
                            "available_buildings"
                        ],
                    )
                )
                intersection = len(left & right)
                union = len(left | right)
                overlaps.append(
                    {
                        "K": budget,
                        "seed_a": left_seed,
                        "seed_b": right_seed,
                        "intersection_count": intersection,
                        "union_count": union,
                        "jaccard_similarity": float(intersection / union),
                    }
                )

    overlap_frame = pd.DataFrame(overlaps)
    composition_frame = pd.DataFrame(compositions)
    _atomic_csv(overlap_frame, out_root / "building_overlap.csv")
    _atomic_csv(composition_frame, out_root / "composition_audit.csv")
    distinct_by_budget = {
        str(budget): len(
            {
                tuple(
                    manifests[seed]["cells"][str(budget)]["available_buildings"]
                )
                for seed in seeds
            }
        )
        for budget in ordered_budgets
    }
    meaningful_difference_pass = all(value == len(seeds) for value in distinct_by_budget.values())
    summary = {
        "schema_version": 1,
        "experiment": "m5_building_candidate_sensitivity_pilot",
        "status": (
            "audit_passed_ready_for_model_evaluation"
            if all_quality_pass and meaningful_difference_pass
            else "audit_failed_not_ready_for_model_evaluation"
        ),
        "sampling_profile": sampling_profile,
        "building_seeds": list(seeds),
        "row_seed": int(row_seed),
        "role_seed": None,
        "role_policy": "positions divisible by 5 are early_stop; all others fit",
        "model_seed": int(model_seed),
        "budgets": list(ordered_budgets),
        "candidate_buildings": int(len(profiles)),
        "candidate_building_sha256": int_array_sha256(
            profiles["building_id"].to_numpy(dtype="int64")
        ),
        "selection": {
            "method": "representative greedy with seed-controlled acceptable set",
            "acceptable_candidate_top_n": DEFAULT_DIVERSIFICATION_TOP_N,
            "relative_score_tolerance": DEFAULT_DIVERSIFICATION_RELATIVE_TOLERANCE,
            "absolute_score_tolerance": DEFAULT_DIVERSIFICATION_ABSOLUTE_TOLERANCE,
            "uncontrolled_rng": False,
        },
        "quality_gate": {
            "maximum_seeded_vs_canonical_discrepancy_ratio": float(
                quality_max_ratio
            ),
            "maximum_absolute_discrepancy_degradation": float(
                quality_max_absolute_degradation
            ),
            "all_passed": all_quality_pass,
        },
        "meaningful_difference_gate": {
            "required_distinct_prefixes_per_K": len(seeds),
            "distinct_prefixes_per_K": distinct_by_budget,
            "passed": meaningful_difference_pass,
        },
        "canonical_best_greedy_building_sha256": {
            str(budget): canonical_manifest["cells"][str(budget)][
                "available_building_sha256"
            ]
            for budget in ordered_budgets
        },
        "ladders": ladder_files,
        "outputs": {
            "overlap": "building_overlap.csv",
            "composition": "composition_audit.csv",
            "profiles": "candidate_building_profiles.csv",
        },
    }
    atomic_write_json(out_root / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--budgets", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--row-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--profiles-csv", type=Path)
    parser.add_argument(
        "--building-metadata",
        type=Path,
        default=ROOT / "data" / "raw" / "m3" / "building_metadata.csv",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=(
            PROC
            / "m5_building_curve"
            / "sensitivity"
            / "building_candidate_pilot"
        ),
    )
    parser.add_argument(
        "--quality-max-ratio",
        type=float,
        default=DEFAULT_PREFIX_QUALITY_MAX_RATIO,
    )
    parser.add_argument(
        "--quality-max-absolute-degradation",
        type=float,
        default=DEFAULT_PREFIX_QUALITY_MAX_ABSOLUTE_DEGRADATION,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.profiles_csv is None:
        frame = load_m3_frame(verbose=True)
        profiles = profiles_from_training_frame(frame)
        source = "lead.load_m3_frame filtered before profiling to building_id % 2 == 0"
    else:
        profiles = pd.read_csv(args.profiles_csv)
        source = str(args.profiles_csv.resolve())
    profiles = _attach_primary_use(profiles, args.building_metadata)
    summary = build_sensitivity_audit(
        profiles,
        args.out_root,
        building_seeds=tuple(args.building_seeds),
        budgets=tuple(args.budgets),
        row_seed=args.row_seed,
        model_seed=args.model_seed,
        quality_max_ratio=args.quality_max_ratio,
        quality_max_absolute_degradation=(
            args.quality_max_absolute_degradation
        ),
    )
    summary["profile_source"] = source
    summary["building_metadata"] = str(args.building_metadata.resolve())
    atomic_write_json(args.out_root / "summary.json", summary)
    print(
        f"{summary['status']}: wrote deterministic audit to {args.out_root}",
        flush=True,
    )
    return 0 if summary["status"] == "audit_passed_ready_for_model_evaluation" else 2


if __name__ == "__main__":
    raise SystemExit(main())
