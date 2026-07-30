"""Deterministic context construction and artifact gates for M5 stories.

The M5 paper probes change the training support while keeping the query and
the building-disjoint split frozen.  This module owns the part of that
contract that must be shared by the TabPFN and tree arms: row identity,
stratification, composition interventions, and provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import BASELINE_FEATURE_COLS, PROC, SHIFTS

M5_CONTEXT_ROOT = PROC / "m5_context_stories"
M5_CONTEXT_SCHEMA_VERSION = 1
M5_FIT_RULE = "building_id % 2 == 0"
M5_HOLDOUT_RULE = "building_id % 2 == 1"
M5_SENTINEL_SITES = {0: "Panther", 2: "Fox", 6: "Peacock", 9: "Bull"}
METER_NAME_TO_ID = {"electricity": 0, "chilledwater": 1, "steam": 2, "hotwater": 3}


def array_sha256(values: np.ndarray) -> str:
    """Hash an ordered raw-index vector using a platform-independent encoding."""
    array = np.asarray(values, dtype="int64")
    return hashlib.sha256(
        np.ascontiguousarray(array).astype("<i8").tobytes()
    ).hexdigest()


def stable_row_priority(raw_index: np.ndarray, *, seed: int) -> np.ndarray:
    """Return deterministic uint64 priorities without mutable RNG state."""
    values = np.asarray(raw_index, dtype="uint64")
    seed_value = np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)
    z = values ^ (seed_value + np.uint64(0x9E3779B97F4A7C15))
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def feature_names(feature_tag: str) -> list[str]:
    """Return the frozen feature list for F0/F4 context cells."""
    if feature_tag.upper() in {"F0", "17", "BASELINE"}:
        return list(BASELINE_FEATURE_COLS)
    if feature_tag.upper() in {"F4", "137", "FULL"}:
        return [
            *BASELINE_FEATURE_COLS,
            *[
                name
                for shift in SHIFTS
                for name in (f"lag_value_diff_{shift}", f"lag_value_ratio_{shift}")
            ],
        ]
    raise ValueError(f"unknown M5 feature tag: {feature_tag!r}")


def parse_context_tag(context_tag: str) -> dict[str, Any]:
    """Parse the public context intervention vocabulary."""
    tag = context_tag.strip().lower()
    if tag in {"pooled_reference", "meter_balanced", "site_balanced"}:
        kind, target, proportion = tag, None, None
    elif tag.startswith("meter_heavy:"):
        parts = tag.split(":")
        if len(parts) != 3:
            raise ValueError("meter_heavy must be meter_heavy:<meter>:<proportion>")
        kind, target, proportion = "meter_heavy", parts[1], float(parts[2])
    elif tag.startswith("meter_excluded:"):
        parts = tag.split(":")
        if len(parts) != 2:
            raise ValueError("meter_excluded must be meter_excluded:<meter>")
        kind, target, proportion = "meter_excluded", parts[1], None
    elif tag.startswith("site_source:"):
        parts = tag.split(":")
        if len(parts) != 2:
            raise ValueError("site_source must be site_source:<site_id>")
        kind, target, proportion = "site_source", parts[1], None
    else:
        raise ValueError(f"unknown M5 context tag: {context_tag!r}")
    if proportion is not None and not 0.0 <= proportion <= 1.0:
        raise ValueError("context proportion must be in [0, 1]")
    return {"kind": kind, "target": target, "proportion": proportion}


def context_tag_path(context_tag: str) -> str:
    """Make an intervention safe and readable as one path component."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", context_tag.strip().lower()).strip("_")


def fixed_score_indices(row_index: np.ndarray, rows: int, *, seed: int) -> np.ndarray:
    values = np.asarray(row_index, dtype="int64")
    if rows <= 0 or len(values) <= rows:
        return values.copy()
    return (
        np.random.RandomState(seed).choice(values, rows, replace=False).astype("int64")
    )


def protocol_source(
    frame: pd.DataFrame, *, seed: int = 42, validation_rows: int = 4_000
) -> dict[str, Any]:
    """Return the frozen even-building source and odd-building holdout metadata."""
    require_columns(frame, ("building_id", "anomaly", "meter", "site_id"))
    raw = raw_indices(frame)
    fit_mask = frame["building_id"].to_numpy() % 2 == 0
    holdout_mask = ~fit_mask
    fit_rows = raw[fit_mask]
    validation = fixed_score_indices(fit_rows, validation_rows, seed=seed + 20_000)
    candidate = fit_rows[~np.isin(fit_rows, validation)]
    holdout = raw[holdout_mask]
    fit_buildings = set(int(x) for x in frame.loc[fit_mask, "building_id"].unique())
    holdout_buildings = set(
        int(x) for x in frame.loc[holdout_mask, "building_id"].unique()
    )
    if fit_buildings & holdout_buildings:
        raise AssertionError("M5 fit/holdout building overlap")
    return {
        "fit_rows": fit_rows,
        "candidate_rows": candidate,
        "validation_rows": validation,
        "holdout_rows": holdout,
        "fit_rule": M5_FIT_RULE,
        "holdout_rule": M5_HOLDOUT_RULE,
        "fit_buildings": fit_buildings,
        "holdout_buildings": holdout_buildings,
    }


def raw_indices(frame: pd.DataFrame) -> np.ndarray:
    if "_raw_index" in frame.columns:
        values = frame["_raw_index"].to_numpy(dtype="int64")
    else:
        values = frame.index.to_numpy(dtype="int64")
    if len(np.unique(values)) != len(values):
        raise ValueError("M5 frame raw indices must be unique")
    return values


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError("M5 context frame missing columns: " + ", ".join(missing))


def _largest_remainder_counts(
    weights: np.ndarray, total: int, capacity: np.ndarray
) -> np.ndarray:
    if total < 0 or int(capacity.sum()) < total:
        raise ValueError("context intervention does not have enough eligible rows")
    weights = np.asarray(weights, dtype="float64").copy()
    capacity = np.asarray(capacity, dtype="int64")
    if weights.sum() <= 0:
        raise ValueError("context intervention has no eligible support")
    weights /= weights.sum()
    desired = total * weights
    counts = np.minimum(np.floor(desired).astype("int64"), capacity)
    while int(counts.sum()) < total:
        remaining = capacity - counts
        if not remaining.any():
            raise ValueError("context intervention allocation exhausted support")
        score = desired - counts.astype("float64")
        score[remaining <= 0] = -math.inf
        choice = int(np.argmax(score))
        counts[choice] += 1
    return counts


def _group_plan(
    frame: pd.DataFrame, candidate_mask: np.ndarray, spec: dict[str, Any]
) -> tuple[str, list[Any], np.ndarray]:
    kind = spec["kind"]
    if kind.startswith("meter"):
        column, values = (
            "meter",
            sorted(frame.loc[candidate_mask, "meter"].unique().tolist()),
        )
    elif kind.startswith("site"):
        column, values = (
            "site_id",
            sorted(frame.loc[candidate_mask, "site_id"].unique().tolist()),
        )
    else:
        column, values = (
            "meter",
            sorted(frame.loc[candidate_mask, "meter"].unique().tolist()),
        )
    if kind == "site_source":
        values = [spec["target"]]
    if kind == "meter_excluded":
        values = [
            value
            for value in values
            if str(value).lower() != str(spec["target"]).lower()
        ]
    if not values:
        raise ValueError(f"context selector {kind} has no eligible groups")
    return (
        column,
        values,
        np.asarray(
            [candidate_mask & (frame[column].to_numpy() == value) for value in values]
        ),
    )


def context_indices(
    frame: pd.DataFrame,
    *,
    context_rows: int,
    context_tag: str,
    seed: int,
    candidate_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Select an exact-size deterministic, label-balanced context."""
    require_columns(frame, ("building_id", "anomaly", "meter", "site_id"))
    if context_rows <= 0 or context_rows % 2:
        raise ValueError("M5 contexts must be positive and even")
    raw = raw_indices(frame)
    candidate_rows = (
        raw if candidate_rows is None else np.asarray(candidate_rows, dtype="int64")
    )
    if np.array_equal(raw, np.arange(len(raw), dtype="int64")):
        candidate_pos = candidate_rows.copy()
    elif np.all(raw[1:] > raw[:-1]):
        candidate_pos = np.searchsorted(raw, candidate_rows)
        if np.any(candidate_pos >= len(raw)) or not np.array_equal(
            raw[candidate_pos], candidate_rows
        ):
            raise ValueError("candidate raw indices are absent from source frame")
    else:
        position = {int(value): index for index, value in enumerate(raw)}
        try:
            candidate_pos = np.asarray(
                [position[int(value)] for value in candidate_rows], dtype="int64"
            )
        except KeyError as error:
            raise ValueError(
                f"candidate row {error.args[0]} is absent from frame"
            ) from error
    if len(np.unique(candidate_rows)) != len(candidate_rows):
        raise ValueError("candidate raw indices must be unique")
    labels = frame["anomaly"].to_numpy()
    priorities = stable_row_priority(raw, seed=seed)
    spec = parse_context_tag(context_tag)
    target = spec["target"]
    if spec["kind"].startswith("meter") and isinstance(target, str):
        target = METER_NAME_TO_ID.get(target.lower(), target)
    selected: list[np.ndarray] = []
    frame_group_values = {
        "meter": frame["meter"].to_numpy(),
        "site_id": frame["site_id"].to_numpy(),
    }
    for label in (0, 1):
        label_positions = candidate_pos[labels[candidate_pos] == label]
        if spec["kind"].startswith("meter"):
            column = "meter"
        elif spec["kind"].startswith("site"):
            column = "site_id"
        else:
            column = "meter"
        values = sorted(np.unique(frame_group_values[column][label_positions]).tolist())
        if spec["kind"] == "site_source":
            values = [target]
        if spec["kind"] == "meter_excluded":
            values = [
                value for value in values if str(value).lower() != str(target).lower()
            ]
        if not values:
            raise ValueError(f"context selector {spec['kind']} has no eligible groups")
        group_masks = [
            label_positions[frame_group_values[column][label_positions] == value]
            for value in values
        ]
        capacities = np.asarray([len(mask) for mask in group_masks], dtype="int64")
        if spec["kind"] == "meter_balanced" or spec["kind"] == "site_balanced":
            weights = np.ones(len(values), dtype="float64")
        elif spec["kind"] == "meter_heavy":
            target_text = str(target).lower()
            weights = np.asarray(
                [
                    spec["proportion"] if str(value).lower() == target_text else 0.0
                    for value in values
                ]
            )
            remainder = max(0.0, 1.0 - float(spec["proportion"]))
            others = np.asarray(
                [
                    capacity if str(value).lower() != target_text else 0
                    for value, capacity in zip(values, capacities)
                ]
            )
            if others.sum():
                weights += remainder * others / others.sum()
        elif spec["kind"] in {"meter_excluded", "site_source"}:
            weights = capacities.astype("float64")
        else:
            weights = capacities.astype("float64")
        quota = _largest_remainder_counts(weights, context_rows // 2, capacities)
        for value, mask, take in zip(values, group_masks, quota, strict=True):
            positions = mask
            order = np.lexsort((raw[positions], priorities[positions]))
            selected.append(positions[order[:take]])
    positions = np.concatenate(selected) if selected else np.empty(0, dtype="int64")
    order = np.lexsort((raw[positions], priorities[positions], labels[positions]))
    return raw[positions[order]].astype("int64", copy=False)


def _counts(frame: pd.DataFrame, indices: np.ndarray, column: str) -> dict[str, int]:
    values = frame.loc[list(map(int, indices)), column]
    return {
        str(key): int(value)
        for key, value in values.value_counts().sort_index().items()
    }


def context_summary(frame: pd.DataFrame, indices: np.ndarray) -> dict[str, Any]:
    raw = raw_indices(frame)
    values = np.asarray(indices, dtype="int64")
    if np.array_equal(raw, np.arange(len(raw), dtype="int64")):
        positions = values
    elif np.all(raw[1:] > raw[:-1]):
        positions = np.searchsorted(raw, values)
    else:
        position = {int(value): index for index, value in enumerate(raw)}
        positions = np.asarray(
            [position[int(value)] for value in values], dtype="int64"
        )
    row_values = frame.iloc[positions]
    labels = row_values["anomaly"].to_numpy()
    meters = row_values["meter"].to_numpy()
    sites = row_values["site_id"].to_numpy()
    frame_for_counts = row_values
    return {
        "rows": int(len(values)),
        "unique_rows": int(len(np.unique(values))),
        "raw_index_sha256": array_sha256(values),
        "label_counts": {
            "negative": int((labels == 0).sum()),
            "positive": int((labels == 1).sum()),
        },
        "meter_counts": {
            str(key): int(value)
            for key, value in pd.Series(meters).value_counts().sort_index().items()
        },
        "meter_label_counts": {
            str(meter): {
                str(label): int(((meters == meter) & (labels == label)).sum())
                for label in (0, 1)
            }
            for meter in sorted(np.unique(meters))
        },
        "site_counts": {
            str(key): int(value)
            for key, value in pd.Series(sites).value_counts().sort_index().items()
        },
        "site_label_counts": {
            str(site): {
                str(label): int(((sites == site) & (labels == label)).sum())
                for label in (0, 1)
            }
            for site in sorted(np.unique(sites))
        },
        "building_count": int(frame_for_counts["building_id"].nunique()),
    }


def build_context_manifest(
    frame: pd.DataFrame,
    indices: np.ndarray,
    *,
    story: str,
    context_tag: str,
    context_rows: int,
    context_seed: int,
    model_seed: int,
    feature_tag: str,
    source_artifact: dict[str, Any] | None = None,
    split: dict[str, Any] | None = None,
    creation_command: str | None = None,
) -> dict[str, Any]:
    indices = np.asarray(indices, dtype="int64")
    summary = context_summary(frame, indices)
    if summary["rows"] != context_rows or summary["unique_rows"] != context_rows:
        raise AssertionError("context row count or uniqueness gate failed")
    return {
        "schema_version": M5_CONTEXT_SCHEMA_VERSION,
        "artifact_type": "m5_context_manifest",
        "story": story,
        "context_tag": context_tag,
        "context_rows": int(context_rows),
        "context_seed": int(context_seed),
        "model_seed": int(model_seed),
        "feature_tag": feature_tag,
        "feature_count": len(feature_names(feature_tag)),
        "feature_names": feature_names(feature_tag),
        "raw_index": [int(value) for value in indices],
        **summary,
        "split": split or {"fit_rule": M5_FIT_RULE, "holdout_rule": M5_HOLDOUT_RULE},
        "scaler_provenance": {
            "fit_on": "context_rows",
            "method": "StandardScaler",
            "statistics": "model-fit artifact",
        },
        "source_artifact": source_artifact
        or {"name": "load_m3_frame", "raw_index_semantics": "M3 positional row index"},
        "creation_command": creation_command,
    }


def validate_context_manifest(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    holdout_rows: np.ndarray | None = None,
) -> None:
    """Fail loudly when an artifact no longer satisfies the M5 protocol."""
    required = {
        "schema_version",
        "raw_index",
        "raw_index_sha256",
        "context_rows",
        "label_counts",
        "meter_label_counts",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(
            "context manifest missing fields: " + ", ".join(sorted(missing))
        )
    if manifest["schema_version"] != M5_CONTEXT_SCHEMA_VERSION:
        raise ValueError("unsupported M5 context manifest schema")
    indices = np.asarray(manifest["raw_index"], dtype="int64")
    if len(indices) != int(manifest["context_rows"]):
        raise AssertionError("context manifest row count mismatch")
    if len(np.unique(indices)) != len(indices):
        raise AssertionError("context contains duplicate raw indices")
    if array_sha256(indices) != manifest["raw_index_sha256"]:
        raise AssertionError("context raw-index digest mismatch")
    source = set(map(int, raw_indices(frame)))
    if not set(map(int, indices)) <= source:
        raise AssertionError("context contains raw indices absent from source frame")
    source_raw = raw_indices(frame)
    if np.array_equal(source_raw, np.arange(len(source_raw), dtype="int64")):
        values = frame.iloc[indices]
    elif "_raw_index" in frame.columns:
        values = frame.set_index("_raw_index").loc[indices]
    else:
        positions = np.searchsorted(source_raw, indices)
        if np.any(positions >= len(source_raw)) or not np.array_equal(
            source_raw[positions], indices
        ):
            raise AssertionError("context raw indices are absent from source frame")
        values = frame.iloc[positions]
    if values["building_id"].mod(2).any():
        raise AssertionError("context contains holdout building rows")
    expected = context_summary(frame, indices)
    for key in (
        "label_counts",
        "meter_label_counts",
        "raw_index_sha256",
        "rows",
        "unique_rows",
    ):
        if manifest.get(key) != expected.get(key):
            raise AssertionError(f"context manifest {key} drifted")
    if holdout_rows is not None and set(map(int, indices)) & set(
        map(int, holdout_rows)
    ):
        raise AssertionError("context overlaps holdout rows")
    spec = parse_context_tag(manifest["context_tag"])
    excluded_target = METER_NAME_TO_ID.get(str(spec["target"]).lower(), spec["target"])
    if spec["kind"] == "meter_excluded" and str(excluded_target).lower() in {
        str(x).lower() for x in expected["meter_counts"]
    }:
        raise AssertionError("meter-excluded context contains excluded meter")


def build_query_artifact(
    frame: pd.DataFrame,
    *,
    holdout_rows: np.ndarray,
    query_set: str = "screening",
    seed: int = 42,
    rows_per_cell: int = 16,
) -> tuple[dict[str, Any], np.ndarray]:
    """Build a small, ordered sentinel query set from the odd-building holdout."""
    require_columns(frame, ("building_id", "anomaly", "meter", "site_id"))
    if rows_per_cell <= 0:
        raise ValueError("rows_per_cell must be positive")
    raw = raw_indices(frame)
    positional = np.array_equal(raw, np.arange(len(raw), dtype="int64"))
    position = (
        None if positional else {int(value): index for index, value in enumerate(raw)}
    )
    holdout = np.asarray(holdout_rows, dtype="int64")
    priority = stable_row_priority(raw, seed=seed)
    if positional:
        holdout_pos = holdout.copy()
    else:
        assert position is not None
        holdout_pos = np.asarray(
            [position[int(value)] for value in holdout], dtype="int64"
        )
    frame_site = frame["site_id"].to_numpy()
    frame_meter = frame["meter"].to_numpy()
    frame_label = frame["anomaly"].to_numpy()
    selected: list[int] = []
    cells: list[dict[str, Any]] = []
    for site, site_name in sorted(M5_SENTINEL_SITES.items()):
        for meter in sorted(frame["meter"].unique()):
            for label in (0, 1):
                candidates_pos = holdout_pos[
                    (frame_site[holdout_pos] == site)
                    & (frame_meter[holdout_pos] == meter)
                    & (frame_label[holdout_pos] == label)
                ]
                order = np.lexsort((raw[candidates_pos], priority[candidates_pos]))
                chosen = raw[candidates_pos[order[:rows_per_cell]]]
                selected.extend(int(row) for row in chosen)
                cells.append(
                    {
                        "site_id": int(site),
                        "site_name": site_name,
                        "meter": int(meter),
                        "label": int(label),
                        "rows": len(chosen),
                    }
                )
    if not selected:
        raise ValueError("screening query selection found no sentinel rows")
    indices = np.asarray(selected, dtype="int64")
    if positional:
        query_positions = indices
    else:
        assert position is not None
        query_positions = np.asarray(
            [position[int(row)] for row in indices], dtype="int64"
        )
    labels = frame.iloc[query_positions]["anomaly"].to_numpy(dtype="int8")
    manifest = {
        "schema_version": M5_CONTEXT_SCHEMA_VERSION,
        "artifact_type": "m5_query_artifact",
        "query_set": query_set,
        "query_seed": int(seed),
        "raw_index": [int(value) for value in indices],
        "raw_index_sha256": array_sha256(indices),
        "query_rows": int(len(indices)),
        "label_counts": {
            "negative": int((labels == 0).sum()),
            "positive": int((labels == 1).sum()),
        },
        "sentinel_sites": {str(k): v for k, v in sorted(M5_SENTINEL_SITES.items())},
        "rows_per_cell": int(rows_per_cell),
        "cells": cells,
        "split": {"holdout_rule": M5_HOLDOUT_RULE, "fit_rule": M5_FIT_RULE},
    }
    if len(np.unique(indices)) != len(indices):
        raise AssertionError("query artifact contains duplicate raw indices")
    if not set(map(int, indices)) <= set(map(int, holdout)):
        raise AssertionError("query artifact is not a holdout subset")
    return manifest, indices


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def manifest_path(
    root: Path, *, story: str, context_tag: str, context_rows: int, seed: int
) -> Path:
    return (
        Path(root)
        / "manifests"
        / story
        / context_tag_path(context_tag)
        / f"n{context_rows}"
        / f"seed{seed}.json"
    )


def query_paths(root: Path, query_set: str) -> tuple[Path, Path]:
    directory = Path(root) / "queries" / query_set
    return directory / "manifest.json", directory / "queries.npz"
