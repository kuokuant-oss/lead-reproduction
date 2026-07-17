"""Pure split and sampling contracts for additive M6 site-transfer runs.

Nothing in this module mutates the frozen M3 implementation under ``src/lead``.
The functions are intentionally model-free so manifests can be prepared and
tested before an expensive full-data run starts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from lead import DOWNSAMPLE_SEEDS


SITE_IDS = tuple(range(16))
A3_SITE_FOLDS: dict[int, tuple[int, ...]] = {
    fold: tuple(site for site in SITE_IDS if site % 4 == fold) for fold in range(4)
}
METER_KEY_COLUMNS = ("site_id", "building_id", "meter")


def array_fingerprint(values: np.ndarray) -> str:
    """Return a stable hash for a numeric/string manifest array."""
    array = np.asarray(values)
    if array.dtype.kind in {"O", "U"}:
        payload = "\n".join(str(value) for value in array.tolist()).encode("utf-8")
    else:
        payload = np.ascontiguousarray(array).view(np.uint8)
    return hashlib.sha256(payload).hexdigest()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"M6 protocol requires columns: {', '.join(missing)}")


def location_split_masks(
    frame: pd.DataFrame,
    split_name: str,
    *,
    fold: int | None = None,
    site_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return source/test masks for A0, A1, A2, A3, or A4."""
    _require_columns(frame, ("site_id", "building_id"))
    if split_name == "a0_building_mod2":
        test = (frame["building_id"] % 2 == 1).to_numpy()
        rule = "test buildings are building_id % 2 == 1"
        unit_type = "building_id"
    elif split_name == "a0_building_mod2_inverse":
        # Complement of a0_building_mod2. Running both folds gives every
        # building an out-of-fold prediction, so A0 can be scored over a whole
        # site instead of only the half a single fold holds out.
        test = (frame["building_id"] % 2 == 0).to_numpy()
        rule = "test buildings are building_id % 2 == 0"
        unit_type = "building_id"
    elif split_name == "a1_even_to_odd":
        test = (frame["site_id"] % 2 == 1).to_numpy()
        rule = "train even sites; test odd sites"
        unit_type = "site_id"
    elif split_name == "a2_odd_to_even":
        test = (frame["site_id"] % 2 == 0).to_numpy()
        rule = "train odd sites; test even sites"
        unit_type = "site_id"
    elif split_name == "a3_group4":
        if fold not in A3_SITE_FOLDS:
            raise ValueError(f"A3 fold must be one of {sorted(A3_SITE_FOLDS)}")
        held_out_sites = A3_SITE_FOLDS[int(fold)]
        test = frame["site_id"].isin(held_out_sites).to_numpy()
        rule = f"test sites have site_id % 4 == {fold}"
        unit_type = "site_id"
    elif split_name == "a4_loso":
        if site_id not in SITE_IDS:
            raise ValueError(f"A4 site_id must be one of {list(SITE_IDS)}")
        test = (frame["site_id"] == site_id).to_numpy()
        rule = f"leave site_id == {site_id} out"
        unit_type = "site_id"
    else:
        raise ValueError(f"Unknown M6 split: {split_name}")
    train = ~test
    manifest = split_manifest(
        frame,
        train,
        test,
        name=split_name,
        rule=rule,
        unit_type=unit_type,
        require_site_disjoint=unit_type == "site_id",
    )
    if fold is not None:
        manifest["fold"] = int(fold)
    if site_id is not None:
        manifest["site_id"] = int(site_id)
    return train, test, manifest


def in_site_oracle_masks(
    frame: pd.DataFrame,
    site_id: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return A5 building-disjoint train/test masks inside one target site."""
    _require_columns(frame, ("site_id", "building_id"))
    in_site = (frame["site_id"] == site_id).to_numpy()
    train = in_site & (frame["building_id"] % 2 == 0).to_numpy()
    test = in_site & (frame["building_id"] % 2 == 1).to_numpy()
    if not train.any() or not test.any():
        raise ValueError(f"site {site_id} needs both even and odd building IDs")
    manifest = split_manifest(
        frame,
        train,
        test,
        name="a5_in_site_oracle",
        rule=(
            f"site_id == {site_id}; oracle train even building IDs; "
            "oracle test odd building IDs"
        ),
        unit_type="building_id_within_site",
        require_site_disjoint=False,
    )
    manifest.update(
        {
            "site_id": int(site_id),
            "target_label_access": "oracle_train_only",
            "excluded_rows": int((~(train | test)).sum()),
        }
    )
    return train, test, manifest


def source_building_calibration_masks(
    frame: pd.DataFrame,
    source_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split source rows into model-fit and building-disjoint calibration rows."""
    _require_columns(frame, ("site_id", "building_id", "anomaly"))
    source_mask = np.asarray(source_mask, dtype=bool)
    calibration = source_mask & (frame["building_id"] % 5 == 4).to_numpy()
    model_fit = source_mask & ~calibration
    if not model_fit.any() or not calibration.any():
        raise ValueError("source calibration needs non-empty fit and calibration rows")
    manifest = split_manifest(
        frame,
        model_fit,
        calibration,
        name="source_building_mod5_calibration",
        rule="within source rows, calibration buildings have building_id % 5 == 4",
        unit_type="building_id_within_source_sites",
        require_site_disjoint=False,
    )
    manifest["source_rows"] = int(source_mask.sum())
    manifest["excluded_source_rows"] = int(
        source_mask.sum() - model_fit.sum() - calibration.sum()
    )
    return model_fit, calibration, manifest


def split_manifest(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    name: str,
    rule: str,
    unit_type: str,
    require_site_disjoint: bool,
) -> dict[str, Any]:
    """Validate a split and describe its rows, groups, and class balance."""
    _require_columns(frame, ("site_id", "building_id", "anomaly"))
    train_mask = np.asarray(train_mask, dtype=bool)
    test_mask = np.asarray(test_mask, dtype=bool)
    if len(train_mask) != len(frame) or len(test_mask) != len(frame):
        raise ValueError("split masks must have one value per frame row")
    if np.any(train_mask & test_mask):
        raise AssertionError(f"{name} has row overlap")
    train_sites = set(int(value) for value in frame.loc[train_mask, "site_id"].unique())
    test_sites = set(int(value) for value in frame.loc[test_mask, "site_id"].unique())
    train_buildings = set(
        int(value) for value in frame.loc[train_mask, "building_id"].unique()
    )
    test_buildings = set(
        int(value) for value in frame.loc[test_mask, "building_id"].unique()
    )
    site_overlap = train_sites & test_sites
    building_overlap = train_buildings & test_buildings
    if require_site_disjoint and site_overlap:
        raise AssertionError(f"{name} has site overlap: {sorted(site_overlap)}")
    if building_overlap:
        raise AssertionError(
            f"{name} has building overlap: {sorted(building_overlap)[:5]}"
        )

    def side(mask: np.ndarray, sites: set[int], buildings: set[int]) -> dict[str, Any]:
        y = frame.loc[mask, "anomaly"]
        return {
            "rows": int(mask.sum()),
            "anomalies": int(y.sum()),
            "anomaly_rate": float(y.mean()),
            "sites": len(sites),
            "site_ids": sorted(sites),
            "buildings": len(buildings),
        }

    return {
        "name": name,
        "rule": rule,
        "unit_type": unit_type,
        "train": side(train_mask, train_sites, train_buildings),
        "test": side(test_mask, test_sites, test_buildings),
        "site_overlap": sorted(site_overlap),
        "building_overlap": sorted(building_overlap),
    }


def grouped_support(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    group_column: str = "site_id",
) -> dict[str, dict[str, Any]]:
    """Record plot-ready row/building/meter/class support by group."""
    _require_columns(
        frame,
        (group_column, "building_id", "meter", "anomaly"),
    )
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(frame):
        raise ValueError("support mask must have one value per frame row")
    subset = frame.loc[
        mask,
        [group_column, "building_id", "meter", "anomaly"],
    ]
    output: dict[str, dict[str, Any]] = {}
    for value, group in subset.groupby(group_column, sort=True):
        anomalies = int(group["anomaly"].sum())
        rows = int(len(group))
        output[str(int(value))] = {
            "group_value": int(value),
            "rows": rows,
            "anomalies": anomalies,
            "anomaly_rate": float(anomalies / rows),
            "buildings": int(group["building_id"].nunique()),
            "meter_series": int(
                group[["building_id", "meter"]].drop_duplicates().shape[0]
            ),
            "meter_types": sorted(int(item) for item in group["meter"].unique()),
        }
    return output


def source_meter_units(frame: pd.DataFrame, source_mask: np.ndarray) -> pd.DataFrame:
    """Return one deterministic row per source-site meter time series."""
    _require_columns(frame, METER_KEY_COLUMNS)
    source_mask = np.asarray(source_mask, dtype=bool)
    if len(source_mask) != len(frame):
        raise ValueError("source mask must have one value per frame row")
    return (
        frame.loc[source_mask, list(METER_KEY_COLUMNS)]
        .drop_duplicates()
        .sort_values(list(METER_KEY_COLUMNS))
        .reset_index(drop=True)
    )


def stratified_nested_meter_order(
    meter_units: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Create a nested, progressively proportional ordering across sites.

    The first block contains one meter from every source site. Remaining slots
    go to the site with the smallest selected/available ratio. Taking any
    prefix of at least ``n_sites`` therefore preserves source-site coverage.
    """
    _require_columns(meter_units, METER_KEY_COLUMNS)
    if meter_units.duplicated(list(METER_KEY_COLUMNS)).any():
        raise ValueError("meter_units must contain unique site/building/meter rows")
    sites = sorted(int(value) for value in meter_units["site_id"].unique())
    if not sites:
        raise ValueError("meter_units is empty")
    rng = np.random.RandomState(seed)
    queues: dict[int, list[tuple[int, int, int]]] = {}
    available: dict[int, int] = {}
    for site in sites:
        values = [
            tuple(int(value) for value in row)
            for row in meter_units.loc[
                meter_units["site_id"] == site,
                list(METER_KEY_COLUMNS),
            ].to_numpy()
        ]
        permutation = rng.permutation(len(values))
        queues[site] = [values[index] for index in permutation]
        available[site] = len(values)

    selected = {site: 0 for site in sites}
    ordered: list[tuple[int, int, int]] = []
    for site in sites:
        ordered.append(queues[site][selected[site]])
        selected[site] += 1
    while len(ordered) < len(meter_units):
        candidates = [site for site in sites if selected[site] < available[site]]
        site = min(
            candidates, key=lambda value: (selected[value] / available[value], value)
        )
        ordered.append(queues[site][selected[site]])
        selected[site] += 1
    return pd.DataFrame(ordered, columns=METER_KEY_COLUMNS)


def meter_budget_manifest(
    frame: pd.DataFrame,
    source_mask: np.ndarray,
    *,
    budget: int,
    seed: int,
) -> dict[str, Any]:
    """Freeze a B1 source-site-stratified nested meter selection."""
    units = source_meter_units(frame, source_mask)
    n_sites = int(units["site_id"].nunique())
    if budget < n_sites:
        raise ValueError(
            f"meter budget {budget} cannot cover all {n_sites} source sites"
        )
    if budget > len(units):
        raise ValueError(f"meter budget {budget} exceeds {len(units)} available")
    selected = stratified_nested_meter_order(units, seed=seed).iloc[:budget].copy()
    allocation = {
        str(site): int(count)
        for site, count in selected.groupby("site_id", sort=True).size().items()
    }
    records = selected.astype(int).to_dict(orient="records")
    fingerprint_values = selected.to_numpy(dtype="int64", copy=True)
    return {
        "schema_version": 1,
        "selection": "source_site_stratified_nested_meter_prefix",
        "meter_key": list(METER_KEY_COLUMNS),
        "seed": int(seed),
        "budget": int(budget),
        "available_meters": int(len(units)),
        "source_sites": n_sites,
        "site_allocation": allocation,
        "all_source_sites_covered": len(allocation) == n_sites,
        "selected_meter_sha256": array_fingerprint(fingerprint_values),
        "selected_meters": records,
    }


def meter_manifest_mask(frame: pd.DataFrame, manifest: dict[str, Any]) -> np.ndarray:
    """Return rows belonging to the exact B1 meters frozen in a manifest."""
    _require_columns(frame, METER_KEY_COLUMNS)
    selected = pd.DataFrame(manifest["selected_meters"])
    _require_columns(selected, METER_KEY_COLUMNS)
    selected_index = pd.MultiIndex.from_frame(selected[list(METER_KEY_COLUMNS)])
    frame_index = pd.MultiIndex.from_frame(frame[list(METER_KEY_COLUMNS)])
    return np.asarray(frame_index.isin(selected_index), dtype=bool)


def matched_anomaly_fit_indices(
    y: pd.Series,
    *,
    positive_budget: int,
    selection_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a B2 fit index while preserving frozen M3 sampling shape.

    M3 uses ``[negs1, pos, negs2, pos]``. B2 first samples the same number of
    *unique* anomaly rows in every split, then retains the two negative draws
    and duplicated positive block. The returned fit set therefore contains
    ``4 * positive_budget`` rows and has effective 50:50 class balance.
    """
    positive = y.index[y == 1].to_numpy()
    negative = y.index[y == 0].to_numpy()
    if positive_budget <= 0:
        raise ValueError("positive_budget must be positive")
    if positive_budget > len(positive):
        raise ValueError(
            f"positive budget {positive_budget} exceeds {len(positive)} anomalies"
        )
    if positive_budget > len(negative):
        raise ValueError(
            f"positive budget {positive_budget} exceeds {len(negative)} normals"
        )
    if positive_budget == len(positive):
        sampled_positive = positive.copy()
    else:
        sampled_positive = np.random.RandomState(selection_seed).choice(
            positive,
            positive_budget,
            replace=False,
        )
    negs1 = np.random.RandomState(DOWNSAMPLE_SEEDS[0]).choice(
        negative,
        positive_budget,
        replace=False,
    )
    negs2 = np.random.RandomState(DOWNSAMPLE_SEEDS[1]).choice(
        negative,
        positive_budget,
        replace=False,
    )
    fit_index = np.concatenate([negs1, sampled_positive, negs2, sampled_positive])
    return fit_index, {
        "selection": "matched_unique_anomaly_rows_with_frozen_m3_downsampling",
        "selection_seed": int(selection_seed),
        "unique_anomaly_rows": int(positive_budget),
        "sampled_normal_rows_per_block": int(positive_budget),
        "positive_blocks": 2,
        "negative_blocks": 2,
        "fit_rows": int(len(fit_index)),
        "effective_fit_anomaly_rate": 0.5,
        "sampled_positive_index_sha256": array_fingerprint(sampled_positive),
        "sampled_negative_1_index_sha256": array_fingerprint(negs1),
        "sampled_negative_2_index_sha256": array_fingerprint(negs2),
        "fit_index_sha256": array_fingerprint(fit_index),
    }
