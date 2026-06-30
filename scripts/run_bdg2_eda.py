"""Run read-only BDG2 EDA for the Phase E pre-modeling slice."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lead import load_m3_frame

ROOT = Path(__file__).resolve().parents[1]
BDG2_DIR = ROOT / "data" / "raw" / "bdg2"
M3_DIR = ROOT / "data" / "raw" / "m3"
JSON_OUT = ROOT / "data" / "processed" / "bdg2_eda.json"
REPORT_OUT = ROOT / "docs" / "reports" / "bdg2-eda.md"
HANDOFF_OUT = ROOT / "docs" / "handoffs" / "2026-06-30-bdg2-eda.md"
ASSET_DIR = ROOT / "docs" / "assets" / "bdg2-eda"

METER_TYPES = (
    "electricity",
    "chilledwater",
    "steam",
    "hotwater",
    "gas",
    "water",
    "irrigation",
    "solar",
)
EXPECTED_HOURS = 17_544
SUFFICIENT_OBS_MISSING_RATE = 0.50


def _non_empty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _normalise_category(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace("_", " ")


def _series_stats(series: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0}
    quantiles = clean.quantile([0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(clean.size),
        "median": float(quantiles.loc[0.5]),
        "iqr": float(quantiles.loc[0.75] - quantiles.loc[0.25]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def _ks_statistic(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = np.sort(left[~np.isnan(left)])
    right = np.sort(right[~np.isnan(right)])
    if left.size == 0 or right.size == 0:
        return None
    values = np.sort(np.unique(np.concatenate([left, right])))
    left_cdf = np.searchsorted(left, values, side="right") / left.size
    right_cdf = np.searchsorted(right, values, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float | None:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return None
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(expected, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_pct = np.maximum(expected_counts / expected.size, 1e-6)
    actual_pct = np.maximum(actual_counts / actual.size, 1e-6)
    return float(
        np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    )


def _categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    expected_counts = expected.value_counts(normalize=True)
    actual_counts = actual.value_counts(normalize=True)
    categories = sorted(set(expected_counts.index) | set(actual_counts.index))
    total = 0.0
    for category in categories:
        expected_pct = max(float(expected_counts.get(category, 0.0)), 1e-6)
        actual_pct = max(float(actual_counts.get(category, 0.0)), 1e-6)
        total += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(total)


def _sample_values(
    values: np.ndarray, *, rng: np.random.Generator, limit: int
) -> np.ndarray:
    flat = values.ravel()
    flat = flat[~np.isnan(flat)]
    if flat.size <= limit:
        return flat.astype("float64", copy=False)
    selected = rng.choice(flat.size, size=limit, replace=False)
    return flat[selected].astype("float64", copy=False)


def _flatline_share(frame: pd.DataFrame, columns: list[str]) -> float:
    current = frame[columns]
    previous = current.shift(1)
    comparable = current.notna() & previous.notna()
    denominator = int(comparable.to_numpy().sum())
    if denominator == 0:
        return 0.0
    flat = (current.eq(previous) & comparable).to_numpy().sum()
    return float(flat / denominator)


def _read_meter(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def _meter_summary(
    *,
    meter: str,
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    meta: pd.DataFrame,
    rng: np.random.Generator,
    sample_limit: int,
) -> dict[str, Any]:
    raw_columns = [column for column in raw.columns if column != "timestamp"]
    cleaned_columns = [column for column in cleaned.columns if column != "timestamp"]
    if raw_columns != cleaned_columns:
        raise ValueError(f"{meter} raw and cleaned building columns differ")

    overlap_by_building = meta.set_index("building_id")["is_gepiii_overlap"]
    bdg2_only_columns = [
        column
        for column in raw_columns
        if not bool(overlap_by_building.get(column, True))
    ]
    overlap_columns = [
        column for column in raw_columns if bool(overlap_by_building.get(column, False))
    ]

    raw_values = raw[raw_columns]
    cleaned_values = cleaned[cleaned_columns]
    raw_notna = raw_values.notna()
    cleaned_notna = cleaned_values.notna()
    value_cells = int(raw_values.shape[0] * raw_values.shape[1])
    raw_missing = int((~raw_notna).to_numpy().sum())
    cleaned_missing = int((~cleaned_notna).to_numpy().sum())
    raw_zero = int(raw_values.eq(0).to_numpy().sum())
    cleaned_zero = int(cleaned_values.eq(0).to_numpy().sum())
    raw_negative = int(raw_values.lt(0).to_numpy().sum())
    cleaned_negative = int(cleaned_values.lt(0).to_numpy().sum())
    removed = int((raw_notna & ~cleaned_notna).to_numpy().sum())
    filled = int((~raw_notna & cleaned_notna).to_numpy().sum())
    changed = int(
        (raw_notna & cleaned_notna & raw_values.ne(cleaned_values)).to_numpy().sum()
    )

    raw_missing_by_building = raw_values.isna().mean()
    sufficient = raw_missing_by_building.le(SUFFICIENT_OBS_MISSING_RATE)
    high_missing = raw_missing_by_building.gt(SUFFICIENT_OBS_MISSING_RATE)

    profiles = {}
    if meter in {"electricity", "chilledwater"}:
        row_mean = raw_values.mean(axis=1, skipna=True)
        profiles = {
            "hour_mean": row_mean.groupby(raw["timestamp"].dt.hour).mean().to_dict(),
            "month_mean": row_mean.groupby(raw["timestamp"].dt.month).mean().to_dict(),
        }

    raw_bdg2_sample = (
        _sample_values(
            raw[bdg2_only_columns].to_numpy(dtype="float64", copy=False),
            rng=rng,
            limit=sample_limit,
        )
        if bdg2_only_columns
        else np.array([], dtype="float64")
    )

    return {
        "meter": meter,
        "rows": int(len(raw)),
        "raw_building_columns": len(raw_columns),
        "bdg2_only_building_columns": len(bdg2_only_columns),
        "gepiii_overlap_building_columns": len(overlap_columns),
        "raw_null_rate": raw_missing / value_cells,
        "cleaned_null_rate": cleaned_missing / value_cells,
        "cleaned_minus_raw_null_rate": (cleaned_missing - raw_missing) / value_cells,
        "raw_zero_share": raw_zero / value_cells,
        "cleaned_zero_share": cleaned_zero / value_cells,
        "raw_negative_share": raw_negative / value_cells,
        "cleaned_negative_share": cleaned_negative / value_cells,
        "raw_flatline_share": _flatline_share(raw, raw_columns),
        "cleaned_flatline_share": _flatline_share(cleaned, cleaned_columns),
        "cleaned_delta": {
            "removed_share": removed / value_cells,
            "filled_share": filled / value_cells,
            "changed_observed_share": changed / value_cells,
            "zero_share_delta": (cleaned_zero - raw_zero) / value_cells,
            "negative_share_delta": (cleaned_negative - raw_negative) / value_cells,
        },
        "missingness_decomposition": {
            "building_level_meter_availability_absent": int(
                len(meta) - len(raw_columns)
            ),
            "timestamp_hours_expected": EXPECTED_HOURS,
            "timestamp_hours_observed_min": int(raw_values.notna().sum().min()),
            "timestamp_hours_observed_median": float(raw_values.notna().sum().median()),
            "timestamp_coverage_median": float(
                raw_values.notna().sum().median() / EXPECTED_HOURS
            ),
            "observation_missingness_raw": raw_missing / value_cells,
            "observation_missingness_cleaned": cleaned_missing / value_cells,
        },
        "bdg2_only_sufficient_obs": {
            "buildings_with_meter": len(bdg2_only_columns),
            "sufficient_obs_buildings": int(sufficient[bdg2_only_columns].sum())
            if bdg2_only_columns
            else 0,
            "high_missing_buildings": int(high_missing[bdg2_only_columns].sum())
            if bdg2_only_columns
            else 0,
            "median_missing_rate": float(
                raw_missing_by_building[bdg2_only_columns].median()
            )
            if bdg2_only_columns
            else None,
        },
        "overlap_sufficient_obs": {
            "buildings_with_meter": len(overlap_columns),
            "sufficient_obs_buildings": int(sufficient[overlap_columns].sum())
            if overlap_columns
            else 0,
            "high_missing_buildings": int(high_missing[overlap_columns].sum())
            if overlap_columns
            else 0,
            "median_missing_rate": float(
                raw_missing_by_building[overlap_columns].median()
            )
            if overlap_columns
            else None,
        },
        "raw_reading_stats": {
            "bdg2_only": _series_stats(pd.Series(raw_bdg2_sample)),
            "all_bdg2": _series_stats(pd.Series(raw_values.to_numpy().ravel())),
        },
        "profiles": profiles,
        "bdg2_only_reading_sample": raw_bdg2_sample,
    }


def _metadata_summary(meta: pd.DataFrame, gepiii_meta: pd.DataFrame) -> dict[str, Any]:
    bdg2_only = meta.loc[~meta["is_gepiii_overlap"]].copy()
    overlap = meta.loc[meta["is_gepiii_overlap"]].copy()
    gepiii_categories = gepiii_meta["primary_use"].map(_normalise_category)
    bdg2_only_categories = bdg2_only["primaryspaceusage"].map(_normalise_category)
    unseen_mask = ~bdg2_only_categories.isin(set(gepiii_categories))
    unseen_categories = sorted(
        category for category in bdg2_only_categories[unseen_mask].unique() if category
    )
    if unseen_mask.any() and not unseen_categories:
        unseen_categories = ["(missing/unmapped)"]

    meter_coverage = {}
    for meter in METER_TYPES:
        yes = meta[meter].fillna("").astype(str).str.lower().eq("yes")
        meter_coverage[meter] = {
            "all_buildings": int(yes.sum()),
            "bdg2_only": int((yes & ~meta["is_gepiii_overlap"]).sum()),
            "gepiii_overlap": int((yes & meta["is_gepiii_overlap"]).sum()),
        }

    return {
        "building_count": int(len(meta)),
        "bdg2_only_count": int((~meta["is_gepiii_overlap"]).sum()),
        "gepiii_overlap_count": int(meta["is_gepiii_overlap"].sum()),
        "site_distribution": {
            "bdg2_only": bdg2_only["site_id"].value_counts().sort_index().to_dict(),
            "gepiii_overlap": overlap["site_id"].value_counts().sort_index().to_dict(),
        },
        "primary_use_distribution": {
            "bdg2_only": bdg2_only["primaryspaceusage"].value_counts().to_dict(),
            "gepiii_overlap": overlap["primaryspaceusage"].value_counts().to_dict(),
            "gepiii": gepiii_meta["primary_use"].value_counts().to_dict(),
        },
        "primary_use_unseen_vs_gepiii": {
            "bdg2_only_unseen_buildings": int(unseen_mask.sum()),
            "bdg2_only_unseen_rate": float(unseen_mask.mean())
            if len(unseen_mask)
            else 0.0,
            "bdg2_only_unique_unseen": unseen_categories,
            "categorical_psi_bdg2_only_vs_gepiii": _categorical_psi(
                gepiii_categories, bdg2_only_categories
            ),
        },
        "square_feet": {
            "bdg2_only": _series_stats(bdg2_only["sqft"]),
            "gepiii_overlap": _series_stats(overlap["sqft"]),
            "gepiii": _series_stats(gepiii_meta["square_feet"]),
            "ks_bdg2_only_vs_gepiii": _ks_statistic(
                bdg2_only["sqft"].to_numpy(dtype="float64"),
                gepiii_meta["square_feet"].to_numpy(dtype="float64"),
            ),
            "psi_bdg2_only_vs_gepiii": _psi(
                gepiii_meta["square_feet"].to_numpy(dtype="float64"),
                bdg2_only["sqft"].to_numpy(dtype="float64"),
            ),
        },
        "year_built": {
            "bdg2_only": _series_stats(bdg2_only["yearbuilt"]),
            "gepiii_overlap": _series_stats(overlap["yearbuilt"]),
            "gepiii": _series_stats(gepiii_meta["year_built"]),
        },
        "floor_count": {
            "bdg2_only": _series_stats(bdg2_only["numberoffloors"]),
            "gepiii_overlap": _series_stats(overlap["numberoffloors"]),
        },
        "timezone_distribution": meta["timezone"].value_counts().sort_index().to_dict(),
        "meter_coverage": meter_coverage,
    }


def _gepiii_reading_sample(
    sample_limit: int, rng: np.random.Generator
) -> dict[str, Any]:
    frame = load_m3_frame(verbose=False)
    values = frame["meter_reading"].to_numpy(dtype="float64", copy=False)
    sample = _sample_values(values, rng=rng, limit=sample_limit)
    by_meter = {}
    for meter_id, group in frame.groupby("meter", observed=True):
        meter_values = group["meter_reading"].to_numpy(dtype="float64", copy=False)
        by_meter[str(int(meter_id))] = _series_stats(pd.Series(meter_values))
    return {
        "rows": int(len(frame)),
        "sample": sample,
        "stats": _series_stats(pd.Series(sample)),
        "zero_share_sample": float(np.mean(sample == 0)) if sample.size else None,
        "negative_share_sample": float(np.mean(sample < 0)) if sample.size else None,
        "by_meter": by_meter,
    }


def _write_figures(payload: dict[str, Any], asset_dir: Path) -> list[dict[str, Any]]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    meta = payload["raw_arrays"]["metadata"]
    bdg2_only_sqft = meta.loc[~meta["is_gepiii_overlap"], "sqft"].dropna().sort_values()
    gepiii_sqft = (
        payload["raw_arrays"]["gepiii_meta"]["square_feet"].dropna().sort_values()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bdg2_only_sqft, np.linspace(0, 1, len(bdg2_only_sqft)), label="BDG2-only")
    ax.plot(gepiii_sqft, np.linspace(0, 1, len(gepiii_sqft)), label="GEPIII")
    ax.set_xscale("log")
    ax.set_xlabel("square_feet")
    ax.set_ylabel("ECDF")
    ax.legend()
    ax.set_title("Square feet distribution")
    path = asset_dir / "square-feet-ecdf.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    figures.append(
        {"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size}
    )

    bdg2_sample = payload["raw_arrays"]["bdg2_only_reading_sample"]
    gepiii_sample = payload["raw_arrays"]["gepiii_reading_sample"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        np.log1p(np.clip(gepiii_sample, a_min=0, a_max=None)),
        bins=60,
        alpha=0.55,
        label="GEPIII",
    )
    ax.hist(
        np.log1p(np.clip(bdg2_sample, a_min=0, a_max=None)),
        bins=60,
        alpha=0.55,
        label="BDG2-only",
    )
    ax.set_xlabel("log1p(non-negative meter_reading)")
    ax.set_ylabel("sampled cells")
    ax.legend()
    ax.set_title("Meter reading sample distribution")
    path = asset_dir / "meter-reading-hist.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    figures.append(
        {"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size}
    )
    return figures


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join("" if value is None else str(value) for value in row)
            + " |"
        )
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _write_report(payload: dict[str, Any], path: Path) -> None:
    meta = payload["metadata"]
    meters = payload["meters"]
    ood = payload["ood"]
    figures = payload["figures"]
    chilledwater = next(item for item in meters if item["meter"] == "chilledwater")

    meter_rows = [
        [
            item["meter"],
            item["raw_building_columns"],
            item["bdg2_only_building_columns"],
            _fmt(item["raw_null_rate"]),
            _fmt(item["cleaned_null_rate"]),
            _fmt(item["raw_zero_share"]),
            _fmt(item["raw_negative_share"]),
            _fmt(item["raw_flatline_share"]),
        ]
        for item in meters
    ]
    delta_rows = [
        [
            item["meter"],
            _fmt(item["cleaned_minus_raw_null_rate"]),
            _fmt(item["cleaned_delta"]["removed_share"]),
            _fmt(item["cleaned_delta"]["filled_share"]),
            _fmt(item["cleaned_delta"]["changed_observed_share"]),
        ]
        for item in meters
    ]
    sufficient_rows = [
        [
            item["meter"],
            item["bdg2_only_sufficient_obs"]["buildings_with_meter"],
            item["bdg2_only_sufficient_obs"]["sufficient_obs_buildings"],
            item["bdg2_only_sufficient_obs"]["high_missing_buildings"],
            _fmt(item["bdg2_only_sufficient_obs"]["median_missing_rate"]),
        ]
        for item in meters
    ]
    coverage_rows = [
        [
            item["meter"],
            item["missingness_decomposition"][
                "building_level_meter_availability_absent"
            ],
            _fmt(item["missingness_decomposition"]["timestamp_coverage_median"]),
            _fmt(item["missingness_decomposition"]["observation_missingness_raw"]),
            _fmt(item["missingness_decomposition"]["observation_missingness_cleaned"]),
        ]
        for item in meters
    ]
    meter_coverage_rows = [
        [
            meter,
            values["all_buildings"],
            values["bdg2_only"],
            values["gepiii_overlap"],
        ]
        for meter, values in meta["meter_coverage"].items()
    ]
    primary_use_unseen_rate = meta["primary_use_unseen_vs_gepiii"][
        "bdg2_only_unseen_rate"
    ]
    ood_rows = [
        [
            "square_feet",
            _fmt(ood["square_feet"]["ks_bdg2_only_vs_gepiii"]),
            _fmt(ood["square_feet"]["psi_bdg2_only_vs_gepiii"]),
            "BDG2-only vs GEPIII metadata",
        ],
        [
            "meter_reading",
            _fmt(ood["meter_reading"]["ks_bdg2_only_vs_gepiii"]),
            _fmt(ood["meter_reading"]["psi_bdg2_only_vs_gepiii"]),
            "sampled raw BDG2-only cells vs GEPIII `load_m3_frame` cells",
        ],
        [
            "primary_use coverage",
            "n/a",
            _fmt(
                meta["primary_use_unseen_vs_gepiii"][
                    "categorical_psi_bdg2_only_vs_gepiii"
                ]
            ),
            "categorical PSI; unseen/unmapped rate " + _fmt(primary_use_unseen_rate),
        ],
    ]

    text = f"""# BDG2 EDA Report

**Date**: {payload["generated_at"][:10]}
**Issue**: [#40](https://github.com/kuokuant-oss/lead-reproduction/issues/40)
**Plan**: [docs/plans/bdg2-eda-plan.md](../plans/bdg2-eda-plan.md)

## Scope And Guardrails

This is a read-only, pre-modeling EDA slice. It reads BDG2 from `data/raw/bdg2`
and reads GEPIII comparison data from frozen GEPIII sources (`load_m3_frame` and
`data/raw/m3/building_metadata.csv`). It does not build a model, create scores,
fabricate labels, report supervised BDG2 metrics, or make readiness/transfer
claims.

The report uses neutral data-quality terms: zero-reading share, negative-reading
share, flatline share, missingness, coverage, and distribution distance.

## Headline Findings

+ BDG2 has {meta["building_count"]} buildings: {meta["bdg2_only_count"]}
  BDG2-only and {meta["gepiii_overlap_count"]} GEPIII-overlap.
+ BDG2-only chilledwater is sparse at the building level:
  {chilledwater["bdg2_only_sufficient_obs"]["buildings_with_meter"]} BDG2-only
  buildings have chilledwater columns, but only
  {chilledwater["bdg2_only_sufficient_obs"]["sufficient_obs_buildings"]} meet the
  sufficient-observation rule (`missing_rate <= 0.50`). This reproduces the
  Phase E Step 4 pooled stop point from the data side.
+ The chilledwater underpowering root cause is mostly building availability plus
  per-building observation missingness: most BDG2-only buildings do not have a
  chilledwater meter, and the available BDG2-only chilledwater set still leaves
  {chilledwater["bdg2_only_sufficient_obs"]["high_missing_buildings"]}
  high-missing buildings.
+ OOD quantification gives comparable scalars: square_feet KS
  `{_fmt(ood["square_feet"]["ks_bdg2_only_vs_gepiii"])}`, meter_reading KS
  `{_fmt(ood["meter_reading"]["ks_bdg2_only_vs_gepiii"])}`, and primary_use
  categorical PSI
  `{_fmt(meta["primary_use_unseen_vs_gepiii"]["categorical_psi_bdg2_only_vs_gepiii"])}`.

## Per-Meter Structure

{_markdown_table(["Meter", "Buildings", "BDG2-only buildings", "Raw null", "Cleaned null", "Raw zero", "Raw negative", "Raw flatline"], meter_rows)}

## Missingness Decomposition

This table separates building-level meter availability from observation-level
missingness. `Absent buildings` means metadata buildings without a column in the
wide meter file.

{_markdown_table(["Meter", "Absent buildings", "Median timestamp coverage", "Raw observation missingness", "Cleaned observation missingness"], coverage_rows)}

## Cleaned-Vs-Raw Delta

{_markdown_table(["Meter", "Null-rate delta", "Raw present -> cleaned missing", "Raw missing -> cleaned present", "Changed observed cells"], delta_rows)}

## BDG2-Only Sufficient-Observation Counts

{_markdown_table(["Meter", "BDG2-only with meter", "Sufficient obs", "High missing", "Median missing rate"], sufficient_rows)}

## Metadata And Meter Coverage

{_markdown_table(["Meter", "All buildings marked yes", "BDG2-only", "GEPIII-overlap"], meter_coverage_rows)}

Primary-use unseen/unmapped rate for BDG2-only vs GEPIII is
`{_fmt(primary_use_unseen_rate)}`.
Unseen or unmapped normalized categories:
`{", ".join(meta["primary_use_unseen_vs_gepiii"]["bdg2_only_unique_unseen"]) or "none"}`.

Square-feet medians:

+ BDG2-only: `{_fmt(meta["square_feet"]["bdg2_only"].get("median"))}`.
+ GEPIII-overlap: `{_fmt(meta["square_feet"]["gepiii_overlap"].get("median"))}`.
+ GEPIII: `{_fmt(meta["square_feet"]["gepiii"].get("median"))}`.

## BDG2-Only Vs GEPIII-Overlap Main Line

+ BDG2-only buildings are concentrated in a smaller set of sites, especially
  Lamb, Panther, Rat, and Swan in the local archive.
+ Meter availability differs sharply by meter. Electricity is broadest; solar
  and irrigation remain narrow. Chilledwater has enough overlap buildings for a
  bridge baseline but not enough BDG2-only sufficient-observation buildings for
  the prior Step 4 frame.
+ Cleaned files increase null rates for every meter in this archive, consistent
  with the Stage 0 inventory. That is a data-quality delta, not a label.

## BDG2 Vs GEPIII OOD Quantification

{_markdown_table(["Feature", "KS", "PSI", "Basis"], ood_rows)}

Figures:

+ ![Square feet ECDF](../assets/bdg2-eda/square-feet-ecdf.png)
+ ![Meter reading histogram](../assets/bdg2-eda/meter-reading-hist.png)

Figure sizes:

{_markdown_table(["Figure", "Bytes"], [[item["path"], item["size_bytes"]] for item in figures])}

## Temporal Profiles

The provenance JSON includes hour/month mean profiles for representative
electricity and chilledwater raw readings. These are descriptive profiles only;
they are not model features, scores, or readiness evidence.

## Provenance

+ Machine-readable summary: `data/processed/bdg2_eda.json` (gitignored shard).
+ Script: `scripts/run_bdg2_eda.py`.
+ BDG2 source: `data/raw/bdg2`.
+ GEPIII comparison source: `load_m3_frame(verbose=False)` and
  `data/raw/m3/building_metadata.csv`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_handoff(payload: dict[str, Any], path: Path) -> None:
    chilledwater = next(
        item for item in payload["meters"] if item["meter"] == "chilledwater"
    )
    ood = payload["ood"]
    text = f"""# Handoff: BDG2 pre-modeling EDA

**Date**: {payload["generated_at"][:10]}
**Issue**: [#40](https://github.com/kuokuant-oss/lead-reproduction/issues/40)

## Completed

+ Registered and executed `docs/plans/bdg2-eda-plan.md`.
+ Added `scripts/run_bdg2_eda.py` as a read-only EDA runner.
+ Wrote `docs/reports/bdg2-eda.md`.
+ Wrote small figures under `docs/assets/bdg2-eda/`.
+ Wrote ignored provenance JSON at `data/processed/bdg2_eda.json`.

## Guardrails Preserved

+ No modeling.
+ No score generation.
+ No fabricated labels.
+ No supervised BDG2 metrics.
+ No readiness or transfer claim.
+ No `src/lead` changes and no M3 numeric-line changes.

## Key Numbers

+ BDG2-only buildings: {payload["metadata"]["bdg2_only_count"]}.
+ GEPIII-overlap buildings: {payload["metadata"]["gepiii_overlap_count"]}.
+ Chilledwater BDG2-only buildings with meter:
  {chilledwater["bdg2_only_sufficient_obs"]["buildings_with_meter"]}.
+ Chilledwater BDG2-only sufficient-observation buildings:
  {chilledwater["bdg2_only_sufficient_obs"]["sufficient_obs_buildings"]}.
+ Chilledwater BDG2-only high-missing buildings:
  {chilledwater["bdg2_only_sufficient_obs"]["high_missing_buildings"]}.
+ Square_feet KS BDG2-only vs GEPIII:
  {_fmt(ood["square_feet"]["ks_bdg2_only_vs_gepiii"])}.
+ Meter_reading KS BDG2-only vs GEPIII:
  {_fmt(ood["meter_reading"]["ks_bdg2_only_vs_gepiii"])}.
+ Primary_use categorical PSI BDG2-only vs GEPIII:
  {_fmt(payload["metadata"]["primary_use_unseen_vs_gepiii"]["categorical_psi_bdg2_only_vs_gepiii"])}.

## Next Review Point

Stop here for human review. Do not proceed into modeling, full transfer,
alternate meter selection, or a new milestone until the EDA report is reviewed.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def run_eda(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    meta = pd.read_csv(args.bdg2_dir / "metadata.csv")
    meta["is_gepiii_overlap"] = _non_empty(meta["building_id_kaggle"])
    gepiii_meta = pd.read_csv(M3_DIR / "building_metadata.csv")

    metadata = _metadata_summary(meta, gepiii_meta)
    meters = []
    bdg2_samples = []
    for meter in METER_TYPES:
        raw = _read_meter(args.bdg2_dir / f"{meter}.csv")
        cleaned = _read_meter(args.bdg2_dir / f"{meter}_cleaned.csv")
        summary = _meter_summary(
            meter=meter,
            raw=raw,
            cleaned=cleaned,
            meta=meta,
            rng=rng,
            sample_limit=args.sample_per_meter,
        )
        bdg2_samples.append(summary.pop("bdg2_only_reading_sample"))
        meters.append(summary)

    bdg2_sample = (
        np.concatenate([sample for sample in bdg2_samples if sample.size])
        if any(sample.size for sample in bdg2_samples)
        else np.array([], dtype="float64")
    )
    gepiii_readings = _gepiii_reading_sample(args.gepiii_sample_size, rng)
    meter_ks = _ks_statistic(bdg2_sample, gepiii_readings["sample"])
    gepiii_reading_sample = gepiii_readings["sample"]
    meter_psi = _psi(gepiii_reading_sample, bdg2_sample)

    payload: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "issue": 40,
        "guardrails": {
            "read_only": True,
            "modeling": False,
            "score_generation": False,
            "label_creation": False,
            "readiness_claim": False,
        },
        "sources": {
            "bdg2_dir": str(args.bdg2_dir),
            "gepiii_metadata": str(M3_DIR / "building_metadata.csv"),
            "gepiii_loader": "load_m3_frame(verbose=False)",
        },
        "metadata": metadata,
        "meters": meters,
        "gepiii": {
            "load_m3_frame_rows": gepiii_readings["rows"],
            "meter_reading_sample_stats": gepiii_readings["stats"],
            "meter_reading_zero_share_sample": gepiii_readings["zero_share_sample"],
            "meter_reading_negative_share_sample": gepiii_readings[
                "negative_share_sample"
            ],
            "meter_reading_by_meter": gepiii_readings["by_meter"],
        },
        "ood": {
            "square_feet": metadata["square_feet"],
            "meter_reading": {
                "bdg2_only_sample_stats": _series_stats(pd.Series(bdg2_sample)),
                "gepiii_sample_stats": gepiii_readings["stats"],
                "bdg2_only_zero_share_sample": float(np.mean(bdg2_sample == 0))
                if bdg2_sample.size
                else None,
                "bdg2_only_negative_share_sample": float(np.mean(bdg2_sample < 0))
                if bdg2_sample.size
                else None,
                "gepiii_zero_share_sample": gepiii_readings["zero_share_sample"],
                "gepiii_negative_share_sample": gepiii_readings[
                    "negative_share_sample"
                ],
                "ks_bdg2_only_vs_gepiii": meter_ks,
                "psi_bdg2_only_vs_gepiii": meter_psi,
            },
            "primary_use": metadata["primary_use_unseen_vs_gepiii"],
        },
        "raw_arrays": {
            "metadata": meta,
            "gepiii_meta": gepiii_meta,
            "bdg2_only_reading_sample": bdg2_sample,
            "gepiii_reading_sample": gepiii_reading_sample,
        },
    }
    payload["figures"] = _write_figures(payload, args.asset_dir)
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.DataFrame):
        raise TypeError("DataFrame should not be serialized directly")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdg2-dir", type=Path, default=BDG2_DIR)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    parser.add_argument("--handoff-out", type=Path, default=HANDOFF_OUT)
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-per-meter", type=int, default=80_000)
    parser.add_argument("--gepiii-sample-size", type=int, default=400_000)
    args = parser.parse_args()

    payload = run_eda(args)
    raw_arrays = payload.pop("raw_arrays")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    payload["raw_arrays"] = raw_arrays
    _write_report(payload, args.report_out)
    _write_handoff(payload, args.handoff_out)
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.report_out}")
    print(f"Wrote {args.handoff_out}")
    for figure in payload["figures"]:
        print(f"Wrote {figure['path']} ({figure['size_bytes']} bytes)")


if __name__ == "__main__":
    main()
