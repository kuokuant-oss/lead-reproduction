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
SUFFICIENCY_SENSITIVITY_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60)
FLATLINE_MIN_RUN_LENGTH = 2
FLATLINE_INCLUDES_ZERO_RUNS = True
FLATLINE_MISSING_BREAKS_RUN = True
FLATLINE_EQUALITY = "exact"
FLATLINE_DENOMINATOR = "adjacent non-missing building-meter-hour comparisons"
FLATLINE_AGGREGATION = "cell-weighted adjacent comparisons"
GEPIII_METER_CODES = {
    "electricity": 0,
    "chilledwater": 1,
    "steam": 2,
    "hotwater": 3,
}
TOP_SITE_NAMES = ("Lamb", "Panther", "Rat", "Swan")


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


def _distribution_distance_variants(
    *, bdg2_sample: np.ndarray, gepiii_sample: np.ndarray
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    definitions = {
        "raw_zero_included": (bdg2_sample, gepiii_sample),
        "log1p_zero_included": (
            np.log1p(np.clip(bdg2_sample, a_min=0, a_max=None)),
            np.log1p(np.clip(gepiii_sample, a_min=0, a_max=None)),
        ),
        "log1p_zero_excluded": (
            np.log1p(bdg2_sample[bdg2_sample > 0]),
            np.log1p(gepiii_sample[gepiii_sample > 0]),
        ),
    }
    for name, (bdg2_values, gepiii_values) in definitions.items():
        variants[name] = {
            "ks_bdg2_only_vs_gepiii": _ks_statistic(bdg2_values, gepiii_values),
            "psi_bdg2_only_vs_gepiii": _psi(gepiii_values, bdg2_values),
            "bdg2_sample_count": int(np.asarray(bdg2_values).size),
            "gepiii_sample_count": int(np.asarray(gepiii_values).size),
        }
    return variants


def _flatline_share(frame: pd.DataFrame, columns: list[str]) -> float:
    current = frame[columns]
    previous = current.shift(1)
    comparable = current.notna() & previous.notna()
    denominator = int(comparable.to_numpy().sum())
    if denominator == 0:
        return 0.0
    flat = (current.eq(previous) & comparable).to_numpy().sum()
    return float(flat / denominator)


def _flatline_definition() -> dict[str, Any]:
    return {
        "min_run_length": FLATLINE_MIN_RUN_LENGTH,
        "zero_runs_included": FLATLINE_INCLUDES_ZERO_RUNS,
        "missing_breaks_run": FLATLINE_MISSING_BREAKS_RUN,
        "equality": FLATLINE_EQUALITY,
        "denominator": FLATLINE_DENOMINATOR,
        "aggregation": FLATLINE_AGGREGATION,
        "reported_share": (
            "For each meter, share of adjacent non-missing building-hour cells "
            "whose reading exactly equals the prior hour for the same building."
        ),
        "zero_reading_share_reported_separately": True,
    }


def _threshold_sensitivity(missing_by_building: pd.Series) -> dict[str, int]:
    return {
        f"{threshold:.2f}": int(missing_by_building.le(threshold).sum())
        for threshold in SUFFICIENCY_SENSITIVITY_THRESHOLDS
    }


def _metadata_field_summary(series: pd.Series, *, is_numeric: bool) -> dict[str, Any]:
    non_null_rate = (
        float(_non_empty(series).mean())
        if not is_numeric
        else float(pd.to_numeric(series, errors="coerce").notna().mean())
    )
    if is_numeric:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        return {
            "non_null_rate": non_null_rate,
            "median": float(numeric.median()) if not numeric.empty else None,
        }
    clean = series.dropna().astype(str).str.strip()
    clean = clean.loc[clean.ne("")]
    top = clean.value_counts().head(1)
    return {
        "non_null_rate": non_null_rate,
        "top_value": str(top.index[0]) if not top.empty else None,
        "top_count": int(top.iloc[0]) if not top.empty else 0,
    }


def _top_site_contribution(
    *, meta: pd.DataFrame, meter_summaries: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    bdg2_only = meta.loc[~meta["is_gepiii_overlap"]].copy()
    chilledwater = meter_summaries["chilledwater"]
    raw_missing_by_building = chilledwater["raw_missing_by_building"]
    rows = []
    for site in TOP_SITE_NAMES:
        site_buildings = bdg2_only.loc[bdg2_only["site_id"].eq(site), "building_id"]
        chilledwater_missing = raw_missing_by_building.reindex(site_buildings).dropna()
        rows.append(
            {
                "site_id": site,
                "bdg2_only_buildings": int(site_buildings.size),
                "bdg2_only_chilledwater_columns": int(chilledwater_missing.size),
                "bdg2_only_chilledwater_sufficient_obs": int(
                    chilledwater_missing.le(SUFFICIENT_OBS_MISSING_RATE).sum()
                ),
            }
        )
    return rows


def _temporal_profile_summaries(meters: list[dict[str, Any]]) -> dict[str, str]:
    summaries: dict[str, str] = {}
    for meter in meters:
        if meter["meter"] not in {"electricity", "chilledwater"}:
            continue
        profiles = meter.get("profiles", {})
        hour_mean = profiles.get("hour_mean", {})
        month_mean = profiles.get("month_mean", {})
        if not hour_mean or not month_mean:
            continue
        hour_series = pd.Series(
            {int(hour): float(value) for hour, value in hour_mean.items()}
        )
        month_series = pd.Series(
            {int(month): float(value) for month, value in month_mean.items()}
        )
        summaries[meter["meter"]] = (
            f"{meter['meter']} has its highest mean reading around hour "
            f"{int(hour_series.idxmax())} and lowest around hour "
            f"{int(hour_series.idxmin())}; by month it peaks in "
            f"{int(month_series.idxmax())} and is lowest in "
            f"{int(month_series.idxmin())}"
        )
    return summaries


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
    bdg2_only_missing = raw_missing_by_building[bdg2_only_columns]

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
            "threshold_sensitivity": _threshold_sensitivity(bdg2_only_missing)
            if bdg2_only_columns
            else {
                f"{threshold:.2f}": 0
                for threshold in SUFFICIENCY_SENSITIVITY_THRESHOLDS
            },
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
        "raw_missing_by_building": raw_missing_by_building,
    }


def _metadata_summary(meta: pd.DataFrame, gepiii_meta: pd.DataFrame) -> dict[str, Any]:
    bdg2_only = meta.loc[~meta["is_gepiii_overlap"]].copy()
    overlap = meta.loc[meta["is_gepiii_overlap"]].copy()
    metadata_fields = {
        "primary_use": ("primaryspaceusage", False, "headline_distance"),
        "square_feet": ("sqft", True, "headline_distance"),
        "sqm": ("sqm", True, "descriptive_only"),
        "year_built": ("yearbuilt", True, "descriptive_only"),
        "floor_count": ("numberoffloors", True, "descriptive_only"),
        "site_id": ("site_id", False, "descriptive_only"),
        "timezone": ("timezone", False, "descriptive_only"),
    }
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

    completeness = {}
    for label, (column, is_numeric, usage) in metadata_fields.items():
        completeness[label] = {
            "source_column": column,
            "usage": usage,
            "bdg2": _metadata_field_summary(meta[column], is_numeric=is_numeric),
            "bdg2_only": _metadata_field_summary(
                bdg2_only[column], is_numeric=is_numeric
            ),
            "gepiii_overlap": _metadata_field_summary(
                overlap[column], is_numeric=is_numeric
            ),
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
        "completeness": completeness,
        "meter_coverage": meter_coverage,
    }


def _gepiii_reading_sample(
    sample_limit: int, rng: np.random.Generator
) -> dict[str, Any]:
    frame = load_m3_frame(verbose=False)
    values = frame["meter_reading"].to_numpy(dtype="float64", copy=False)
    sample = _sample_values(values, rng=rng, limit=sample_limit)
    by_meter = {}
    samples_by_meter = {}
    for meter_id, group in frame.groupby("meter", observed=True):
        meter_values = group["meter_reading"].to_numpy(dtype="float64", copy=False)
        meter_sample = _sample_values(meter_values, rng=rng, limit=sample_limit)
        meter_key = str(int(meter_id))
        by_meter[meter_key] = _series_stats(pd.Series(meter_values))
        samples_by_meter[meter_key] = meter_sample
    return {
        "rows": int(len(frame)),
        "sample": sample,
        "stats": _series_stats(pd.Series(sample)),
        "zero_share_sample": float(np.mean(sample == 0)) if sample.size else None,
        "negative_share_sample": float(np.mean(sample < 0)) if sample.size else None,
        "by_meter": by_meter,
        "samples_by_meter": samples_by_meter,
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
    flatline = payload["definitions"]["flatline"]
    chilledwater = next(item for item in meters if item["meter"] == "chilledwater")
    building_count = f"{meta['building_count']:,}"
    bdg2_only_count = f"{meta['bdg2_only_count']:,}"
    gepiii_overlap_count = f"{meta['gepiii_overlap_count']:,}"
    chilledwater_bdg2_only = chilledwater["bdg2_only_sufficient_obs"][
        "buildings_with_meter"
    ]
    chilledwater_sufficient = chilledwater["bdg2_only_sufficient_obs"][
        "sufficient_obs_buildings"
    ]
    chilledwater_high_missing = chilledwater["bdg2_only_sufficient_obs"][
        "high_missing_buildings"
    ]

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
    metadata_rows = []
    for label, values in meta["completeness"].items():
        bdg2_summary = values["bdg2"]
        only_summary = values["bdg2_only"]
        overlap_summary = values["gepiii_overlap"]
        if "median" in bdg2_summary:
            bdg2_detail = f"median {_fmt(bdg2_summary['median'])}"
            only_detail = f"median {_fmt(only_summary['median'])}"
            overlap_detail = f"median {_fmt(overlap_summary['median'])}"
        else:
            bdg2_detail = (
                f"top {bdg2_summary['top_value']} ({bdg2_summary['top_count']})"
            )
            only_detail = (
                f"top {only_summary['top_value']} ({only_summary['top_count']})"
            )
            overlap_detail = (
                f"top {overlap_summary['top_value']} ({overlap_summary['top_count']})"
            )
        metadata_rows.append(
            [
                label,
                values["source_column"],
                values["usage"],
                _fmt(bdg2_summary["non_null_rate"]),
                bdg2_detail,
                only_detail,
                overlap_detail,
            ]
        )
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
    chilledwater_threshold_rows = [
        [threshold, count]
        for threshold, count in chilledwater["bdg2_only_sufficient_obs"][
            "threshold_sensitivity"
        ].items()
    ]
    threshold_counts = [
        count
        for _, count in chilledwater["bdg2_only_sufficient_obs"][
            "threshold_sensitivity"
        ].items()
    ]
    threshold_interpretation = (
        "The verdict is robust across these thresholds because the count remains\n"
        "well below a powered frame."
        if max(threshold_counts) < 10
        else "The verdict is gate-sensitive because relaxed thresholds sharply\n"
        "increase the eligible building count."
    )
    top_site_rows = [
        [
            item["site_id"],
            item["bdg2_only_buildings"],
            item["bdg2_only_chilledwater_columns"],
            item["bdg2_only_chilledwater_sufficient_obs"],
        ]
        for item in payload["bdg2_only_top_site_contribution"]
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
    per_meter_rows = []
    for meter, values in ood["meter_reading"]["per_meter_distances"].items():
        for variant, distances in values["variants"].items():
            per_meter_rows.append(
                [
                    meter,
                    variant,
                    _fmt(distances["ks_bdg2_only_vs_gepiii"]),
                    _fmt(distances["psi_bdg2_only_vs_gepiii"]),
                    _fmt(values["bdg2_only_zero_share_sample"]),
                    _fmt(values["gepiii_zero_share_sample"]),
                ]
            )
    temporal_summaries = payload["temporal_profile_summaries"]

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

+ BDG2 contains {building_count} buildings, but meter availability is
  highly uneven across meters.
+ Electricity is broadly available; chilledwater, steam, hotwater, gas, water,
  irrigation, and solar have much narrower building coverage.
+ Several meters show high zero-reading or flatline shares, especially
  irrigation, water, gas, and hotwater. These can reflect operational
  off-periods described by Miller et al. 2020, not data faults.
+ Cleaned files increase null rates for every meter, reflecting BDG2's own
  outlier/zero removal rules described by Miller et al. 2020: Twitter
  AnomalyDetection outlier removal, removal of zero-reading runs longer than 24
  hours, and removal of electricity zeros. This is a data-quality delta, not a
  label.
+ For BDG2-only buildings, chilledwater is especially underpowered: of the
  {bdg2_only_count} BDG2-only buildings, {chilledwater_bdg2_only} have
  chilledwater columns but only {chilledwater_sufficient} meet the
  sufficient-observation rule (`missing_rate <= 0.50`); this reproduces the
  Phase E Step 4 stop point from the data side.
+ The GEPIII comparison is used only to contextualize coverage and distribution
  differences, not to make modeling or transfer-readiness claims.

## Dataset Provenance And Cleaning

The BDG2 data descriptor is tracked in
[docs/reference/papers/bdg2-miller-2020.md](../reference/papers/bdg2-miller-2020.md).
The PDF is kept locally at `docs/reference/papers/bdg2-miller-2020.pdf` and is
gitignored because it exceeds the repo's 500 KB large-file gate.

Miller et al. 2020 describe the raw release pipeline as unit conversion,
negative readings set to missing, removal of meters with more than 50% negative
readings, removal of meters with more than 100 consecutive days of missing
readings, log plus three-standard-deviation outlier removal, and four-decimal
rounding. The cleaned release then applies additional Twitter AnomalyDetection
outlier removal, removes zero-reading runs longer than 24 hours, and removes
electricity zeros. These release-level rules explain why raw negative-reading
share is zero in this EDA and why cleaned null rates are higher than raw null
rates for every meter.

## BDG2 Data-Quality Inventory

### Per-Meter Structure

{_markdown_table(["Meter", "Buildings", "BDG2-only buildings", "Raw null", "Cleaned null", "Raw zero", "Raw negative", "Raw flatline"], meter_rows)}

### Flatline Definition

Flatline share is reported with an explicit rule: minimum run length
`{flatline["min_run_length"]}`; zero-reading runs are
`{"included" if flatline["zero_runs_included"] else "excluded"}`; missing values
break runs; equality is `{flatline["equality"]}`. The denominator is
{flatline["denominator"]}; aggregation is {flatline["aggregation"]}.
Zero-reading share is reported separately, so zero prevalence is not hidden
inside the flatline statistic.

### Missingness Decomposition

This table separates building-level meter availability from observation-level
missingness. `Absent buildings` means metadata buildings without a column in the
wide meter file.

{_markdown_table(["Meter", "Absent buildings", "Median timestamp coverage", "Raw observation missingness", "Cleaned observation missingness"], coverage_rows)}

### Cleaned-Vs-Raw Delta

{_markdown_table(["Meter", "Null-rate delta", "Raw present -> cleaned missing", "Raw missing -> cleaned present", "Changed observed cells"], delta_rows)}

For every meter, raw-to-cleaned missing is positive and raw missing-to-cleaned
present is zero; consistent with cleaned files removing additional observations
rather than filling raw gaps.

### Metadata Completeness

{_markdown_table(["Field", "Source column", "Usage", "BDG2 non-null", "BDG2 summary", "BDG2-only summary", "GEPIII-overlap summary"], metadata_rows)}

## BDG2-Only Sufficiency

BDG2 has {bdg2_only_count} BDG2-only buildings and
{gepiii_overlap_count} GEPIII-overlap buildings. The table below
summarizes BDG2-only meter availability and the sufficient-observation split.
For chilledwater, {chilledwater_bdg2_only} BDG2-only buildings have meter
columns, {chilledwater_sufficient} meet the `missing_rate <= 0.50` rule, and
{chilledwater_high_missing} are high-missing. This is the data-side reason the
Phase E Step 4 chilledwater frame remains underpowered.

{_markdown_table(["Meter", "BDG2-only with meter", "Sufficient obs", "High missing", "Median missing rate"], sufficient_rows)}

### Chilledwater Sufficiency Threshold Sensitivity

{_markdown_table(["Missing-rate threshold", "Sufficient BDG2-only chilledwater buildings"], chilledwater_threshold_rows)}

{threshold_interpretation}

### BDG2-Only Top-Site Contribution

{_markdown_table(["Site", "BDG2-only buildings", "BDG2-only chilledwater columns", "BDG2-only chilledwater sufficient obs"], top_site_rows)}

## GEPIII Comparison As Context

The GEPIII comparison is a diagnostic lens for coverage and distribution
differences. It is not a modeling result, not a transfer result, and not a
readiness claim.

### Meter Coverage Context

{_markdown_table(["Meter", "All buildings marked yes", "BDG2-only", "GEPIII-overlap"], meter_coverage_rows)}

Primary-use unseen/unmapped rate for BDG2-only vs GEPIII is
`{_fmt(primary_use_unseen_rate)}`.
Unseen or unmapped normalized categories:
`{", ".join(meta["primary_use_unseen_vs_gepiii"]["bdg2_only_unique_unseen"]) or "none"}`.

Square-feet medians:

+ BDG2-only: `{_fmt(meta["square_feet"]["bdg2_only"].get("median"))}`.
+ GEPIII-overlap: `{_fmt(meta["square_feet"]["gepiii_overlap"].get("median"))}`.
+ GEPIII: `{_fmt(meta["square_feet"]["gepiii"].get("median"))}`.

BDG2-only buildings are concentrated in a smaller set of sites, especially
Lamb, Panther, Rat, and Swan in the local archive. Meter availability differs
sharply by meter. Electricity is broadest; solar and irrigation remain narrow.
Chilledwater has enough overlap buildings for a bridge baseline but not enough
BDG2-only sufficient-observation buildings for the prior Step 4 frame.

### Reference Distribution Distances

{_markdown_table(["Feature", "KS", "PSI", "Basis"], ood_rows)}

The meter_reading distance compares sampled BDG2 raw cells against GEPIII
Kaggle-release cells via `load_m3_frame`. Part of this distance reflects known
release-level differences described by Miller et al. 2020: meter-type mix,
zero inflation, site composition, Kaggle unit-conversion errors, and
UTC-vs-local weather timestamps that BDG2 raw/cleaned fixed but the Kaggle
subset left as-is. It should therefore not be read as building behavior alone;
future refinement should prioritize per-meter, log1p, and zero-excluded
distances.

### Per-Meter Reference Distances

{_markdown_table(["Meter", "Variant", "KS", "PSI", "BDG2-only zero share", "GEPIII zero share"], per_meter_rows)}

Figures:

+ ![Square feet ECDF](../assets/bdg2-eda/square-feet-ecdf.png)
+ ![Meter reading histogram](../assets/bdg2-eda/meter-reading-hist.png)

Figure sizes:

{_markdown_table(["Figure", "Bytes"], [[item["path"], item["size_bytes"]] for item in figures])}

## Temporal Profiles

The provenance JSON includes hour/month mean profiles for representative
electricity and chilledwater raw readings. These are descriptive profiles only;
they are not model features, scores, or readiness evidence.

+ {temporal_summaries.get("electricity", "electricity profile unavailable")}.
+ {temporal_summaries.get("chilledwater", "chilledwater profile unavailable")}.

## Methodological Caveats And Review Notes

+ Released-raw negative-reading share is measured on the released BDG2 raw
  files. It does not imply the original site-source feeds never contained
  negative readings: Miller et al. 2020 describe setting negative readings to
  missing and removing meters with more than 50% negative readings during
  release processing.
+ Cleaned null rate above raw null rate is a data-quality delta, not a label.
  Miller et al. 2020 describe the cleaned files as applying Twitter
  AnomalyDetection outlier removal, removing zero-reading runs longer than
  24 hours, and removing electricity zeros.
+ Pooled meter_reading KS/PSI is a headline diagnostic only. It mixes meter-type
  composition, zero inflation, site composition, and known BDG2-vs-GEPIII
  release-regime differences; it should not be interpreted as a pure building
  behavior distance.

## Provenance

+ Machine-readable summary: `data/processed/bdg2_eda.json` (gitignored shard).
+ Script: `scripts/run_bdg2_eda.py`.
+ BDG2 source: `data/raw/bdg2`.
+ BDG2 paper reference: `docs/reference/papers/bdg2-miller-2020.md`.
+ GEPIII comparison source: `load_m3_frame(verbose=False)` and
  `data/raw/m3/building_metadata.csv`.
+ Distance scalar sampling: per-meter BDG2 sample
  `{payload["sampling"]["sample_per_meter"]}`, GEPIII sample
  `{payload["sampling"]["gepiii_sample_size"]}`, seed
  `{payload["sampling"]["seed"]}`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_handoff(payload: dict[str, Any], path: Path) -> None:
    chilledwater = next(
        item for item in payload["meters"] if item["meter"] == "chilledwater"
    )
    ood = payload["ood"]
    threshold_text = ", ".join(
        f"{threshold}: {count}"
        for threshold, count in chilledwater["bdg2_only_sufficient_obs"][
            "threshold_sensitivity"
        ].items()
    )
    chilledwater_distances = ood["meter_reading"]["per_meter_distances"][
        "chilledwater"
    ]["variants"]
    top_site_rows = [
        [
            item["site_id"],
            item["bdg2_only_buildings"],
            item["bdg2_only_chilledwater_columns"],
            item["bdg2_only_chilledwater_sufficient_obs"],
        ]
        for item in payload["bdg2_only_top_site_contribution"]
    ]
    text = f"""# Handoff: BDG2 pre-modeling EDA

**Date**: {payload["generated_at"][:10]}
**Issue**: [#40](https://github.com/kuokuant-oss/lead-reproduction/issues/40)

## Completed

+ Registered and executed `docs/plans/bdg2-eda-plan.md`.
+ Added `scripts/run_bdg2_eda.py` as a read-only EDA runner.
+ Wrote `docs/reports/bdg2-eda.md`.
+ Wrote small figures under `docs/assets/bdg2-eda/`.
+ Wrote ignored provenance JSON at `data/processed/bdg2_eda.json`.
+ Added the Miller et al. 2020 BDG2 data descriptor reference card and reframed
  the report around data-quality inventory, BDG2-only sufficiency, and reference
  distribution distances.
+ Applied the review refinement patch: explicit flatline definition,
  chilledwater threshold sensitivity, metadata completeness, BDG2-only top-site
  contribution, per-meter reference distances, temporal summaries, and
  methodological caveats.

## Guardrails Preserved

+ No modeling.
+ No score generation.
+ No fabricated labels.
+ No supervised BDG2 metrics.
+ No readiness or transfer claim.
+ No `src/lead` changes and no M3 numeric-line changes.
+ No distance-scalar recalculation logic changes.

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
+ Chilledwater threshold sensitivity:
  {threshold_text}.
+ Chilledwater per-meter raw-zero-included KS/PSI:
  {_fmt(chilledwater_distances["raw_zero_included"]["ks_bdg2_only_vs_gepiii"])}
  / {_fmt(chilledwater_distances["raw_zero_included"]["psi_bdg2_only_vs_gepiii"])}.
+ Chilledwater per-meter log1p-zero-included KS/PSI:
  {_fmt(chilledwater_distances["log1p_zero_included"]["ks_bdg2_only_vs_gepiii"])}
  / {_fmt(chilledwater_distances["log1p_zero_included"]["psi_bdg2_only_vs_gepiii"])}.
+ Chilledwater per-meter log1p-zero-excluded KS/PSI:
  {_fmt(chilledwater_distances["log1p_zero_excluded"]["ks_bdg2_only_vs_gepiii"])}
  / {_fmt(chilledwater_distances["log1p_zero_excluded"]["psi_bdg2_only_vs_gepiii"])}.
+ Distance scalar sampling:
  per-meter BDG2 sample {payload["sampling"]["sample_per_meter"]}, GEPIII sample
  {payload["sampling"]["gepiii_sample_size"]}, seed {payload["sampling"]["seed"]}.

BDG2-only top-site contribution:

{_markdown_table(["Site", "BDG2-only buildings", "Chilledwater columns", "Sufficient obs"], top_site_rows)}

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
    bdg2_samples_by_meter = {}
    meter_summaries_for_derived = {}
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
        bdg2_sample_for_meter = summary.pop("bdg2_only_reading_sample")
        bdg2_samples.append(bdg2_sample_for_meter)
        bdg2_samples_by_meter[meter] = bdg2_sample_for_meter
        meter_summaries_for_derived[meter] = {
            "raw_missing_by_building": summary.pop("raw_missing_by_building")
        }
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
    per_meter_distances = {}
    for meter, meter_code in GEPIII_METER_CODES.items():
        meter_bdg2_sample = bdg2_samples_by_meter[meter]
        meter_gepiii_sample = gepiii_readings["samples_by_meter"].get(
            str(meter_code), np.array([], dtype="float64")
        )
        per_meter_distances[meter] = {
            "bdg2_only_zero_share_sample": float(np.mean(meter_bdg2_sample == 0))
            if meter_bdg2_sample.size
            else None,
            "gepiii_zero_share_sample": float(np.mean(meter_gepiii_sample == 0))
            if meter_gepiii_sample.size
            else None,
            "variants": _distribution_distance_variants(
                bdg2_sample=meter_bdg2_sample,
                gepiii_sample=meter_gepiii_sample,
            ),
        }
    top_site_contribution = _top_site_contribution(
        meta=meta, meter_summaries=meter_summaries_for_derived
    )
    temporal_profile_summaries = _temporal_profile_summaries(meters)

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
            "bdg2_paper_reference": "docs/reference/papers/bdg2-miller-2020.md",
        },
        "sampling": {
            "seed": args.seed,
            "sample_per_meter": args.sample_per_meter,
            "gepiii_sample_size": args.gepiii_sample_size,
        },
        "definitions": {
            "flatline": _flatline_definition(),
            "sufficient_obs_missing_rate": SUFFICIENT_OBS_MISSING_RATE,
            "sufficiency_sensitivity_thresholds": list(
                SUFFICIENCY_SENSITIVITY_THRESHOLDS
            ),
        },
        "metadata": metadata,
        "meters": meters,
        "bdg2_only_top_site_contribution": top_site_contribution,
        "temporal_profile_summaries": temporal_profile_summaries,
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
                "per_meter_distances": per_meter_distances,
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
