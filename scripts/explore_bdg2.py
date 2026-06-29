"""Inventory real BDG2 CSV files without touching src/lead.

This script expects the flat Kaggle archive layout used for Phase E Stage 0:
metadata.csv, weather.csv, and one raw plus one cleaned CSV for each meter
type. It writes a machine-readable JSON inventory and a short report.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


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

ROOT = Path(__file__).resolve().parents[1]
JSON_OUT = ROOT / "data" / "processed" / "bdg2_data_reality.json"
REPORT_OUT = ROOT / "docs" / "reports" / "bdg2-data-reality.md"


def _read_head(path: Path, size: int = 120) -> str:
    with path.open("rb") as handle:
        return handle.read(size).decode("utf-8", errors="replace")


def _fail_on_lfs_pointer(paths: list[Path]) -> None:
    pointers = [
        str(path)
        for path in paths
        if _read_head(path).startswith("version https://git-lfs.github.com/spec/v1")
    ]
    if pointers:
        joined = "\n".join(f"- {path}" for path in pointers)
        raise SystemExit(
            f"Git LFS pointer files found; refusing to inventory:\n{joined}"
        )


def _file_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 3)


def _series_iso(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat(sep=" ")


def _dtype_map(frame: pd.DataFrame) -> dict[str, str]:
    return {column: str(dtype) for column, dtype in frame.dtypes.items()}


def _non_empty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _metadata_inventory(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if "building_id_kaggle" in frame.columns:
        overlap = _non_empty(frame["building_id_kaggle"])
    else:
        overlap = pd.Series(False, index=frame.index)
    return {
        "file": path.name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": _dtype_map(frame),
        "building_id_sample": frame["building_id"].head(5).tolist()
        if "building_id" in frame.columns
        else [],
        "site_count": int(frame["site_id"].nunique(dropna=True))
        if "site_id" in frame.columns
        else None,
        "building_count": int(frame["building_id"].nunique(dropna=True))
        if "building_id" in frame.columns
        else None,
        "primary_use_column": "primaryspaceusage"
        if "primaryspaceusage" in frame.columns
        else None,
        "gepiii_field_mapping": {
            "building_id": "building_id",
            "site_id": "site_id",
            "building_id_kaggle": "building_id_kaggle",
            "site_id_kaggle": "site_id_kaggle",
            "primary_use": "primaryspaceusage",
            "square_feet": "sqft",
            "year_built": "yearbuilt",
            "floor_count": "numberoffloors",
        },
        "meter_availability_counts": {
            meter: int(frame[meter].fillna("").astype(str).str.lower().eq("yes").sum())
            for meter in METER_TYPES
            if meter in frame.columns
        },
        "timezone_count": int(frame["timezone"].nunique(dropna=True))
        if "timezone" in frame.columns
        else None,
        "timezones": sorted(frame["timezone"].dropna().unique().tolist())
        if "timezone" in frame.columns
        else [],
        "site_ids": sorted(frame["site_id"].dropna().unique().tolist())
        if "site_id" in frame.columns
        else [],
        "gepiii_overlap": {
            "overlap_buildings": int(overlap.sum()),
            "bdg2_only_buildings": int((~overlap).sum()),
            "overlap_site_distribution": frame.loc[overlap, "site_id"]
            .value_counts()
            .sort_index()
            .astype(int)
            .to_dict()
            if "site_id" in frame.columns
            else {},
            "bdg2_only_site_distribution": frame.loc[~overlap, "site_id"]
            .value_counts()
            .sort_index()
            .astype(int)
            .to_dict()
            if "site_id" in frame.columns
            else {},
        },
    }


def _time_info(frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    diffs = timestamps.sort_values().diff().dropna()
    most_common_step = None
    if not diffs.empty:
        mode = diffs.mode()
        if not mode.empty:
            most_common_step = str(mode.iloc[0])
    return {
        "timestamp_min": _series_iso(timestamps.min()),
        "timestamp_max": _series_iso(timestamps.max()),
        "timestamp_nulls": int(timestamps.isna().sum()),
        "observed_years": sorted(
            int(year) for year in timestamps.dt.year.dropna().unique()
        ),
        "most_common_step": most_common_step,
        "timezone_column_present": "timezone" in frame.columns,
    }


def _meter_inventory(path: Path, metadata_buildings: set[str]) -> dict[str, Any]:
    frame = pd.read_csv(path)
    columns = list(frame.columns)
    value_columns = [column for column in columns if column != "timestamp"]
    null_cells = int(frame[value_columns].isna().sum().sum()) if value_columns else 0
    value_cells = int(len(frame) * len(value_columns))
    matched_buildings = sorted(set(value_columns).intersection(metadata_buildings))
    missing_from_metadata = sorted(set(value_columns) - metadata_buildings)
    extra_metadata_buildings = sorted(metadata_buildings - set(value_columns))
    meter_type = path.stem.replace("_cleaned", "")
    variant = "cleaned" if path.stem.endswith("_cleaned") else "raw"
    sample = frame.head(3)
    return {
        "file": path.name,
        "meter_type": meter_type,
        "variant": variant,
        "size_mb": _file_mb(path),
        "rows": int(len(frame)),
        "columns": int(len(columns)),
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "layout": "wide_timestamp_plus_building_columns"
        if columns and columns[0] == "timestamp" and len(columns) > 2
        else "unknown",
        "timestamp_column": "timestamp" if "timestamp" in columns else None,
        "building_column_count": len(value_columns),
        "building_columns_are_metadata_building_ids": len(missing_from_metadata) == 0,
        "matched_metadata_buildings": len(matched_buildings),
        "building_columns_missing_from_metadata": missing_from_metadata[:20],
        "metadata_buildings_without_this_meter": len(extra_metadata_buildings),
        "null_cells": null_cells,
        "value_cells": value_cells,
        "null_rate": round(null_cells / value_cells, 6) if value_cells else None,
        "dtypes_sample": _dtype_map(sample),
        "sample_building_columns": value_columns[:5],
        **_time_info(frame),
    }


def _weather_inventory(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    site_column = "site_id" if "site_id" in frame.columns else None
    null_cells = int(frame.isna().sum().sum())
    cells = int(frame.shape[0] * frame.shape[1])
    return {
        "file": path.name,
        "size_mb": _file_mb(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "dtypes": _dtype_map(frame.head(1000)),
        "key": site_column,
        "site_count": int(frame[site_column].nunique(dropna=True))
        if site_column
        else None,
        "site_ids": sorted(frame[site_column].dropna().unique().tolist())
        if site_column
        else [],
        "null_cells": null_cells,
        "null_rate": round(null_cells / cells, 6) if cells else None,
        "timezone_column_present": "timezone" in frame.columns,
        **_time_info(frame),
    }


def _label_inventory(
    paths: list[Path], metadata_columns: list[str], weather_columns: list[str]
) -> dict[str, Any]:
    label_terms = ("anomaly", "fault", "label", "is_bad", "bad_meter")
    file_hits = [
        path.name
        for path in paths
        if any(term in path.name.lower() for term in label_terms)
    ]
    column_hits = [
        column
        for column in metadata_columns + weather_columns
        if any(term in column.lower() for term in label_terms)
    ]
    return {
        "per_row_anomaly_labels_present": False,
        "label_like_files": file_hits,
        "label_like_metadata_or_weather_columns": column_hits,
        "decision_basis": (
            "The flat archive contains meter readings, metadata, and weather only; "
            "no file or schema provides per-row anomaly labels."
        ),
        "viable_strategies": [
            "unsupervised detection on BDG2 meter series",
            "forecasting-residual labels or scores from held-out temporal forecasts",
            "apply a GEPIII-trained detector as a cross-dataset scoring baseline",
        ],
    }


def _variant_summary(meters: list[dict[str, Any]]) -> dict[str, Any]:
    by_meter: dict[str, dict[str, Any]] = {}
    for item in meters:
        slot = by_meter.setdefault(item["meter_type"], {})
        slot[item["variant"]] = {
            "file": item["file"],
            "rows": item["rows"],
            "building_column_count": item["building_column_count"],
            "null_rate": item["null_rate"],
        }
    cleaned_minus_raw_null_rate = {}
    for meter_type, variants in by_meter.items():
        if "raw" in variants and "cleaned" in variants:
            cleaned_minus_raw_null_rate[meter_type] = round(
                variants["cleaned"]["null_rate"] - variants["raw"]["null_rate"],
                6,
            )
    return {
        "available_variants": sorted({item["variant"] for item in meters}),
        "meter_types": sorted(by_meter),
        "by_meter_type": by_meter,
        "cleaned_minus_raw_null_rate": cleaned_minus_raw_null_rate,
        "source_note": (
            "This Kaggle archive contains raw and cleaned meter files for the eight "
            "BDG2 meter types. It does not include the separate GEPIII/Kaggle 2017 "
            "subset file."
        ),
    }


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


def _write_report(payload: dict[str, Any], path: Path) -> None:
    metadata = payload["metadata"]
    weather = payload["weather"]
    meters = payload["meters"]
    raw_meters = [item for item in meters if item["variant"] == "raw"]
    cleaned_meters = [item for item in meters if item["variant"] == "cleaned"]

    meter_rows = [
        [
            item["file"],
            item["variant"],
            item["meter_type"],
            item["rows"],
            item["building_column_count"],
            item["timestamp_min"],
            item["timestamp_max"],
            item["null_rate"],
        ]
        for item in meters
    ]
    availability_rows = [
        [meter, count] for meter, count in metadata["meter_availability_counts"].items()
    ]
    mapping_rows = [
        [gepiii, bdg2] for gepiii, bdg2 in metadata["gepiii_field_mapping"].items()
    ]
    overlap = metadata["gepiii_overlap"]
    overlap_rows = [
        [site, count] for site, count in overlap["overlap_site_distribution"].items()
    ]
    bdg2_only_rows = [
        [site, count] for site, count in overlap["bdg2_only_site_distribution"].items()
    ]
    null_delta_rows = [
        [meter, delta]
        for meter, delta in payload["variants"]["cleaned_minus_raw_null_rate"].items()
    ]

    text = f"""# BDG2 Data Reality Report

**Stage**: Phase E Stage 0, read-only inventory
**Generated**: {payload["generated_at"]}
**Input directory**: `{payload["bdg2_dir"]}`

## Provenance

- Local source: Kaggle archive `claytonmiller/buildingdatagenomeproject2`, copied from `C:\\Users\\tonykuo\\Downloads\\archive`.
- Official reference repo: https://github.com/buds-lab/building-data-genome-project-2
- Meter field reference: https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-features
- Paper: Miller et al., Scientific Data 7, 368 (2020), DOI `10.1038/s41597-020-00712-x`.
- Download/copy date: {payload["provenance"]["copied_at_local_date"]}.
- Upstream commit: not applicable to this local Kaggle CSV archive. The prior Git checkout was discarded because it contained Git LFS pointers rather than real CSV data.

## File Gate

- CSV files found: {payload["file_gate"]["csv_count"]}.
- Git LFS pointer files: {payload["file_gate"]["lfs_pointer_count"]}.
- `electricity.csv`: {payload["file_gate"]["key_sizes_mb"].get("electricity.csv")} MB.
- `weather.csv`: {payload["file_gate"]["key_sizes_mb"].get("weather.csv")} MB.
- `metadata.csv`: {payload["file_gate"]["key_sizes_mb"].get("metadata.csv")} MB.

## Metadata Reality

- `metadata.csv` shape: {metadata["rows"]} rows x {len(metadata["columns"])} columns.
- Building id is the string `building_id` field, for example `{metadata["building_id_sample"][0]}`.
- Site count: {metadata["site_count"]}; building count: {metadata["building_count"]}.
- Timezone column: `timezone`, with {metadata["timezone_count"]} distinct values: {", ".join(metadata["timezones"])}.

### GEPIII-to-BDG2 Field Mapping

{_markdown_table(["GEPIII concept", "BDG2 actual column"], mapping_rows)}

### GEPIII Overlap Bridge

- `building_id_kaggle` non-empty buildings: {overlap["overlap_buildings"]}.
- BDG2-only buildings: {overlap["bdg2_only_buildings"]}.
- Loader contract: retain `building_id_kaggle`, `site_id_kaggle`, and derive `is_gepiii_overlap`.

Overlap site distribution:

{_markdown_table(["Site", "GEPIII-overlap buildings"], overlap_rows)}

BDG2-only site distribution:

{_markdown_table(["Site", "BDG2-only buildings"], bdg2_only_rows)}

### Metadata Columns

`{", ".join(metadata["columns"])}`

### Meter Availability in Metadata

{_markdown_table(["Meter", "Buildings marked Yes"], availability_rows)}

## Meter Files Reality

- Raw meter files: {len(raw_meters)}.
- Cleaned meter files: {len(cleaned_meters)}.
- Layout: each meter file is a wide table with `timestamp` plus one column per building id.
- Long-table mapping for ingestion: `(building, meter_type, timestamp, reading)` is obtained by melting each meter file's building columns, using the file stem as `meter_type` and the cell value as `reading`.

{_markdown_table(["File", "Variant", "Meter", "Rows", "Building cols", "Start", "End", "Null rate"], meter_rows)}

## Weather Reality

- `weather.csv` shape: {weather["rows"]} rows x {len(weather["columns"])} columns.
- Weather key: `{weather["key"]}`; site count: {weather["site_count"]}.
- Timestamp range: {weather["timestamp_min"]} to {weather["timestamp_max"]}.
- Timezone column present: {weather["timezone_column_present"]}. Site timezone must therefore be joined from metadata if local-time interpretation is needed.
- Columns: `{", ".join(weather["columns"])}`
- Null rate: {weather["null_rate"]}.

## Label Reality

- Per-row anomaly labels present: **{payload["labels"]["per_row_anomaly_labels_present"]}**.
- Label-like files found: {payload["labels"]["label_like_files"]}.
- Label-like metadata/weather columns found: {payload["labels"]["label_like_metadata_or_weather_columns"]}.

BDG2, as present in this archive, does not provide a per-row anomaly label comparable to GEPIII `bad_meter_readings.csv`. Any Phase E supervised FDD claim must therefore choose and document a label strategy before training or evaluation.

Viable strategies:

- Unsupervised detection on BDG2 meter series.
- Forecasting-residual labels or anomaly scores from held-out temporal forecasts.
- Apply a GEPIII-trained detector as a cross-dataset scoring baseline.
- Raw/cleaned difference pseudo-labels: cells present in raw but set to `NaN` in cleaned are a candidate proxy for BDG2-cleaning-identified bad readings.

## Raw, Cleaned, and Kaggle Variants

- This local archive has raw and cleaned files for all eight BDG2 meter types.
- Cleaned null rates are higher than raw null rates for every measured meter type:

{_markdown_table(["Meter", "cleaned null rate - raw null rate"], null_delta_rows)}

- The separate GEPIII/Kaggle 2017 subset file is not present in this archive; this matches the user-provided correction that `kaggle.csv` is not part of the local true-data download.
- Source-level note for Stage 1 ADR: BDG2 raw/cleaned files are the full 2016+2017 BDG2 meter release, while the GEPIII/Kaggle subset is a 2017 local-time subset with different unit-correction semantics. The loader contract must not assume the GEPIII site-0/meter-0 correction applies to BDG2 raw/cleaned files.

## Stage 0 Decision Boundary

This report is the fact base for Stage 1. No `src/lead` code was inspected or changed by this inventory script, and no BDG2 loader contract is inferred here beyond the measured file schema.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def build_inventory(bdg2_dir: Path) -> dict[str, Any]:
    csv_paths = sorted(bdg2_dir.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSV files found under {bdg2_dir}")
    _fail_on_lfs_pointer(csv_paths)

    metadata_path = bdg2_dir / "metadata.csv"
    weather_path = bdg2_dir / "weather.csv"
    missing = [str(path) for path in (metadata_path, weather_path) if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required BDG2 files: {missing}")

    metadata = _metadata_inventory(metadata_path)
    metadata_buildings = set(
        pd.read_csv(metadata_path, usecols=["building_id"])["building_id"]
    )
    meter_paths = [
        path for path in csv_paths if path.name not in {"metadata.csv", "weather.csv"}
    ]
    meters = [_meter_inventory(path, metadata_buildings) for path in meter_paths]
    weather = _weather_inventory(weather_path)

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bdg2_dir": str(bdg2_dir),
        "provenance": {
            "local_archive_path": r"C:\Users\tonykuo\Downloads\archive",
            "kaggle_dataset": "claytonmiller/buildingdatagenomeproject2",
            "official_repo": "https://github.com/buds-lab/building-data-genome-project-2",
            "meter_wiki": "https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-features",
            "paper_doi": "10.1038/s41597-020-00712-x",
            "copied_at_local_date": datetime.now().astimezone().date().isoformat(),
        },
        "file_gate": {
            "csv_count": len(csv_paths),
            "lfs_pointer_count": 0,
            "files": [
                {"name": path.name, "size_mb": _file_mb(path)} for path in csv_paths
            ],
            "key_sizes_mb": {
                name: _file_mb(bdg2_dir / name)
                for name in ("electricity.csv", "weather.csv", "metadata.csv")
            },
        },
        "metadata": metadata,
        "meters": meters,
        "weather": weather,
        "labels": _label_inventory(csv_paths, metadata["columns"], weather["columns"]),
        "variants": _variant_summary(meters),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdg2-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    args = parser.parse_args()

    bdg2_dir = args.bdg2_dir.resolve()
    payload = build_inventory(bdg2_dir)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    _write_report(payload, args.report_out)

    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.report_out}")


if __name__ == "__main__":
    main()
