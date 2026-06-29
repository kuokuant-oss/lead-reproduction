"""BDG2 ingestion helpers grounded in the real Phase E Stage 0 schema."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .data import ROOT


BDG2 = ROOT / "data" / "raw" / "bdg2"

BDG2_METER_TYPES = (
    "electricity",
    "chilledwater",
    "steam",
    "hotwater",
    "gas",
    "water",
    "irrigation",
    "solar",
)

BDG2_METADATA_COLUMNS = (
    "building_id",
    "site_id",
    "building_id_kaggle",
    "site_id_kaggle",
    "primaryspaceusage",
    "sqft",
    "yearbuilt",
    "numberoffloors",
    "timezone",
)

BDG2_WEATHER_RENAME = {
    "airTemperature": "air_temperature",
    "cloudCoverage": "cloud_coverage",
    "dewTemperature": "dew_temperature",
    "precipDepth1HR": "precip_depth_1_hr",
    "precipDepth6HR": "precip_depth_6_hr",
    "seaLvlPressure": "sea_level_pressure",
    "windDirection": "wind_direction",
    "windSpeed": "wind_speed",
}


def _normalise_meter_types(meter_types: Iterable[str] | None) -> tuple[str, ...]:
    if meter_types is None:
        return BDG2_METER_TYPES
    normalised = tuple(meter_types)
    unknown = sorted(set(normalised) - set(BDG2_METER_TYPES))
    if unknown:
        raise ValueError(f"Unknown BDG2 meter types: {unknown}")
    if not normalised:
        raise ValueError("meter_types must include at least one meter")
    return normalised


def _meter_path(bdg2_dir: Path, meter_type: str, variant: str) -> Path:
    if variant == "raw":
        filename = f"{meter_type}.csv"
    elif variant == "cleaned":
        filename = f"{meter_type}_cleaned.csv"
    else:
        raise ValueError("variant must be 'raw' or 'cleaned'")
    return bdg2_dir / filename


def _assert_columns(frame: pd.DataFrame, columns: Iterable[str], *, file: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{file} missing required columns: {', '.join(missing)}")


def _load_metadata(bdg2_dir: Path) -> pd.DataFrame:
    path = bdg2_dir / "metadata.csv"
    meta = pd.read_csv(path)
    _assert_columns(meta, BDG2_METADATA_COLUMNS, file=path.name)
    if meta["building_id"].isna().any():
        raise ValueError("metadata.csv has null building_id values")
    duplicate_count = int(meta["building_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"metadata.csv has {duplicate_count} duplicate building_id rows"
        )
    meta["is_gepiii_overlap"] = meta["building_id_kaggle"].notna() & meta[
        "building_id_kaggle"
    ].astype(str).str.strip().ne("")
    return meta.rename(
        columns={
            "primaryspaceusage": "primary_use",
            "sqft": "square_feet",
            "yearbuilt": "year_built",
            "numberoffloors": "floor_count",
        }
    )


def _load_meter_long(
    path: Path,
    *,
    meter_type: str,
    metadata_buildings: set[str],
    building_ids: Iterable[str] | None,
    nrows: int | None,
) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    _assert_columns(header, ("timestamp",), file=path.name)
    meter_buildings = [column for column in header.columns if column != "timestamp"]
    unknown_buildings = sorted(set(meter_buildings) - metadata_buildings)
    if unknown_buildings:
        sample = ", ".join(unknown_buildings[:5])
        raise ValueError(
            f"{path.name} has building columns absent from metadata: {sample}"
        )

    selected_buildings = (
        list(building_ids) if building_ids is not None else meter_buildings
    )
    missing_in_meter = sorted(set(selected_buildings) - set(meter_buildings))
    if missing_in_meter:
        sample = ", ".join(missing_in_meter[:5])
        raise ValueError(f"{path.name} does not contain requested buildings: {sample}")

    wide = pd.read_csv(path, usecols=["timestamp", *selected_buildings], nrows=nrows)
    long = wide.melt(
        id_vars="timestamp",
        var_name="building_id",
        value_name="meter_reading",
    )
    expected_rows = len(wide) * len(selected_buildings)
    if len(long) != expected_rows:
        raise ValueError(
            f"{path.name} reshape length mismatch: expected {expected_rows}, got {len(long)}"
        )
    long["timestamp"] = pd.to_datetime(long["timestamp"])
    long["meter"] = meter_type
    return long


def _load_weather(bdg2_dir: Path) -> pd.DataFrame:
    path = bdg2_dir / "weather.csv"
    weather = pd.read_csv(path)
    _assert_columns(weather, ("timestamp", "site_id"), file=path.name)
    weather = weather.rename(columns=BDG2_WEATHER_RENAME)
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    duplicate_count = int(weather.duplicated(["site_id", "timestamp"]).sum())
    if duplicate_count:
        raise ValueError(
            f"weather.csv has {duplicate_count} duplicate (site_id, timestamp) rows"
        )
    return weather


def _assert_unique_row_keys(frame: pd.DataFrame) -> None:
    duplicate_count = int(frame.duplicated(["building_id", "meter", "timestamp"]).sum())
    if duplicate_count:
        raise ValueError(
            "BDG2 row identity is undefined: "
            f"{duplicate_count} duplicate (building_id, meter, timestamp) keys"
        )


def load_bdg2_frame(
    *,
    bdg2_dir: Path | str = BDG2,
    variant: str = "cleaned",
    meter_types: Iterable[str] | None = None,
    building_ids: Iterable[str] | None = None,
    nrows: int | None = None,
    include_weather: bool = True,
) -> pd.DataFrame:
    """Load BDG2 meter CSVs into a long frame.

    The real BDG2 archive has one wide CSV per meter type. This loader melts
    those files into rows keyed by ``building_id``, ``meter``, and ``timestamp``,
    then joins measured metadata and optional site-level weather. BDG2 has no
    per-row anomaly label, so this function intentionally does not create an
    ``anomaly`` column.
    """

    bdg2_path = Path(bdg2_dir)
    meters = _normalise_meter_types(meter_types)
    requested_buildings = tuple(building_ids) if building_ids is not None else None

    meta = _load_metadata(bdg2_path)
    metadata_buildings = set(meta["building_id"])
    if requested_buildings is not None:
        unknown_requested = sorted(set(requested_buildings) - metadata_buildings)
        if unknown_requested:
            sample = ", ".join(unknown_requested[:5])
            raise ValueError(f"Requested buildings absent from metadata: {sample}")

    frames = [
        _load_meter_long(
            _meter_path(bdg2_path, meter_type, variant),
            meter_type=meter_type,
            metadata_buildings=metadata_buildings,
            building_ids=requested_buildings,
            nrows=nrows,
        )
        for meter_type in meters
    ]
    meter_frame = pd.concat(frames, ignore_index=True)
    _assert_unique_row_keys(meter_frame)

    meta_cols = [
        "building_id",
        "site_id",
        "building_id_kaggle",
        "site_id_kaggle",
        "is_gepiii_overlap",
        "primary_use",
        "square_feet",
        "year_built",
        "floor_count",
        "timezone",
    ]
    frame = meter_frame.merge(meta[meta_cols], on="building_id", how="left")
    if frame["site_id"].isna().any():
        raise ValueError("BDG2 metadata join left rows without site_id")

    if include_weather:
        weather = _load_weather(bdg2_path)
        frame = frame.merge(weather, on=["site_id", "timestamp"], how="left")

    ordered = [
        "building_id",
        "site_id",
        "building_id_kaggle",
        "site_id_kaggle",
        "is_gepiii_overlap",
        "timestamp",
        "meter",
        "meter_reading",
        "primary_use",
        "square_feet",
        "year_built",
        "floor_count",
        "timezone",
    ]
    ordered.extend(column for column in BDG2_WEATHER_RENAME.values() if column in frame)
    remaining = [column for column in frame.columns if column not in ordered]
    return frame[ordered + remaining]
