"""Read-only BDG2 meter/weather timestamp alignment diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METER_TYPES = ("electricity", "chilledwater")
VARIANTS = ("raw", "cleaned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bdg2-dir",
        type=Path,
        default=Path("data/raw/bdg2"),
        help="Local BDG2 directory containing metadata, weather, and meter CSVs.",
    )
    parser.add_argument(
        "--min-buildings",
        type=int,
        default=10,
        help="Minimum buildings for a site/meter aggregate.",
    )
    return parser.parse_args()


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return series * np.nan
    return (series - series.mean()) / std


def circular_hour_delta(load_hour: int, temp_hour: int) -> int:
    return int(((load_hour - temp_hour + 12) % 24) - 12)


def best_lag(temp: pd.Series, load: pd.Series) -> tuple[int, float, int]:
    best: tuple[int, float, int] | None = None
    for lag in range(-12, 13):
        shifted_temp = temp.shift(lag)
        aligned = pd.concat([shifted_temp, load], axis=1, keys=["temp", "load"])
        aligned = aligned.dropna()
        if len(aligned) < 1000:
            continue
        corr = float(aligned["temp"].corr(aligned["load"]))
        if pd.isna(corr):
            continue
        candidate = (lag, corr, int(len(aligned)))
        if best is None or abs(candidate[1]) > abs(best[1]):
            best = candidate
    if best is None:
        return (0, float("nan"), 0)
    return best


def weather_by_site(weather: pd.DataFrame) -> dict[str, pd.Series]:
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    out: dict[str, pd.Series] = {}
    for site_id, site_weather in weather.groupby("site_id", sort=False):
        series = (
            site_weather.sort_values("timestamp")
            .set_index("timestamp")["airTemperature"]
            .astype(float)
        )
        full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
        series = series.reindex(full_index)
        out[str(site_id)] = series.interpolate(method="time", limit_direction="both")
    return out


def site_meter_columns(
    meta: pd.DataFrame, meter_type: str, min_buildings: int
) -> dict[str, list[str]]:
    available = meta[meta[meter_type].astype(str).str.lower().eq("yes")]
    grouped = available.groupby("site_id")["building_id"].apply(list)
    return {
        str(site_id): buildings
        for site_id, buildings in grouped.items()
        if len(buildings) >= min_buildings
    }


def meter_path(bdg2_dir: Path, meter_type: str, variant: str) -> Path:
    if variant == "raw":
        return bdg2_dir / f"{meter_type}.csv"
    return bdg2_dir / f"{meter_type}_cleaned.csv"


def diagnose_meter_variant(
    *,
    bdg2_dir: Path,
    meta: pd.DataFrame,
    weather_series: dict[str, pd.Series],
    meter_type: str,
    variant: str,
    min_buildings: int,
) -> list[dict[str, object]]:
    sites = site_meter_columns(meta, meter_type, min_buildings)
    if not sites:
        return []
    all_buildings = sorted(
        {building for buildings in sites.values() for building in buildings}
    )
    path = meter_path(bdg2_dir, meter_type, variant)
    meter = pd.read_csv(path, usecols=["timestamp", *all_buildings])
    meter["timestamp"] = pd.to_datetime(meter["timestamp"])
    meter = meter.set_index("timestamp").sort_index()

    rows: list[dict[str, object]] = []
    for site_id, buildings in sites.items():
        if site_id not in weather_series:
            continue
        temp = weather_series[site_id]
        load = meter[buildings].mean(axis=1, skipna=True)
        joined = pd.concat(
            [temp, load], axis=1, keys=["temp", "load"], sort=False
        ).dropna()
        if len(joined) < 1000:
            continue

        temp_z = zscore(joined["temp"])
        load_z = zscore(np.log1p(joined["load"].clip(lower=0)))
        lag, corr, n_aligned = best_lag(temp_z, load_z)

        temp_profile = joined["temp"].groupby(joined.index.hour).mean()
        load_profile = joined["load"].groupby(joined.index.hour).mean()
        temp_peak = int(temp_profile.idxmax())
        load_peak = int(load_profile.idxmax())
        rows.append(
            {
                "site_id": site_id,
                "meter": meter_type,
                "variant": variant,
                "buildings": int(len(buildings)),
                "rows": int(len(joined)),
                "temperature_peak_hour": temp_peak,
                "load_peak_hour": load_peak,
                "load_minus_temperature_peak_hours": circular_hour_delta(
                    load_peak, temp_peak
                ),
                "best_temperature_lag_hours": int(lag),
                "best_abs_correlation": round(float(abs(corr)), 4),
                "best_signed_correlation": round(float(corr), 4),
                "correlation_rows": n_aligned,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    meta = pd.read_csv(args.bdg2_dir / "metadata.csv")
    weather = pd.read_csv(args.bdg2_dir / "weather.csv")
    weather_missing = float(weather["airTemperature"].isna().mean())
    weather_series = weather_by_site(weather)

    rows: list[dict[str, object]] = []
    for meter_type in METER_TYPES:
        for variant in VARIANTS:
            rows.extend(
                diagnose_meter_variant(
                    bdg2_dir=args.bdg2_dir,
                    meta=meta,
                    weather_series=weather_series,
                    meter_type=meter_type,
                    variant=variant,
                    min_buildings=args.min_buildings,
                )
            )

    result = pd.DataFrame(rows).sort_values(["meter", "variant", "site_id"])
    if result.empty:
        raise SystemExit("No site/meter aggregates met the diagnostic threshold")

    print(f"weather_airTemperature_missing_rate={weather_missing:.6f}")
    print(
        result.to_string(
            index=False,
            columns=[
                "site_id",
                "meter",
                "variant",
                "buildings",
                "rows",
                "temperature_peak_hour",
                "load_peak_hour",
                "load_minus_temperature_peak_hours",
                "best_temperature_lag_hours",
                "best_signed_correlation",
                "best_abs_correlation",
            ],
        )
    )

    summary = (
        result.groupby(["meter", "variant"])
        .agg(
            sites=("site_id", "count"),
            median_abs_peak_delta=(
                "load_minus_temperature_peak_hours",
                lambda s: float(np.median(np.abs(s))),
            ),
            median_abs_best_lag=(
                "best_temperature_lag_hours",
                lambda s: float(np.median(np.abs(s))),
            ),
            median_abs_correlation=("best_abs_correlation", "median"),
        )
        .reset_index()
    )
    print()
    print("summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
