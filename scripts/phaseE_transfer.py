"""Shared helpers for Phase E BDG2 transfer runners.

These helpers keep Phase E scoring under ADR 0019: BDG2 outputs are unlabeled
score-transfer summaries, not supervised ground-truth metrics.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from lead import (
    BASELINE_FEATURE_COLS,
    DOWNSAMPLE_SEEDS,
    RANDOM_STATE,
    SHIFTS,
    add_value_change_features,
    assert_no_building_overlap,
    downsample_indices,
    load_m3_frame,
)


METER_CODE = {"electricity": 0, "chilledwater": 1, "steam": 2, "hotwater": 3}
BDG2_DIR = Path("data/raw/bdg2")
HIGH_MISSING_RATE = 0.5
MIN_STRATUM_BUILDINGS = 5
MIN_STRATUM_ROWS = 17_544


def log(message: str) -> None:
    print(message, flush=True)


def json_clean(value: Any) -> Any:
    """Convert NumPy/pandas scalars and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def m3_primary_use_mapping() -> dict[str, int]:
    meta = pd.read_csv(Path("data/raw/m3/building_metadata.csv"))
    classes = sorted(meta["primary_use"].fillna("Unknown").unique())
    return {label: idx for idx, label in enumerate(classes)}


def site_building_summary(bdg2_dir: Path, *, meter: str) -> pd.DataFrame:
    meta = pd.read_csv(bdg2_dir / "metadata.csv")
    available = meta[meta[meter].astype(str).str.lower().eq("yes")].copy()
    available["is_gepiii_overlap"] = available["building_id_kaggle"].notna() & (
        available["building_id_kaggle"].astype(str).str.strip() != ""
    )
    summary = (
        available.groupby("site_id")
        .agg(
            buildings=("building_id", "nunique"),
            bdg2_only_buildings=("is_gepiii_overlap", lambda s: int((~s).sum())),
            gepiii_overlap_buildings=("is_gepiii_overlap", lambda s: int(s.sum())),
        )
        .reset_index()
    )
    summary["site_id"] = summary["site_id"].astype(str)
    return summary.sort_values(
        ["buildings", "bdg2_only_buildings", "site_id"],
        ascending=[False, False, True],
    )


def selected_site_buildings(
    bdg2_dir: Path, *, meter: str, site: str | None
) -> tuple[str, list[str]]:
    meta = pd.read_csv(bdg2_dir / "metadata.csv")
    available = meta[meta[meter].astype(str).str.lower().eq("yes")]
    if site is None:
        site = str(
            available.groupby("site_id")["building_id"]
            .count()
            .sort_values(ascending=False)
            .index[0]
        )
    site_rows = available[available["site_id"].astype(str).eq(str(site))]
    if site_rows.empty:
        raise ValueError(f"No {meter} buildings found for BDG2 site {site}")
    return str(site), site_rows["building_id"].astype(str).tolist()


def pilot_sites(bdg2_dir: Path, *, meter: str) -> list[str]:
    summary = site_building_summary(bdg2_dir, meter=meter)
    if summary.empty:
        raise ValueError(f"No BDG2 sites with meter={meter}")
    fox = summary[summary["site_id"].eq("Fox")]
    sites: list[str] = []
    if not fox.empty:
        sites.append("Fox")
    else:
        sites.append(str(summary.iloc[0]["site_id"]))

    candidates = summary[~summary["site_id"].isin(sites)].sort_values(
        ["bdg2_only_buildings", "buildings", "site_id"],
        ascending=[False, False, True],
    )
    if not candidates.empty:
        sites.append(str(candidates.iloc[0]["site_id"]))
    return sites


def all_sites(bdg2_dir: Path, *, meter: str) -> list[str]:
    summary = site_building_summary(bdg2_dir, meter=meter)
    return [str(site) for site in summary["site_id"].tolist()]


def schema_summary(frame: pd.DataFrame) -> dict[str, Any]:
    required = [
        "building_id",
        "meter",
        "timestamp",
        "meter_reading",
        "building_id_kaggle",
        "site_id_kaggle",
        "is_gepiii_overlap",
        "air_temperature",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"BDG2 frame missing columns: {missing}")
    return {
        "rows": int(len(frame)),
        "buildings": int(frame["building_id"].nunique()),
        "site_id": sorted(frame["site_id"].astype(str).unique()),
        "meter_values": sorted(frame["meter"].astype(str).unique()),
        "has_required_schema": True,
        "has_weather_join": "air_temperature" in frame.columns,
        "air_temperature_missing_rate": float(frame["air_temperature"].isna().mean()),
        "is_gepiii_overlap_counts": {
            str(key): int(value)
            for key, value in frame["is_gepiii_overlap"].value_counts().items()
        },
    }


def m3_source_table() -> dict[str, Any]:
    df = load_m3_frame(verbose=True)
    mask_val = (df["building_id"] % 5 == 4).to_numpy()
    overlap = assert_no_building_overlap(
        set(df.loc[~mask_val, "building_id"].unique()),
        set(df.loc[mask_val, "building_id"].unique()),
        split_name="80_20_mod5",
    )
    log("Adding GEPIII M3.2 offline value-change features")
    train_full = add_value_change_features(df.loc[~mask_val], list(SHIFTS))
    value_cols = [
        column for column in train_full.columns if column.startswith("lag_value_")
    ]
    feature_cols = BASELINE_FEATURE_COLS + value_cols
    if len(feature_cols) != 137:
        raise AssertionError(f"Expected 137 M3.2 features, got {len(feature_cols)}")
    y_train = train_full["anomaly"]
    ds_idx = downsample_indices(y_train)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[ds_idx, feature_cols])
    return {
        "x_train": x_train,
        "y_fit": y_train.loc[ds_idx],
        "feature_cols": feature_cols,
        "scaler": scaler,
        "source_summary": {
            "feature_table": "m3_2_137_features",
            "feature_regime": "offline",
            "value_change_regime": "row_offset",
            "split": "80_20_mod5_source_train",
            "building_overlap": int(len(overlap)),
            "fit_rows": int(len(ds_idx)),
            "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
            "random_state": RANDOM_STATE,
        },
    }


def fit_gepiii_lightgbm_detector() -> dict[str, Any]:
    t0 = time.perf_counter()
    table = m3_source_table()
    model = lgb.LGBMClassifier(
        n_estimators=100,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(table["x_train"], table["y_fit"])
    summary = dict(table["source_summary"])
    summary.update(
        {"detector": "m3_2_lightgbm_gbdt", "fit_seconds": time.perf_counter() - t0}
    )
    return {
        "kind": "single_model",
        "model": model,
        "scaler": table["scaler"],
        "feature_cols": table["feature_cols"],
        "source_summary": summary,
    }


def fit_gepiii_seed42_ensemble() -> dict[str, Any]:
    t0 = time.perf_counter()
    table = m3_source_table()
    seed = RANDOM_STATE
    models = {
        "lightgbm": lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=seed),
        "xgboost": xgb.XGBClassifier(
            n_estimators=100,
            eval_metric="logloss",
            verbosity=0,
            random_state=seed,
        ),
        "catboost": CatBoostClassifier(
            iterations=1000,
            verbose=False,
            random_seed=seed,
            allow_writing_files=False,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            random_state=seed,
        ),
    }
    train_seconds: dict[str, float] = {}
    for name, model in models.items():
        mt0 = time.perf_counter()
        log(f"Fitting GEPIII source model: {name}")
        x_fit = (
            np.nan_to_num(table["x_train"], nan=0)
            if name == "hist_gradient_boosting"
            else table["x_train"]
        )
        model.fit(x_fit, table["y_fit"])
        train_seconds[name] = time.perf_counter() - mt0
    summary = dict(table["source_summary"])
    summary.update(
        {
            "detector": "m3_4_seed42_four_model_ensemble",
            "model_names": list(models),
            "ensemble_weighting": "equal_weight_mean_predict_proba",
            "fit_seconds": time.perf_counter() - t0,
            "model_fit_seconds": train_seconds,
        }
    )
    return {
        "kind": "ensemble",
        "models": models,
        "scaler": table["scaler"],
        "feature_cols": table["feature_cols"],
        "source_summary": summary,
    }


def prepare_bdg2_features(
    frame: pd.DataFrame,
    *,
    meter: str,
    primary_use_mapping: dict[str, int],
    feature_cols: list[str],
) -> pd.DataFrame:
    if meter not in METER_CODE:
        raise ValueError(f"No GEPIII meter code mapping for BDG2 meter {meter}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["meter"] = METER_CODE[meter]
    out["hour"] = out["timestamp"].dt.hour.astype("int8")
    out["weekday"] = out["timestamp"].dt.weekday.astype("int8")
    out["month"] = out["timestamp"].dt.month.astype("int8")
    out["dayofyear"] = (
        out["timestamp"].dt.dayofyear + out["timestamp"].dt.hour / 24
    ).astype("float32")
    out["primary_use_enc"] = (
        out["primary_use"].fillna("Unknown").map(primary_use_mapping).fillna(-1)
    ).astype("int16")
    out["log_square_feet"] = np.log1p(out["square_feet"]).astype("float32")
    featured = add_value_change_features(
        out,
        list(SHIFTS),
        value_change_regime="row_offset_meter_aware",
    )
    missing = [column for column in feature_cols if column not in featured.columns]
    if missing:
        raise ValueError(f"BDG2 feature frame missing model columns: {missing}")
    return featured


def predict_scores(detector: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    x_score = detector["scaler"].transform(features[detector["feature_cols"]])
    if detector["kind"] == "single_model":
        return detector["model"].predict_proba(x_score)[:, 1]
    preds = []
    for name, model in detector["models"].items():
        x_model = (
            np.nan_to_num(x_score, nan=0)
            if name == "hist_gradient_boosting"
            else x_score
        )
        preds.append(model.predict_proba(x_model)[:, 1])
    return np.mean(preds, axis=0)


def score_summary(scores: np.ndarray, mask: pd.Series | np.ndarray) -> dict[str, Any]:
    mask_array = mask.to_numpy() if isinstance(mask, pd.Series) else np.asarray(mask)
    subset = scores[mask_array]
    if len(subset) == 0:
        return {"rows": 0}
    finite = np.isfinite(subset)
    finite_scores = subset[finite]
    if len(finite_scores) == 0:
        return {
            "rows": int(len(subset)),
            "finite_scores": 0,
            "score_coverage": 0.0,
            "missing_score_rate": 1.0,
        }
    quantiles = np.quantile(
        finite_scores, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
    )
    return {
        "rows": int(len(subset)),
        "finite_scores": int(finite.sum()),
        "score_coverage": float(finite.mean()),
        "missing_score_rate": float(1 - finite.mean()),
        "score_min": float(quantiles[0]),
        "score_p01": float(quantiles[1]),
        "score_p05": float(quantiles[2]),
        "score_p25": float(quantiles[3]),
        "score_median": float(quantiles[4]),
        "score_p75": float(quantiles[5]),
        "score_p95": float(quantiles[6]),
        "score_p99": float(quantiles[7]),
        "score_max": float(quantiles[8]),
    }


def building_meter_completeness(featured: pd.DataFrame) -> pd.DataFrame:
    """Classify each (building_id, meter) by direct reading coverage."""
    if featured.empty:
        return pd.DataFrame(
            columns=[
                "building_id",
                "meter",
                "rows",
                "meter_reading_missing_rate",
                "completeness",
            ]
        )
    grouped = (
        featured.groupby(["building_id", "meter"], dropna=False)["meter_reading"]
        .agg(rows="size", missing=lambda s: int(s.isna().sum()))
        .reset_index()
    )
    grouped["meter_reading_missing_rate"] = grouped["missing"] / grouped["rows"]
    grouped["completeness"] = np.where(
        grouped["meter_reading_missing_rate"] > HIGH_MISSING_RATE,
        "high_missing",
        "sufficient_obs",
    )
    return grouped.drop(columns=["missing"])


def completeness_label(featured: pd.DataFrame) -> pd.Series:
    completeness = building_meter_completeness(featured)
    if completeness.empty:
        return pd.Series("high_missing", index=featured.index, dtype="object")
    keyed = featured[["building_id", "meter"]].merge(
        completeness[["building_id", "meter", "completeness"]],
        on=["building_id", "meter"],
        how="left",
    )
    return keyed["completeness"].fillna("high_missing").set_axis(featured.index)


def distribution(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = values.dropna()
    if finite.empty:
        return {
            "rows": int(len(values)),
            "finite_rows": 0,
            "missing_rate": float(values.isna().mean()) if len(values) else 0.0,
        }
    quantiles = finite.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    return {
        "rows": int(len(values)),
        "finite_rows": int(len(finite)),
        "missing_rate": float(values.isna().mean()),
        "min": float(quantiles.loc[0]),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(quantiles.loc[1]),
    }


def ood_summary(
    featured: pd.DataFrame, *, feature_cols: Iterable[str]
) -> dict[str, Any]:
    if featured.empty:
        return {
            "primary_use_unseen_rate": 0.0,
            "feature_missing_rates": {},
            "model_feature_missing_rate": None,
            "square_feet_distribution": distribution(featured["square_feet"]),
            "meter_reading_distribution": distribution(featured["meter_reading"]),
        }
    lag_cols = [
        "lag_value_diff_1",
        "lag_value_ratio_1",
        "lag_value_diff_24",
        "lag_value_ratio_24",
        "lag_value_diff_168",
        "lag_value_ratio_168",
    ]
    missing_cols = ["meter_reading", "log_square_feet", *lag_cols]
    return {
        "primary_use_unseen_rate": float((featured["primary_use_enc"] < 0).mean()),
        "feature_missing_rates": {
            column: float(featured[column].isna().mean())
            for column in missing_cols
            if column in featured.columns
        },
        "model_feature_missing_rate": float(
            featured[list(feature_cols)].isna().mean().mean()
        ),
        "square_feet_distribution": distribution(featured["square_feet"]),
        "meter_reading_distribution": distribution(featured["meter_reading"]),
    }


def stratified_score_report(
    *,
    featured: pd.DataFrame,
    scores: np.ndarray,
    feature_cols: list[str],
) -> dict[str, Any]:
    overlap_mask = featured["is_gepiii_overlap"].astype(bool)
    all_mask = pd.Series(True, index=featured.index)
    strata = {
        "all": all_mask,
        "gepiii_overlap": overlap_mask,
        "bdg2_only": ~overlap_mask,
    }
    report: dict[str, Any] = {}
    for name, mask in strata.items():
        subset = featured.loc[mask]
        report[name] = {
            "score_summary": score_summary(scores, mask),
            "ood_summary": ood_summary(subset, feature_cols=feature_cols),
            "buildings": int(subset["building_id"].nunique()),
            "rows": int(len(subset)),
        }
    completeness = completeness_label(featured)
    completeness_strata: dict[str, Any] = {}
    for overlap_name, overlap in [
        ("gepiii_overlap", True),
        ("bdg2_only", False),
    ]:
        for completeness_name in ["sufficient_obs", "high_missing"]:
            mask = (featured["is_gepiii_overlap"].astype(bool) == overlap) & (
                completeness == completeness_name
            )
            subset = featured.loc[mask]
            key = f"{overlap_name}__{completeness_name}"
            completeness_strata[key] = {
                "score_summary": score_summary(scores, mask),
                "ood_summary": ood_summary(subset, feature_cols=feature_cols),
                "buildings": int(subset["building_id"].nunique()),
                "rows": int(len(subset)),
            }
    report["completeness_strata"] = completeness_strata
    report["building_meter_completeness"] = building_meter_completeness(
        featured
    ).to_dict(orient="records")
    if report["bdg2_only"]["score_summary"].get("rows", 0):
        overlap_median = report["gepiii_overlap"]["score_summary"].get("score_median")
        bdg2_only_median = report["bdg2_only"]["score_summary"].get("score_median")
        delta = (
            overlap_median - bdg2_only_median
            if overlap_median is not None and bdg2_only_median is not None
            else None
        )
        report["overlap_vs_bdg2_only"] = {
            "median_score_delta_overlap_minus_bdg2_only": delta,
            "interpretation": (
                "Unlabeled score-distribution contrast only; not accuracy, "
                "not readiness, and not calibrated risk."
            ),
        }
    return report


def stratum_is_powered(
    stratum: dict[str, Any],
    *,
    min_buildings: int = MIN_STRATUM_BUILDINGS,
    min_rows: int = MIN_STRATUM_ROWS,
) -> bool:
    return (
        int(stratum.get("buildings", 0)) >= min_buildings
        and int(stratum.get("rows", stratum.get("score_summary", {}).get("rows", 0)))
        >= min_rows
    )
