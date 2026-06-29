"""Phase E Step 3 BDG2 unlabeled transfer smoke under ADR 0019."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lead import (
    BASELINE_FEATURE_COLS,
    DOWNSAMPLE_SEEDS,
    RANDOM_STATE,
    SHIFTS,
    add_value_change_features,
    assert_no_building_overlap,
    downsample_indices,
    load_bdg2_frame,
    load_m3_frame,
)


BDG2_DIR = Path("data/raw/bdg2")
OUT = Path(".scratch/phaseE-step3-bdg2-transfer-smoke.json")
METER_TYPE = "chilledwater"
METER_CODE = {"electricity": 0, "chilledwater": 1, "steam": 2, "hotwater": 3}


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdg2-dir", type=Path, default=BDG2_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--meter", default=METER_TYPE, choices=sorted(METER_CODE))
    parser.add_argument(
        "--site",
        default=None,
        help="BDG2 site id. Defaults to the site with most selected-meter buildings.",
    )
    return parser.parse_args()


def m3_primary_use_mapping() -> dict[str, int]:
    meta = pd.read_csv(Path("data/raw/m3/building_metadata.csv"))
    classes = sorted(meta["primary_use"].fillna("Unknown").unique())
    return {label: idx for idx, label in enumerate(classes)}


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
    site_rows = available[available["site_id"].astype(str).eq(site)]
    if site_rows.empty:
        raise ValueError(f"No {meter} buildings found for BDG2 site {site}")
    return site, site_rows["building_id"].astype(str).tolist()


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
        raise ValueError(f"BDG2 smoke frame missing columns: {missing}")
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


def fit_gepiii_lightgbm_detector() -> dict[str, Any]:
    t0 = time.perf_counter()
    df = load_m3_frame(verbose=True)
    mask_val = (df["building_id"] % 5 == 4).to_numpy()
    overlap = assert_no_building_overlap(
        set(df.loc[~mask_val, "building_id"].unique()),
        set(df.loc[mask_val, "building_id"].unique()),
        split_name="80_20_mod5",
    )

    log("Adding GEPIII M3.2 offline value-change features for source detector")
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

    model = lgb.LGBMClassifier(
        n_estimators=100,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train.loc[ds_idx])
    elapsed = time.perf_counter() - t0
    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "source_summary": {
            "detector": "m3_2_lightgbm_gbdt",
            "feature_table": "m3_2_137_features",
            "feature_regime": "offline",
            # Single-meter BDG2 scoring uses row_offset_meter_aware; Step 1 proved
            # that this is row-by-row equivalent to row_offset for one meter.
            "value_change_regime": "row_offset",
            "split": "80_20_mod5_source_train",
            "building_overlap": int(len(overlap)),
            "fit_rows": int(len(ds_idx)),
            "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
            "random_state": RANDOM_STATE,
            "fit_seconds": float(elapsed),
        },
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


def score_summary(scores: np.ndarray, mask: pd.Series) -> dict[str, Any]:
    subset = scores[mask.to_numpy()]
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


def score_bdg2_cleaned(
    *,
    frame: pd.DataFrame,
    detector: dict[str, Any],
    meter: str,
) -> dict[str, Any]:
    primary_mapping = m3_primary_use_mapping()
    featured = prepare_bdg2_features(
        frame,
        meter=meter,
        primary_use_mapping=primary_mapping,
        feature_cols=detector["feature_cols"],
    )
    x = detector["scaler"].transform(featured[detector["feature_cols"]])
    t0 = time.perf_counter()
    scores = detector["model"].predict_proba(x)[:, 1]
    score_seconds = time.perf_counter() - t0

    overlap_mask = featured["is_gepiii_overlap"].astype(bool)
    all_mask = pd.Series(True, index=featured.index)
    overlap_stats = score_summary(scores, overlap_mask)
    bdg2_only_stats = score_summary(scores, ~overlap_mask)
    return {
        "scored_variant": "cleaned",
        "feature_regime": "offline",
        "single_meter_value_change_equivalence": (
            "row_offset_meter_aware is equivalent to row_offset for this "
            "single-meter slice; multi-meter transfer must align train/serve "
            "value-change semantics before scoring."
        ),
        "value_change_regime": "row_offset_meter_aware",
        "rows_scored": int(len(featured)),
        "score_seconds": float(score_seconds),
        "score_rows_per_second": float(len(featured) / score_seconds)
        if score_seconds > 0
        else None,
        "feature_missing_rates": {
            column: float(featured[column].isna().mean())
            for column in detector["feature_cols"]
            if featured[column].isna().any()
        },
        "primary_use_unseen_rate": float((featured["primary_use_enc"] < 0).mean()),
        "score_summary": {
            "all": score_summary(scores, all_mask),
            "gepiii_overlap": overlap_stats,
            "bdg2_only": bdg2_only_stats,
        },
        "overlap_vs_bdg2_only": {
            "median_score_delta_overlap_minus_bdg2_only": (
                overlap_stats.get("score_median", float("nan"))
                - bdg2_only_stats.get("score_median", float("nan"))
            )
            if bdg2_only_stats.get("rows", 0)
            else None,
            "note": "Unlabeled score distribution contrast only; not a headline metric.",
        },
    }


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    site, buildings = selected_site_buildings(
        args.bdg2_dir, meter=args.meter, site=args.site
    )
    log(f"Selected BDG2 site={site} meter={args.meter} buildings={len(buildings)}")

    raw = load_bdg2_frame(
        bdg2_dir=args.bdg2_dir,
        variant="raw",
        meter_types=[args.meter],
        building_ids=buildings,
        include_weather=True,
    )
    raw_summary = schema_summary(raw)
    del raw

    cleaned = load_bdg2_frame(
        bdg2_dir=args.bdg2_dir,
        variant="cleaned",
        meter_types=[args.meter],
        building_ids=buildings,
        include_weather=True,
    )
    cleaned_summary = schema_summary(cleaned)

    detector = fit_gepiii_lightgbm_detector()
    scoring = score_bdg2_cleaned(frame=cleaned, detector=detector, meter=args.meter)
    result = {
        "schema_version": 1,
        "experiment": "phaseE_step3_bdg2_transfer_smoke",
        "adr": "0019-bdg2-evaluation-paradigm",
        "metric_contract": {
            "path": "unlabeled_score_transfer",
            "bdg2_ground_truth_metrics_reported": False,
            "forbidden_metric_keys_absent": [
                "roc_auc",
                "pr_auc",
                "precision",
                "recall",
                "f1",
            ],
            "headline_metric": False,
        },
        "selection": {
            "detector": "m3_2_lightgbm_gbdt",
            "tabpfn_step3_status": (
                "tabpfn_not_scored_in_smoke_full_score_cost_tradeoff"
            ),
            "meter": args.meter,
            "site_id": site,
            "site_selection": "most_buildings_for_meter",
            "buildings": int(len(buildings)),
        },
        "bdg2_load": {
            "raw": raw_summary,
            "cleaned": cleaned_summary,
        },
        "detector_source": detector["source_summary"],
        "scoring": scoring,
        "elapsed_seconds": float(time.perf_counter() - t0),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"Saved {args.out}")


if __name__ == "__main__":
    main()
