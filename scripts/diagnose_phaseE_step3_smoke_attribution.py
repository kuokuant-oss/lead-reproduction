"""Attribute the Phase E Step 3 Fox/chilledwater score split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lead import load_bdg2_frame, load_m3_frame
from run_phaseE_step3_bdg2_transfer_smoke import (
    OUT as SMOKE_OUT,
    METER_TYPE,
    fit_gepiii_lightgbm_detector,
    prepare_bdg2_features,
    score_summary,
    selected_site_buildings,
)


OUT = Path(".scratch/phaseE-step3.5-smoke-attribution.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdg2-dir", type=Path, default=Path("data/raw/bdg2"))
    parser.add_argument("--meter", default=METER_TYPE)
    parser.add_argument("--site", default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke-out", type=Path, default=SMOKE_OUT)
    return parser.parse_args()


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


def missing_rates(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return {
        column: float(frame[column].isna().mean())
        for column in columns
        if column in frame.columns
    }


def stratum_summary(
    *,
    featured: pd.DataFrame,
    scores: np.ndarray,
    mask: pd.Series,
    score_summary_name: str,
) -> dict[str, Any]:
    subset = featured.loc[mask].copy()
    lag_cols = [
        "lag_value_diff_1",
        "lag_value_ratio_1",
        "lag_value_diff_24",
        "lag_value_ratio_24",
        "lag_value_diff_168",
        "lag_value_ratio_168",
    ]
    return {
        "rows": int(len(subset)),
        "buildings": int(subset["building_id"].nunique()),
        "primary_use_unseen_rate": float((subset["primary_use_enc"] < 0).mean())
        if len(subset)
        else 0.0,
        "feature_missing_rates": {
            "log_square_feet": float(subset["log_square_feet"].isna().mean())
            if len(subset)
            else 0.0,
            "meter_reading": float(subset["meter_reading"].isna().mean())
            if len(subset)
            else 0.0,
            **missing_rates(subset, lag_cols),
        },
        "square_feet_distribution": distribution(subset["square_feet"]),
        "log_square_feet_distribution": distribution(subset["log_square_feet"]),
        "meter_reading_distribution": distribution(subset["meter_reading"]),
        "score_summary": score_summary(scores, mask),
        "score_summary_name": score_summary_name,
    }


def source_detector_reference() -> dict[str, Any]:
    """Reference GEPIII source ranges for the selected detector family."""
    df = load_m3_frame(verbose=False)
    source = df.loc[df["building_id"] % 5 != 4].copy()
    chilledwater = source[source["meter"] == 1].copy()
    return {
        "source_split": "GEPIII 80_20_mod5 train rows only",
        "all_source_rows": int(len(source)),
        "chilledwater_source_rows": int(len(chilledwater)),
        "all_source_square_feet": distribution(np.expm1(source["log_square_feet"])),
        "chilledwater_source_square_feet": distribution(
            np.expm1(chilledwater["log_square_feet"])
        ),
        "all_source_meter_reading": distribution(source["meter_reading"]),
        "chilledwater_source_meter_reading": distribution(
            chilledwater["meter_reading"]
        ),
    }


def ood_flags(
    *,
    stratum: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    source_sq = source["chilledwater_source_square_feet"]
    source_mr = source["chilledwater_source_meter_reading"]
    sq = stratum["square_feet_distribution"]
    mr = stratum["meter_reading_distribution"]
    return {
        "square_feet_median_vs_source_median_ratio": float(
            sq["median"] / source_sq["median"]
        )
        if source_sq.get("median")
        else None,
        "meter_reading_median_vs_source_median_ratio": float(
            mr["median"] / source_mr["median"]
        )
        if source_mr.get("median")
        else None,
        "square_feet_p99_above_source_p99": bool(sq["p99"] > source_sq["p99"]),
        "meter_reading_p99_above_source_p99": bool(mr["p99"] > source_mr["p99"]),
        "meter_reading_max_above_source_max": bool(mr["max"] > source_mr["max"]),
    }


def attribution_call(strata: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    overlap = strata["gepiii_overlap"]
    bdg2_only = strata["bdg2_only"]
    ood = {
        "gepiii_overlap": ood_flags(stratum=overlap, source=source),
        "bdg2_only": ood_flags(stratum=bdg2_only, source=source),
    }
    evidence = {
        "primary_use_unseen_rate_delta": float(
            bdg2_only["primary_use_unseen_rate"] - overlap["primary_use_unseen_rate"]
        ),
        "log_square_feet_missing_delta": float(
            bdg2_only["feature_missing_rates"]["log_square_feet"]
            - overlap["feature_missing_rates"]["log_square_feet"]
        ),
        "meter_reading_missing_delta": float(
            bdg2_only["feature_missing_rates"]["meter_reading"]
            - overlap["feature_missing_rates"]["meter_reading"]
        ),
        "bdg2_only_meter_reading_median_vs_overlap": float(
            bdg2_only["meter_reading_distribution"]["median"]
            / overlap["meter_reading_distribution"]["median"]
        )
        if overlap["meter_reading_distribution"].get("median")
        else None,
        "bdg2_only_square_feet_median_vs_overlap": float(
            bdg2_only["square_feet_distribution"]["median"]
            / overlap["square_feet_distribution"]["median"]
        )
        if overlap["square_feet_distribution"].get("median")
        else None,
    }
    return {
        "call": "ii_detector_ood_or_feature_distribution_artifact_more_likely",
        "not_accuracy_or_readiness": True,
        "reason": (
            "BDG2 has no labels in this path, primary_use_unseen_rate is 0 in both "
            "strata, and the large score split tracks feature distribution and "
            "missingness differences rather than supervised evidence. Treat the "
            "BDG2-only uplift as OOD/extrapolation risk until a full-transfer plan "
            "adds OOD flags and stratified reporting."
        ),
        "evidence": evidence,
        "ood_flags": ood,
        "mitigation_for_full_transfer": [
            "Report OOD flags beside every score stratum.",
            "Prefer rank/quantile summaries over absolute score interpretation.",
            "Separate BDG2-only and GEPIII-overlap outputs before any headline.",
            "Consider detector-feature-distribution filters before headline tables.",
        ],
    }


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    site, buildings = selected_site_buildings(
        args.bdg2_dir, meter=args.meter, site=args.site
    )
    detector = fit_gepiii_lightgbm_detector()
    frame = load_bdg2_frame(
        bdg2_dir=args.bdg2_dir,
        variant="cleaned",
        meter_types=[args.meter],
        building_ids=buildings,
        include_weather=True,
    )
    primary_mapping = {
        label: idx
        for idx, label in enumerate(
            sorted(
                pd.read_csv(Path("data/raw/m3/building_metadata.csv"))["primary_use"]
                .fillna("Unknown")
                .unique()
            )
        )
    }
    featured = prepare_bdg2_features(
        frame,
        meter=args.meter,
        primary_use_mapping=primary_mapping,
        feature_cols=detector["feature_cols"],
    )
    x = detector["scaler"].transform(featured[detector["feature_cols"]])
    score_t0 = time.perf_counter()
    scores = detector["model"].predict_proba(x)[:, 1]
    score_seconds = time.perf_counter() - score_t0

    overlap_mask = featured["is_gepiii_overlap"].astype(bool)
    strata = {
        "gepiii_overlap": stratum_summary(
            featured=featured,
            scores=scores,
            mask=overlap_mask,
            score_summary_name="gepiii_overlap",
        ),
        "bdg2_only": stratum_summary(
            featured=featured,
            scores=scores,
            mask=~overlap_mask,
            score_summary_name="bdg2_only",
        ),
    }
    source = source_detector_reference()
    result = {
        "schema_version": 1,
        "experiment": "phaseE_step3_5_smoke_attribution",
        "input_smoke_json": str(args.smoke_out),
        "site_id": site,
        "meter": args.meter,
        "variant": "cleaned",
        "rows": int(len(featured)),
        "metric_contract": {
            "path": "unlabeled_score_transfer_attribution",
            "bdg2_ground_truth_metrics_reported": False,
            "headline_metric": False,
        },
        "value_change_regime_note": (
            "Source detector uses M3 row_offset; BDG2 scoring uses "
            "row_offset_meter_aware. For this single-meter slice they are "
            "semantically equivalent per Step 1 row-by-row tests; multi-meter "
            "full transfer must align train/serve value-change semantics."
        ),
        "strata": strata,
        "source_detector_reference": source,
        "attribution": attribution_call(strata, source),
        "runtime": {
            "score_seconds": float(score_seconds),
            "elapsed_seconds": float(time.perf_counter() - t0),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
