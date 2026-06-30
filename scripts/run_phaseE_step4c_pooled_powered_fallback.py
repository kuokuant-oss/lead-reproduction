"""Phase E Step 4c pooled meter fallback.

This runner pools raw BDG2 score-transfer evidence across sites
only to test whether the sufficient-observation BDG2-only stratum becomes
powered. It is not a full-transfer headline or BDG2 ground-truth evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lead import load_bdg2_frame
from phaseE_transfer import (
    BDG2_DIR,
    MIN_STRATUM_BUILDINGS,
    MIN_STRATUM_ROWS,
    all_sites,
    completeness_label,
    distribution,
    fit_gepiii_seed42_ensemble,
    json_clean,
    log,
    m3_primary_use_mapping,
    prepare_bdg2_features,
    predict_scores,
    schema_summary,
    score_summary,
    selected_site_buildings,
    site_building_summary,
)
from run_phaseE_step4a_bdg2_transfer import (
    ENTRY_METER_CHOICES,
    ENTRY_METER_DEFAULT,
    SCORE_UPLIFT_RATIO,
    lightgbm_sidecar,
    pilot_gate,
    score_site_variant,
)


OUT = Path(".scratch/phaseE-step4c-pooled-powered-fallback.json")
STRATUM_KEYS = [
    "all",
    "gepiii_overlap",
    "bdg2_only",
    "gepiii_overlap__sufficient_obs",
    "gepiii_overlap__high_missing",
    "bdg2_only__sufficient_obs",
    "bdg2_only__high_missing",
]
MISSING_RATE_COLUMNS = [
    "meter_reading",
    "log_square_feet",
    "lag_value_diff_1",
    "lag_value_ratio_1",
    "lag_value_diff_24",
    "lag_value_ratio_24",
    "lag_value_diff_168",
    "lag_value_ratio_168",
]


@dataclass
class PooledStratum:
    rows: int = 0
    buildings: set[str] = field(default_factory=set)
    scores: list[np.ndarray] = field(default_factory=list)
    square_feet: list[np.ndarray] = field(default_factory=list)
    meter_reading: list[np.ndarray] = field(default_factory=list)
    primary_use_unseen: int = 0
    model_missing_values: int = 0
    model_total_values: int = 0
    feature_missing_values: dict[str, int] = field(default_factory=dict)
    feature_total_values: dict[str, int] = field(default_factory=dict)

    def add(
        self,
        *,
        featured: pd.DataFrame,
        scores: np.ndarray,
        mask: pd.Series | np.ndarray,
        feature_cols: list[str],
    ) -> None:
        mask_array = (
            mask.to_numpy() if isinstance(mask, pd.Series) else np.asarray(mask)
        )
        subset = featured.loc[mask_array]
        self.rows += int(len(subset))
        if subset.empty:
            return

        self.buildings.update(subset["building_id"].astype(str).unique().tolist())
        self.scores.append(np.asarray(scores[mask_array], dtype="float64"))
        self.square_feet.append(
            pd.to_numeric(subset["square_feet"], errors="coerce").to_numpy(
                dtype="float64"
            )
        )
        self.meter_reading.append(
            pd.to_numeric(subset["meter_reading"], errors="coerce").to_numpy(
                dtype="float64"
            )
        )
        self.primary_use_unseen += int((subset["primary_use_enc"] < 0).sum())

        model_missing = subset[feature_cols].isna()
        self.model_missing_values += int(model_missing.to_numpy().sum())
        self.model_total_values += int(model_missing.size)

        for column in MISSING_RATE_COLUMNS:
            if column not in subset.columns:
                continue
            values = subset[column]
            self.feature_missing_values[column] = self.feature_missing_values.get(
                column, 0
            ) + int(values.isna().sum())
            self.feature_total_values[column] = self.feature_total_values.get(
                column, 0
            ) + int(len(values))

    def to_report(self) -> dict[str, Any]:
        scores = concatenate_or_empty(self.scores)
        square_feet = concatenate_or_empty(self.square_feet)
        meter_reading = concatenate_or_empty(self.meter_reading)
        return {
            "score_summary": score_summary(scores, np.ones(len(scores), dtype=bool)),
            "ood_summary": {
                "primary_use_unseen_rate": float(self.primary_use_unseen / self.rows)
                if self.rows
                else 0.0,
                "feature_missing_rates": {
                    column: float(
                        self.feature_missing_values[column]
                        / self.feature_total_values[column]
                    )
                    for column in sorted(self.feature_missing_values)
                    if self.feature_total_values[column]
                },
                "model_feature_missing_rate": float(
                    self.model_missing_values / self.model_total_values
                )
                if self.model_total_values
                else None,
                "square_feet_distribution": distribution(pd.Series(square_feet)),
                "meter_reading_distribution": distribution(pd.Series(meter_reading)),
            },
            "buildings": int(len(self.buildings)),
            "rows": int(self.rows),
        }


def concatenate_or_empty(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.array([], dtype="float64")
    return np.concatenate(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdg2-dir", type=Path, default=BDG2_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--meter",
        default=ENTRY_METER_DEFAULT,
        choices=ENTRY_METER_CHOICES,
        help=(
            "Entry meter for within-context transfer scoring. Electricity is "
            "the default; chilledwater remains supported for deferred Level-3 "
            "weather-conditioned review."
        ),
    )
    parser.add_argument("--sites", nargs="*", default=None)
    return parser.parse_args()


def empty_cells() -> dict[str, PooledStratum]:
    return {key: PooledStratum() for key in STRATUM_KEYS}


def add_site_to_pool(
    cells: dict[str, PooledStratum],
    *,
    featured: pd.DataFrame,
    scores: np.ndarray,
    feature_cols: list[str],
) -> None:
    overlap = featured["is_gepiii_overlap"].astype(bool)
    completeness = completeness_label(featured)
    masks = {
        "all": pd.Series(True, index=featured.index),
        "gepiii_overlap": overlap,
        "bdg2_only": ~overlap,
        "gepiii_overlap__sufficient_obs": overlap & completeness.eq("sufficient_obs"),
        "gepiii_overlap__high_missing": overlap & completeness.eq("high_missing"),
        "bdg2_only__sufficient_obs": (~overlap) & completeness.eq("sufficient_obs"),
        "bdg2_only__high_missing": (~overlap) & completeness.eq("high_missing"),
    }
    for key, mask in masks.items():
        cells[key].add(
            featured=featured,
            scores=scores,
            mask=mask,
            feature_cols=feature_cols,
        )


def pooled_stratified_report(cells: dict[str, PooledStratum]) -> dict[str, Any]:
    report = {key: cells[key].to_report() for key in STRATUM_KEYS}
    overlap_median = report["gepiii_overlap"]["score_summary"].get("score_median")
    bdg2_only_median = report["bdg2_only"]["score_summary"].get("score_median")
    report["overlap_vs_bdg2_only"] = {
        "median_score_delta_overlap_minus_bdg2_only": (
            overlap_median - bdg2_only_median
            if overlap_median is not None and bdg2_only_median is not None
            else None
        ),
        "interpretation": (
            "Unlabeled pooled score-distribution contrast only; not accuracy, "
            "not readiness, and not calibrated risk."
        ),
    }
    report["completeness_strata"] = {
        key: report[key]
        for key in [
            "gepiii_overlap__sufficient_obs",
            "gepiii_overlap__high_missing",
            "bdg2_only__sufficient_obs",
            "bdg2_only__high_missing",
        ]
    }
    return report


def pooled_gate(
    pooled: dict[str, Any], *, meter: str = ENTRY_METER_DEFAULT
) -> dict[str, Any]:
    gate = pilot_gate(
        [
            {
                "variant": "raw",
                "site_id": f"pooled_{meter}",
                "stratified": pooled,
            }
        ]
    )
    gate["source_gate_verdict"] = gate["verdict"]
    if gate["verdict"] == "no_bdg2_only_sufficient_obs":
        gate["verdict"] = "no_pooled_bdg2_only_sufficient_obs"
        gate["allowed_next_step"] = "stop_and_report"
        gate["failures"] = [
            failure.replace("pilot has", f"pooled {meter} has")
            for failure in gate["failures"]
        ]
    elif gate["allowed_next_step"] == "within_context_packet_path":
        gate["status"] = "passed"
        gate["allowed_next_step"] = "within_context_packet_path"
    gate["note"] = (
        "Pooled fallback checks cross-site raw score-transfer strata only. It "
        "reports multi-building transfer stability as confidence metadata; it "
        "does not authorize readiness or BDG2 ground-truth claims."
    )
    gate["parameters"].update(
        {
            "min_stratum_buildings": MIN_STRATUM_BUILDINGS,
            "min_stratum_rows": MIN_STRATUM_ROWS,
            "score_uplift_ratio": SCORE_UPLIFT_RATIO,
        }
    )
    return gate


def score_sites(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    sites = args.sites or all_sites(args.bdg2_dir, meter=args.meter)
    log(f"Phase E Step 4c pooled fallback meter={args.meter} sites={sites}")
    detector = fit_gepiii_seed42_ensemble()
    primary_use_mapping = m3_primary_use_mapping()
    cells = empty_cells()
    site_summaries: list[dict[str, Any]] = []

    for site in sites:
        site_id, buildings = selected_site_buildings(
            args.bdg2_dir, meter=args.meter, site=site
        )
        log(f"Scoring pooled fallback site={site_id}")
        site_t0 = time.perf_counter()
        frame = load_bdg2_frame(
            bdg2_dir=args.bdg2_dir,
            variant="raw",
            meter_types=[args.meter],
            building_ids=buildings,
            include_weather=True,
        )
        load_seconds = time.perf_counter() - site_t0
        feature_t0 = time.perf_counter()
        featured = prepare_bdg2_features(
            frame,
            meter=args.meter,
            primary_use_mapping=primary_use_mapping,
            feature_cols=detector["feature_cols"],
        )
        feature_seconds = time.perf_counter() - feature_t0
        score_t0 = time.perf_counter()
        scores = predict_scores(detector, featured)
        score_seconds = time.perf_counter() - score_t0
        add_site_to_pool(
            cells,
            featured=featured,
            scores=scores,
            feature_cols=detector["feature_cols"],
        )
        site_summaries.append(
            {
                "site_id": site_id,
                "buildings_requested": int(len(buildings)),
                "schema": schema_summary(frame),
                "timing": {
                    "load_seconds": float(load_seconds),
                    "feature_seconds": float(feature_seconds),
                    "score_seconds": float(score_seconds),
                    "score_rows_per_second": float(len(featured) / score_seconds)
                    if score_seconds > 0
                    else None,
                },
            }
        )

    pooled = pooled_stratified_report(cells)
    return {
        "schema_version": 1,
        "experiment": "phaseE_step4c_pooled_powered_fallback",
        "adr": "0019-bdg2-evaluation-paradigm",
        "metric_contract": {
            "path": "unlabeled_score_transfer_pooled_fallback",
            "bdg2_ground_truth_metrics_reported": False,
            "raw_cleaned_pseudo_label_metrics_reported": False,
            "headline_metric": False,
            "headline_scope_rule": (
                "This pooled fallback is a diagnostic stratum-power check, not "
                "a full-transfer headline or readiness gate."
            ),
        },
        "selection": {
            "meter": args.meter,
            "variant": "raw",
            "detector": detector["source_summary"]["detector"],
            "site_ids": sites,
            "site_selection_table": site_building_summary(
                args.bdg2_dir, meter=args.meter
            ).to_dict(orient="records"),
        },
        "detector_source": detector["source_summary"],
        "site_summaries": site_summaries,
        "pooled_stratified": pooled,
        "pooled_gate": pooled_gate(pooled, meter=args.meter),
        "control_anchor": score_site_variant(
            bdg2_dir=args.bdg2_dir,
            detector=lightgbm_sidecar(detector),
            primary_use_mapping=primary_use_mapping,
            meter=args.meter,
            site="Fox",
            variant="cleaned",
        ),
        "interpretation_boundary": (
            "Rows are pooled only after site-level raw scoring. This artifact "
            "contains no BDG2 labels, no supervised BDG2 metrics, and no full "
            "transfer success/readiness claim."
        ),
        "elapsed_seconds": float(time.perf_counter() - t0),
    }


def main() -> None:
    args = parse_args()
    result = score_sites(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(json_clean(result), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    log(f"Saved {args.out}")


if __name__ == "__main__":
    main()
