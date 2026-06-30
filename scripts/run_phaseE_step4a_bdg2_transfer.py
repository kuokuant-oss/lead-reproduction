"""Phase E Step 4a BDG2 transfer runner.

Default mode is a two-site pilot. Use ``--mode full`` only after the pilot gate
passes. Outputs are unlabeled score-transfer summaries under ADR 0019.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from phaseE_transfer import (
    BDG2_DIR,
    all_sites,
    fit_gepiii_lightgbm_detector,
    fit_gepiii_seed42_ensemble,
    json_clean,
    load_bdg2_scoring_frame,
    log,
    m3_primary_use_mapping,
    multi_building_transfer_stability,
    pilot_sites,
    prepare_bdg2_features,
    predict_scores,
    schema_summary,
    selected_site_buildings,
    site_building_summary,
    stratified_score_report,
)


OUT = Path(".scratch/phaseE-step4a-bdg2-transfer-pilot.json")
ENTRY_METER_CHOICES = ["electricity", "chilledwater"]
ENTRY_METER_DEFAULT = "electricity"
SCORE_UPLIFT_RATIO = 3.0
OOD_DISTRIBUTION_RATIO = 2.0
OOD_MISSING_DELTA = 0.10


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
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument(
        "--sites",
        nargs="*",
        default=None,
        help="Explicit BDG2 site ids. Defaults to Fox + one BDG2-only-rich site in pilot.",
    )
    parser.add_argument(
        "--detector",
        choices=["ensemble", "lightgbm"],
        default="ensemble",
        help="Primary transfer detector. The plan uses ensemble; LightGBM is a sidecar.",
    )
    parser.add_argument(
        "--include-cleaned",
        action="store_true",
        help="Also score cleaned as bridge/sensitivity companion.",
    )
    return parser.parse_args()


def detector_for(name: str) -> dict[str, Any]:
    if name == "ensemble":
        return fit_gepiii_seed42_ensemble()
    if name == "lightgbm":
        return fit_gepiii_lightgbm_detector()
    raise ValueError(f"Unknown detector: {name}")


def lightgbm_sidecar(detector: dict[str, Any]) -> dict[str, Any]:
    if detector["kind"] == "single_model":
        return detector
    model = detector["models"]["lightgbm"]
    summary = dict(detector["source_summary"])
    summary.update(
        {
            "detector": "m3_4_seed42_lightgbm_member_control_anchor",
            "parent_detector": detector["source_summary"]["detector"],
        }
    )
    return {
        "kind": "single_model",
        "model": model,
        "scaler": detector["scaler"],
        "feature_cols": detector["feature_cols"],
        "source_summary": summary,
    }


def choose_sites(args: argparse.Namespace) -> list[str]:
    if args.sites:
        return [str(site) for site in args.sites]
    if args.mode == "pilot":
        return pilot_sites(args.bdg2_dir, meter=args.meter)
    return all_sites(args.bdg2_dir, meter=args.meter)


def score_site_variant(
    *,
    bdg2_dir: Path,
    detector: dict[str, Any],
    primary_use_mapping: dict[str, int],
    meter: str,
    site: str,
    variant: str,
) -> dict[str, Any]:
    site_id, buildings = selected_site_buildings(bdg2_dir, meter=meter, site=site)
    load_t0 = time.perf_counter()
    frame = load_bdg2_scoring_frame(
        bdg2_dir=bdg2_dir,
        variant=variant,
        meter_types=[meter],
        building_ids=buildings,
        include_weather=True,
    )
    load_seconds = time.perf_counter() - load_t0
    feature_t0 = time.perf_counter()
    featured = prepare_bdg2_features(
        frame,
        meter=meter,
        primary_use_mapping=primary_use_mapping,
        feature_cols=detector["feature_cols"],
    )
    feature_seconds = time.perf_counter() - feature_t0
    score_t0 = time.perf_counter()
    scores = predict_scores(detector, featured)
    score_seconds = time.perf_counter() - score_t0
    return {
        "site_id": site_id,
        "variant": variant,
        "buildings_requested": int(len(buildings)),
        "schema": schema_summary(frame),
        "feature_regime": "offline",
        "value_change_regime": "row_offset_meter_aware",
        "single_meter_value_change_equivalence": (
            f"For {meter}-only scoring, row_offset_meter_aware preserves the "
            "M3 row_offset semantics without crossing meter types."
        ),
        "timing": {
            "load_seconds": float(load_seconds),
            "feature_seconds": float(feature_seconds),
            "score_seconds": float(score_seconds),
            "score_rows_per_second": float(len(featured) / score_seconds)
            if score_seconds > 0
            else None,
        },
        "stratified": stratified_score_report(
            featured=featured,
            scores=scores,
            feature_cols=detector["feature_cols"],
        ),
    }


def median_ratio(
    numerator: dict[str, Any], denominator: dict[str, Any]
) -> float | None:
    top = numerator.get("median")
    bottom = denominator.get("median")
    if top is None or bottom in (None, 0):
        return None
    return float(top / bottom)


def ood_evidence(bdg2: dict[str, Any], overlap: dict[str, Any]) -> dict[str, Any]:
    bdg2_ood = bdg2["ood_summary"]
    overlap_ood = overlap["ood_summary"]
    square_feet_ratio = median_ratio(
        bdg2_ood["square_feet_distribution"],
        overlap_ood["square_feet_distribution"],
    )
    meter_reading_ratio = median_ratio(
        bdg2_ood["meter_reading_distribution"],
        overlap_ood["meter_reading_distribution"],
    )
    bdg2_missing = bdg2_ood.get("model_feature_missing_rate")
    overlap_missing = overlap_ood.get("model_feature_missing_rate")
    model_missing_delta = (
        float(bdg2_missing - overlap_missing)
        if bdg2_missing is not None and overlap_missing is not None
        else None
    )
    primary_use_unseen_delta = float(
        bdg2_ood.get("primary_use_unseen_rate", 0.0)
        - overlap_ood.get("primary_use_unseen_rate", 0.0)
    )
    ratio_flags = [
        value is not None
        and (value >= OOD_DISTRIBUTION_RATIO or value <= 1 / OOD_DISTRIBUTION_RATIO)
        for value in [square_feet_ratio, meter_reading_ratio]
    ]
    return {
        "square_feet_median_ratio_bdg2_vs_overlap": square_feet_ratio,
        "meter_reading_median_ratio_bdg2_vs_overlap": meter_reading_ratio,
        "model_feature_missing_rate_delta_bdg2_minus_overlap": model_missing_delta,
        "primary_use_unseen_rate_delta_bdg2_minus_overlap": primary_use_unseen_delta,
        "ood_signal": bool(
            any(ratio_flags)
            or (
                model_missing_delta is not None
                and abs(model_missing_delta) >= OOD_MISSING_DELTA
            )
            or abs(primary_use_unseen_delta) >= OOD_MISSING_DELTA
        ),
    }


def pilot_gate(site_results: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    raw_reports: list[dict[str, Any]] = []
    bdg2_sufficient_sites: list[dict[str, Any]] = []
    overlap_sufficient_sites: list[dict[str, Any]] = []
    sufficient_comparisons: list[dict[str, Any]] = []
    stability_by_site: dict[str, Any] = {}
    for result in site_results:
        if result["variant"] != "raw":
            continue
        raw_reports.append(result)
        stratified = result["stratified"]
        coverage = stratified["all"]["score_summary"].get("score_coverage", 0.0)
        if coverage < 1.0:
            failures.append(f"{result['site_id']} raw score coverage {coverage}")
        completeness = stratified["completeness_strata"]
        bdg2_sufficient = completeness["bdg2_only__sufficient_obs"]
        overlap_sufficient = completeness["gepiii_overlap__sufficient_obs"]
        stability_by_site[result["site_id"]] = {
            "bdg2_only__sufficient_obs": multi_building_transfer_stability(
                bdg2_sufficient
            ),
            "gepiii_overlap__sufficient_obs": multi_building_transfer_stability(
                overlap_sufficient
            ),
        }
        if (
            int(bdg2_sufficient.get("buildings", 0)) > 0
            and int(
                bdg2_sufficient.get(
                    "rows", bdg2_sufficient.get("score_summary", {}).get("rows", 0)
                )
            )
            > 0
        ):
            bdg2_sufficient_sites.append(
                {"site_id": result["site_id"], **bdg2_sufficient}
            )
        if (
            int(overlap_sufficient.get("buildings", 0)) > 0
            and int(
                overlap_sufficient.get(
                    "rows", overlap_sufficient.get("score_summary", {}).get("rows", 0)
                )
            )
            > 0
        ):
            overlap_sufficient_sites.append(
                {"site_id": result["site_id"], **overlap_sufficient}
            )
        bdg2_median = bdg2_sufficient["score_summary"].get("score_median")
        overlap_median = overlap_sufficient["score_summary"].get("score_median")
        if bdg2_median is not None and overlap_median is not None:
            ood = ood_evidence(bdg2_sufficient, overlap_sufficient)
            sufficient_comparisons.append(
                {
                    "site_id": result["site_id"],
                    "bdg2_only_sufficient_median": bdg2_median,
                    "gepiii_overlap_sufficient_median": overlap_median,
                    "median_delta_bdg2_minus_overlap": bdg2_median - overlap_median,
                    "median_ratio_bdg2_vs_overlap": bdg2_median / overlap_median
                    if overlap_median
                    else None,
                    "multi_building_transfer_stability": {
                        "bdg2_only__sufficient_obs": multi_building_transfer_stability(
                            bdg2_sufficient
                        ),
                        "gepiii_overlap__sufficient_obs": multi_building_transfer_stability(
                            overlap_sufficient
                        ),
                    },
                    "ood_evidence": ood,
                }
            )
    verdict = "within_context_evidence_available"
    allowed_next_step = "within_context_packet_path"
    if failures:
        verdict = "plumbing_failed"
        allowed_next_step = "stop_and_diagnose"
    elif not bdg2_sufficient_sites:
        verdict = "no_bdg2_only_sufficient_obs"
        allowed_next_step = "stop_and_report"
        failures.append("pilot has no bdg2_only__sufficient_obs evidence")
    else:
        uplift = [
            item
            for item in sufficient_comparisons
            if item["median_ratio_bdg2_vs_overlap"] is not None
            and item["median_ratio_bdg2_vs_overlap"] > SCORE_UPLIFT_RATIO
        ]
        if uplift:
            verdict = (
                "within_context_evidence_available_with_ood_signal"
                if any(item["ood_evidence"]["ood_signal"] for item in uplift)
                else "within_context_evidence_available_with_score_uplift"
            )
    status = "passed" if allowed_next_step == "within_context_packet_path" else "failed"
    return {
        "status": status,
        "verdict": verdict,
        "failures": failures,
        "allowed_next_step": allowed_next_step,
        "raw_sites_checked": [result["site_id"] for result in raw_reports],
        "bdg2_only_sufficient_obs_sites": [
            item["site_id"] for item in bdg2_sufficient_sites
        ],
        "gepiii_overlap_sufficient_obs_sites": [
            item["site_id"] for item in overlap_sufficient_sites
        ],
        "multi_building_transfer_stability": stability_by_site,
        "sufficient_obs_comparisons": sufficient_comparisons,
        "note": (
            "This gate checks whether the pilot contains BDG2-only "
            "sufficient-observation evidence for within-context packets. The "
            "multi-building powered bar is reported as confidence only, not as "
            "a blocking entry gate. It is not an accuracy or readiness gate."
        ),
        "parameters": {
            "score_uplift_ratio": SCORE_UPLIFT_RATIO,
            "ood_distribution_ratio": OOD_DISTRIBUTION_RATIO,
            "ood_missing_delta": OOD_MISSING_DELTA,
        },
    }


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    sites = choose_sites(args)
    log(f"Phase E Step 4a mode={args.mode} meter={args.meter} sites={sites}")
    detector = detector_for(args.detector)
    primary_use_mapping = m3_primary_use_mapping()
    variants = ["raw"]
    if args.include_cleaned:
        variants.append("cleaned")
    site_results = []
    for site in sites:
        for variant in variants:
            log(f"Scoring site={site} variant={variant}")
            site_results.append(
                score_site_variant(
                    bdg2_dir=args.bdg2_dir,
                    detector=detector,
                    primary_use_mapping=primary_use_mapping,
                    meter=args.meter,
                    site=site,
                    variant=variant,
                )
            )
    control_anchor = None
    if args.mode == "pilot":
        control_anchor = score_site_variant(
            bdg2_dir=args.bdg2_dir,
            detector=lightgbm_sidecar(detector),
            primary_use_mapping=primary_use_mapping,
            meter=args.meter,
            site="Fox",
            variant="cleaned",
        )

    result = {
        "schema_version": 1,
        "experiment": "phaseE_step4a_bdg2_transfer",
        "mode": args.mode,
        "adr": "0019-bdg2-evaluation-paradigm",
        "metric_contract": {
            "path": "unlabeled_score_transfer",
            "bdg2_ground_truth_metrics_reported": False,
            "raw_cleaned_pseudo_label_metrics_reported": False,
            "headline_metric": args.mode == "full",
            "headline_scope_rule": (
                "Any headline must include BDG2-only buildings or held-out BDG2 "
                "sites; GEPIII-overlap rows are bridge/calibration evidence only."
            ),
        },
        "selection": {
            "meter": args.meter,
            "scored_variants": variants,
            "primary_variant": "raw",
            "detector": detector["source_summary"]["detector"],
            "tabpfn_status": "not_used_in_step4a_full_score_cost_tradeoff",
            "site_ids": sites,
            "site_selection_table": site_building_summary(
                args.bdg2_dir, meter=args.meter
            ).to_dict(orient="records"),
        },
        "detector_source": detector["source_summary"],
        "control_anchor": control_anchor,
        "site_results": site_results,
        "pilot_gate": pilot_gate(site_results) if args.mode == "pilot" else None,
        "interpretation_boundary": (
            "Scores are unlabeled transfer outputs. Absolute scores are not "
            "calibrated BDG2 risk; missingness/OOD summaries must travel with "
            "every score stratum per unknown #26."
        ),
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(json_clean(result), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    log(f"Saved {args.out}")


if __name__ == "__main__":
    main()
