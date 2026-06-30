"""Phase E Step 4b small-slice TabPFN-vs-GBDT BDG2 rank comparison.

This is an unlabeled BDG2 query comparison. It reports score/rank agreement and
latency, not BDG2 accuracy. TabPFN runs only with local execution settings.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lead import load_bdg2_frame
from phaseE_transfer import (
    BDG2_DIR,
    RANDOM_STATE,
    completeness_label,
    fit_gepiii_lightgbm_detector,
    json_clean,
    log,
    m3_primary_use_mapping,
    prepare_bdg2_features,
    predict_scores,
    score_summary,
    selected_site_buildings,
    stratified_score_report,
)


OUT = Path(".scratch/phaseE-step4b-tabpfn-vs-gbdt-bdg2.json")
DEFAULT_MODEL_PATH = Path(".tabpfn-cache/tabpfn-v3-classifier-v3_default.ckpt")
ENTRY_METER_CHOICES = ["electricity", "chilledwater"]
ENTRY_METER_DEFAULT = "electricity"


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
    parser.add_argument("--site", default="Fox")
    parser.add_argument("--variant", default="raw", choices=["raw", "cleaned"])
    parser.add_argument("--max-context-rows", type=int, default=1000)
    parser.add_argument("--max-query-rows", type=int, default=1000)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--skip-tabpfn", action="store_true")
    parser.add_argument("--allow-tabpfn-failure", action="store_true")
    return parser.parse_args()


def tabpfn_classifier(*, model_path: Path | None = None) -> Any:
    from tabpfn import TabPFNClassifier

    kwargs: dict[str, Any] = {}
    if model_path is not None:
        kwargs["model_path"] = str(model_path)
    try:
        return TabPFNClassifier(**kwargs)
    except TypeError:
        return TabPFNClassifier()


def bounded_context(
    x_train: np.ndarray,
    y_fit: pd.Series,
    *,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, pd.Series]:
    rng = np.random.default_rng(seed)
    y = y_fit.reset_index(drop=True)
    pos_idx = y[y == 1].index.to_numpy()
    neg_idx = y[y == 0].index.to_numpy()
    per_class = max(1, max_rows // 2)
    take_pos = min(per_class, len(pos_idx))
    take_neg = min(max_rows - take_pos, len(neg_idx))
    chosen = np.concatenate(
        [
            rng.choice(pos_idx, size=take_pos, replace=False),
            rng.choice(neg_idx, size=take_neg, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return x_train[chosen], y.iloc[chosen].reset_index(drop=True)


def bounded_query(featured: pd.DataFrame, *, max_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    overlap_mask = featured["is_gepiii_overlap"].astype(bool)
    for mask in [~overlap_mask, overlap_mask]:
        subset = featured.loc[mask]
        if subset.empty:
            continue
        take = min(max_rows // 2, len(subset))
        parts.append(
            subset.sample(n=take, random_state=int(rng.integers(0, 1_000_000)))
        )
    if not parts:
        return featured.head(0)
    query = pd.concat(parts).sort_values(["building_id", "timestamp"])
    if len(query) > max_rows:
        query = query.sample(n=max_rows, random_state=seed)
    return query


def rank_agreement(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    frame = pd.DataFrame({"gbdt_score": a, "tabpfn_score": b}).replace(
        [np.inf, -np.inf], np.nan
    )
    frame = frame.dropna()
    if frame.empty:
        return {"rows": 0, "spearman": None, "top_decile_overlap": None}
    spearman = frame["gbdt_score"].corr(frame["tabpfn_score"], method="spearman")
    n_top = max(1, int(np.ceil(len(frame) * 0.1)))
    g_top = set(frame["gbdt_score"].nlargest(n_top).index)
    t_top = set(frame["tabpfn_score"].nlargest(n_top).index)
    return {
        "rows": int(len(frame)),
        "spearman": float(spearman) if pd.notna(spearman) else None,
        "top_decile_overlap": float(len(g_top & t_top) / n_top),
    }


def rank_agreement_by_stratum(
    featured: pd.DataFrame, gbdt_scores: np.ndarray, tabpfn_scores: np.ndarray
) -> dict[str, Any]:
    completeness = completeness_label(featured)
    strata: dict[str, Any] = {
        "all": rank_agreement(gbdt_scores, tabpfn_scores),
    }
    overlap = featured["is_gepiii_overlap"].astype(bool)
    for overlap_name, overlap_value in [
        ("gepiii_overlap", True),
        ("bdg2_only", False),
    ]:
        for completeness_name in ["sufficient_obs", "high_missing"]:
            mask = (overlap == overlap_value) & (completeness == completeness_name)
            key = f"{overlap_name}__{completeness_name}"
            strata[key] = rank_agreement(
                gbdt_scores[mask.to_numpy()], tabpfn_scores[mask.to_numpy()]
            )
    return strata


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    if args.model_path:
        os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(args.model_path.parent))
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

    detector = fit_gepiii_lightgbm_detector()
    site, buildings = selected_site_buildings(
        args.bdg2_dir, meter=args.meter, site=args.site
    )
    frame = load_bdg2_frame(
        bdg2_dir=args.bdg2_dir,
        variant=args.variant,
        meter_types=[args.meter],
        building_ids=buildings,
        include_weather=True,
    )
    featured = prepare_bdg2_features(
        frame,
        meter=args.meter,
        primary_use_mapping=m3_primary_use_mapping(),
        feature_cols=detector["feature_cols"],
    )
    query = bounded_query(featured, max_rows=args.max_query_rows, seed=RANDOM_STATE)
    query_x = detector["scaler"].transform(query[detector["feature_cols"]])
    gbdt_t0 = time.perf_counter()
    gbdt_scores = predict_scores(detector, query)
    gbdt_seconds = time.perf_counter() - gbdt_t0

    tabpfn: dict[str, Any]
    if args.skip_tabpfn:
        tabpfn = {"status": "skipped", "reason": "--skip-tabpfn"}
    else:
        try:
            from phaseE_transfer import m3_source_table

            source = m3_source_table()
            context_x, context_y = bounded_context(
                source["x_train"],
                source["y_fit"],
                max_rows=args.max_context_rows,
                seed=RANDOM_STATE,
            )
            model_path = (
                args.model_path
                if args.model_path and args.model_path.exists()
                else None
            )
            model = tabpfn_classifier(model_path=model_path)
            fit_t0 = time.perf_counter()
            model.fit(context_x, context_y)
            fit_seconds = time.perf_counter() - fit_t0
            pred_t0 = time.perf_counter()
            tabpfn_scores = model.predict_proba(query_x)[:, 1]
            predict_seconds = time.perf_counter() - pred_t0
            tabpfn = {
                "status": "completed",
                "context_rows": int(len(context_y)),
                "query_rows": int(len(query_x)),
                "model_path": str(model_path) if model_path is not None else None,
                "model_path_exists": bool(args.model_path.exists())
                if args.model_path
                else False,
                "fit_seconds": float(fit_seconds),
                "predict_proba_seconds": float(predict_seconds),
                "predict_ms_per_row": float(predict_seconds * 1000 / len(query_x))
                if len(query_x)
                else None,
                "score_summary": score_summary(
                    tabpfn_scores, np.ones(len(tabpfn_scores), dtype=bool)
                ),
                "rank_agreement_with_gbdt": rank_agreement(gbdt_scores, tabpfn_scores),
                "rank_agreement_with_gbdt_by_stratum": rank_agreement_by_stratum(
                    query, gbdt_scores, tabpfn_scores
                ),
            }
        except Exception as exc:
            if not args.allow_tabpfn_failure:
                raise
            tabpfn = {"status": "failed", "error": repr(exc)}

    result = {
        "schema_version": 1,
        "experiment": "phaseE_step4b_tabpfn_vs_gbdt_bdg2_small_slice",
        "adr": "0019-bdg2-evaluation-paradigm",
        "metric_contract": {
            "path": "unlabeled_rank_comparison",
            "bdg2_ground_truth_metrics_reported": False,
            "headline_metric": False,
            "forbidden_metric_keys_absent": [
                "roc_auc",
                "pr_auc",
                "precision",
                "recall",
                "f1",
            ],
        },
        "selection": {
            "site_id": site,
            "meter": args.meter,
            "variant": args.variant,
            "query_rows": int(len(query)),
            "max_context_rows": int(args.max_context_rows),
            "max_query_rows": int(args.max_query_rows),
            "tabpfn_scope": "small_unlabeled_query_slice_only",
        },
        "gbdt": {
            "detector": detector["source_summary"],
            "score_seconds": float(gbdt_seconds),
            "predict_ms_per_row": float(gbdt_seconds * 1000 / len(query))
            if len(query)
            else None,
            "score_summary": score_summary(
                gbdt_scores, np.ones(len(gbdt_scores), dtype=bool)
            ),
            "stratified": stratified_score_report(
                featured=query,
                scores=gbdt_scores,
                feature_cols=detector["feature_cols"],
            ),
        },
        "tabpfn": tabpfn,
        "interpretation_boundary": (
            "This compares score rankings on unlabeled BDG2 query rows. It does "
            "not report BDG2 accuracy or readiness. Rank agreement must be read "
            "by completeness and overlap stratum, not as a mixed headline."
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
