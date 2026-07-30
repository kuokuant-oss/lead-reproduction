"""CPU-only E0 characterization of the M5 F4 meter-specific learner gap.

This is deliberately a reader of frozen full-holdout predictions.  It never
fits or scores a model, and it does not read the independent-query artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.metrics import average_precision_score, roc_auc_score

try:  # supports both ``python scripts/...`` and package imports in tests
    from scripts._research_checkpoint import (
        CheckpointError,
        ResearchCheckpointStore,
        atomic_write_json,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script entry point
    from _research_checkpoint import (
        CheckpointError,
        ResearchCheckpointStore,
        atomic_write_json,
    )

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = PROC / "m5_meter_specific_learner_gap"
CONTEXTS = (5_000, 10_000, 20_000, 50_000, 100_000)
MODELS = ("tabpfn", "trees")
METER_NAMES = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}
PRIMARY_GAP_METRICS = ("pr_auc", "roc_auc")
EXPECTED_VALIDATION_INTERRUPTION_EXIT = 75
# Human authorization label: AUTHORIZE E0 FORMAL RUN.
FORMAL_AUTHORIZATION_TOKEN = "AUTHORIZE_E0_FORMAL_RUN"
FORMAL_BOOTSTRAP_DRAWS_PER_METER = 1_000
FORMAL_TRANCHE_DRAWS_PER_METER = 42


def prediction_path(model: str, context: int) -> Path:
    if model == "tabpfn":
        suffix = (
            "n8_predictions.npz"
            if context == 100_000
            else f"context{context}_n8_predictions.npz"
        )
        return PROC / f"m5_tabpfn_137_full_test_{suffix}"
    return PROC / f"m5_tree_ensemble_f137_context{context}_predictions.npz"


def array_digest(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype="int64").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_metadata(raw_index: np.ndarray) -> pd.DataFrame:
    """Load the frozen raw meter/building fields in prediction-row order."""
    raw = pd.read_csv(
        ROOT / "data" / "raw" / "m3" / "train.csv",
        usecols=["building_id", "meter"],
        dtype={"building_id": "int16", "meter": "int8"},
    )
    if raw_index.min() < 0 or raw_index.max() >= len(raw):
        raise AssertionError("raw_index exceeds M3 training frame")
    selected = raw.iloc[raw_index].reset_index(drop=True)
    if not set(selected["meter"].unique()).issubset(METER_NAMES):
        raise AssertionError("unexpected meter identity in frozen holdout")
    return selected


def audit_payload(
    *,
    path: Path,
    model: str,
    context: int,
    metadata: dict[str, np.ndarray],
    score: np.ndarray,
    meter: np.ndarray,
) -> dict[str, object]:
    return {
        "model": model,
        "context_rows": context,
        "path": str(path.relative_to(ROOT)),
        "sha256": file_digest(path),
        "rows": len(score),
        "raw_index_digest": array_digest(metadata["raw_index"]),
        "label_digest": array_digest(metadata["anomaly"]),
        "building_digest": array_digest(metadata["building_id"]),
        "site_digest": array_digest(metadata["site_id"]),
        "meter_digest": array_digest(meter),
        "finite_scores": int(np.isfinite(score).sum()),
        "nonfinite_scores": int((~np.isfinite(score)).sum()),
        "score_min": float(score.min()),
        "score_max": float(score.max()),
        "score_mean": float(score.mean(dtype="float64")),
    }


def assert_same_identity(
    reference: dict[str, np.ndarray], current: dict[str, np.ndarray], *, source: str
) -> None:
    for field in reference:
        if not np.array_equal(reference[field], current[field]):
            raise AssertionError(f"{source}: {field} identity drift")


def validate_prediction_fields(
    metadata: dict[str, np.ndarray], score: np.ndarray, *, source: str
) -> None:
    if len(np.unique(metadata["raw_index"])) != len(metadata["raw_index"]):
        raise AssertionError(f"{source}: duplicate raw indices")
    if not np.isfinite(score).all():
        raise AssertionError(f"{source}: non-finite scores")


def load_predictions() -> tuple[
    dict[str, dict[int, np.ndarray]], dict[str, np.ndarray], pd.DataFrame, pd.DataFrame
]:
    """Load and hard-fail on any artifact identity drift."""
    all_scores: dict[str, dict[int, np.ndarray]] = {model: {} for model in MODELS}
    reference: dict[str, np.ndarray] | None = None
    raw_metadata: pd.DataFrame | None = None
    audits: list[dict[str, object]] = []
    for model in MODELS:
        for context in CONTEXTS:
            path = prediction_path(model, context)
            if not path.exists():
                raise FileNotFoundError(f"required frozen artifact is absent: {path}")
            with np.load(path) as payload:
                raw_key = "raw_index" if model == "tabpfn" else "validation_raw_index"
                score_key = "tabpfn" if model == "tabpfn" else "ensemble"
                metadata = {
                    "raw_index": np.asarray(payload[raw_key], dtype="int64"),
                    "anomaly": np.asarray(payload["anomaly"], dtype="int8"),
                    "site_id": np.asarray(payload["site_id"], dtype="int8"),
                    "building_id": np.asarray(payload["building_id"], dtype="int16"),
                }
                score = np.asarray(payload[score_key], dtype="float32")
            validate_prediction_fields(metadata, score, source=f"{model}/{context}")
            if reference is None:
                reference = metadata
                raw_metadata = load_raw_metadata(metadata["raw_index"])
                if not np.array_equal(
                    raw_metadata["building_id"].to_numpy(), metadata["building_id"]
                ):
                    raise AssertionError(
                        "raw building IDs differ from frozen prediction artifact"
                    )
            else:
                assert_same_identity(reference, metadata, source=f"{model}/{context}")
            assert raw_metadata is not None
            audits.append(
                audit_payload(
                    path=path,
                    model=model,
                    context=context,
                    metadata=metadata,
                    score=score,
                    meter=raw_metadata["meter"].to_numpy(),
                )
            )
            all_scores[model][context] = score
            print(f"loaded {model}/{context:,}", flush=True)
    assert reference is not None and raw_metadata is not None
    base = pd.DataFrame(reference)
    base["meter"] = raw_metadata["meter"].to_numpy()
    base["meter_name"] = base["meter"].map(METER_NAMES)
    contract = base.groupby("building_id", observed=True)["site_id"].nunique()
    if (contract > 1).any():
        raise AssertionError(
            "building-disjoint contract failed: a building spans sites"
        )
    return all_scores, reference, base, pd.DataFrame(audits)


def percentile_ranks(
    score: np.ndarray, meter: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    global_rank = pd.Series(score).rank(method="average", pct=True).to_numpy("float64")
    within_rank = (
        pd.DataFrame({"score": score, "meter": meter})
        .groupby("meter", sort=False)["score"]
        .rank(method="average", pct=True)
        .to_numpy("float64")
    )
    return global_rank, within_rank


def metric_values(
    y: np.ndarray, score: np.ndarray, global_rank: np.ndarray, within_rank: np.ndarray
) -> dict[str, float]:
    positives = score[y == 1]
    negatives = score[y == 0]
    if not len(positives) or not len(negatives):
        raise ValueError("single-class metric sample")
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "anomaly_mean_score": float(positives.mean(dtype="float64")),
        "anomaly_median_score": float(np.median(positives)),
        "normal_mean_score": float(negatives.mean(dtype="float64")),
        "normal_median_score": float(np.median(negatives)),
        "score_separation": float(
            positives.mean(dtype="float64") - negatives.mean(dtype="float64")
        ),
        "anomaly_global_percentile_rank": float(global_rank[y == 1].mean()),
        "anomaly_within_meter_percentile_rank": float(within_rank[y == 1].mean()),
        "normal_global_percentile_rank": float(global_rank[y == 0].mean()),
        "normal_within_meter_percentile_rank": float(within_rank[y == 0].mean()),
        "anomaly_vs_normal_within_meter_pairwise_auc": float(roc_auc_score(y, score)),
    }


def per_meter_metrics(
    scores: dict[str, dict[int, np.ndarray]], base: pd.DataFrame
) -> tuple[pd.DataFrame, dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, object]] = []
    ranks: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    meter = base["meter"].to_numpy()
    y_all = base["anomaly"].to_numpy()
    for model in MODELS:
        for context in CONTEXTS:
            score = scores[model][context]
            ranks[model, context] = percentile_ranks(score, meter)
            global_rank, within_rank = ranks[model, context]
            for code, name in METER_NAMES.items():
                mask = meter == code
                if not mask.any():
                    continue
                y = y_all[mask]
                values = metric_values(
                    y, score[mask], global_rank[mask], within_rank[mask]
                )
                rows.append(
                    {
                        "model": model,
                        "context_rows": context,
                        "meter": name,
                        "rows": int(mask.sum()),
                        "positives": int(y.sum()),
                        "negatives": int((y == 0).sum()),
                        "prevalence": float(y.mean()),
                        **values,
                    }
                )
    return pd.DataFrame(rows), ranks


def learner_gaps(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in metrics
        if column
        not in {
            "model",
            "context_rows",
            "meter",
            "rows",
            "positives",
            "negatives",
            "prevalence",
        }
    ]
    tab = metrics.loc[metrics["model"] == "tabpfn"].set_index(["meter", "context_rows"])
    tree = metrics.loc[metrics["model"] == "trees"].set_index(["meter", "context_rows"])
    rows = []
    for index in tab.index:
        for metric in metric_columns:
            rows.append(
                {
                    "meter": index[0],
                    "context_rows": index[1],
                    "metric": metric,
                    "tabpfn_minus_trees": float(
                        tab.loc[index, metric] - tree.loc[index, metric]
                    ),
                }
            )
    return pd.DataFrame(rows)


def context_responses(metrics: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for meter in METER_NAMES.values():
        for metric in PRIMARY_GAP_METRICS:
            values = gaps.loc[
                (gaps["meter"] == meter) & (gaps["metric"] == metric)
            ].sort_values("context_rows")
            if values.empty:
                continue
            gap = values["tabpfn_minus_trees"].to_numpy()
            contexts = values["context_rows"].to_numpy(dtype="int64")
            slope = float(np.polyfit(np.log10(contexts), gap, 1)[0])
            rows.append(
                {
                    "meter": meter,
                    "quantity": "learner_gap",
                    "metric": metric,
                    "contrast": "5k_to_100k",
                    "estimate": float(gap[-1] - gap[0]),
                    "slope_log10_context": slope,
                    "endpoint_change": float(gap[-1] - gap[0]),
                }
            )
            for before, after, prior, current in zip(
                contexts[:-1], contexts[1:], gap[:-1], gap[1:], strict=True
            ):
                rows.append(
                    {
                        "meter": meter,
                        "quantity": "learner_gap",
                        "metric": metric,
                        "contrast": f"{before}_to_{after}",
                        "estimate": float(current - prior),
                        "slope_log10_context": np.nan,
                        "endpoint_change": np.nan,
                    }
                )
        for model in MODELS:
            frame = metrics.loc[
                (metrics["meter"] == meter) & (metrics["model"] == model)
            ].sort_values("context_rows")
            if frame.empty:
                continue
            for metric in ("pr_auc", "roc_auc", "anomaly_within_meter_percentile_rank"):
                values = frame[metric].to_numpy()
                contexts = frame["context_rows"].to_numpy(dtype="int64")
                rows.append(
                    {
                        "meter": meter,
                        "quantity": model,
                        "metric": metric,
                        "contrast": "5k_to_100k",
                        "estimate": float(values[-1] - values[0]),
                        "slope_log10_context": float(
                            np.polyfit(np.log10(contexts), values, 1)[0]
                        ),
                        "endpoint_change": float(values[-1] - values[0]),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_metrics(
    scores: dict[str, dict[int, np.ndarray]],
    base: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Cluster-bootstrap frozen rows, with one matched building draw per context/model."""
    rng = np.random.default_rng(seed)
    y = base["anomaly"].to_numpy()
    meter = base["meter"].to_numpy()
    building = base["building_id"].to_numpy()
    rows: list[dict[str, object]] = []
    invalid = {"single_class": 0}
    for code, name in METER_NAMES.items():
        local = np.flatnonzero(meter == code)
        if not len(local):
            continue
        buildings = np.unique(building[local])
        by_building = {
            int(value): local[building[local] == value] for value in buildings
        }
        for draw in range(draws):
            selected_buildings = rng.choice(
                buildings, size=len(buildings), replace=True
            )
            sampled = np.concatenate(
                [by_building[int(value)] for value in selected_buildings]
            )
            y_draw = y[sampled]
            if y_draw.min() == y_draw.max():
                invalid["single_class"] += 1
                continue
            for context in CONTEXTS:
                by_model: dict[str, dict[str, float]] = {}
                for model in MODELS:
                    score = scores[model][context][sampled]
                    # Ranks are re-estimated in the resampled rows, not copied from full holdout.
                    rank = (
                        pd.Series(score)
                        .rank(method="average", pct=True)
                        .to_numpy("float64")
                    )
                    stats = metric_values(y_draw, score, rank, rank)
                    by_model[model] = stats
                    for metric in PRIMARY_GAP_METRICS:
                        rows.append(
                            {
                                "draw": draw,
                                "meter": name,
                                "context_rows": context,
                                "quantity": model,
                                "metric": metric,
                                "estimate": stats[metric],
                            }
                        )
                for metric in PRIMARY_GAP_METRICS:
                    rows.append(
                        {
                            "draw": draw,
                            "meter": name,
                            "context_rows": context,
                            "quantity": "tabpfn_minus_trees",
                            "metric": metric,
                            "estimate": by_model["tabpfn"][metric]
                            - by_model["trees"][metric],
                        }
                    )
            # The same selected-building multiplicities are retained for every context.
            for metric in PRIMARY_GAP_METRICS:
                endpoint = (
                    scores["tabpfn"][100_000][sampled],
                    scores["trees"][100_000][sampled],
                    scores["tabpfn"][5_000][sampled],
                    scores["trees"][5_000][sampled],
                )
                end_gap = (
                    metric_values(
                        y_draw,
                        endpoint[0],
                        np.zeros(len(y_draw)),
                        np.zeros(len(y_draw)),
                    )[metric]
                    - metric_values(
                        y_draw,
                        endpoint[1],
                        np.zeros(len(y_draw)),
                        np.zeros(len(y_draw)),
                    )[metric]
                )
                start_gap = (
                    metric_values(
                        y_draw,
                        endpoint[2],
                        np.zeros(len(y_draw)),
                        np.zeros(len(y_draw)),
                    )[metric]
                    - metric_values(
                        y_draw,
                        endpoint[3],
                        np.zeros(len(y_draw)),
                        np.zeros(len(y_draw)),
                    )[metric]
                )
                rows.append(
                    {
                        "draw": draw,
                        "meter": name,
                        "context_rows": 100_000,
                        "quantity": "learner_gap_change_5k_to_100k",
                        "metric": metric,
                        "estimate": end_gap - start_gap,
                    }
                )
        print(f"bootstrapped {name}: {draws:,} building draws", flush=True)
    return pd.DataFrame(rows), invalid


def bootstrap_intervals(
    draws: pd.DataFrame,
    point_metrics: pd.DataFrame,
    gaps: pd.DataFrame,
    slopes: pd.DataFrame,
) -> pd.DataFrame:
    point_rows: list[dict[str, object]] = []
    for _, row in point_metrics.iterrows():
        if row["model"] in MODELS:
            for metric in PRIMARY_GAP_METRICS:
                point_rows.append(
                    {
                        "meter": row["meter"],
                        "context_rows": row["context_rows"],
                        "quantity": row["model"],
                        "metric": metric,
                        "estimate": row[metric],
                    }
                )
    for _, row in gaps.loc[gaps["metric"].isin(PRIMARY_GAP_METRICS)].iterrows():
        point_rows.append(
            {
                "meter": row["meter"],
                "context_rows": row["context_rows"],
                "quantity": "tabpfn_minus_trees",
                "metric": row["metric"],
                "estimate": row["tabpfn_minus_trees"],
            }
        )
    for _, row in slopes.loc[
        slopes["contrast"].eq("5k_to_100k") & slopes["quantity"].eq("learner_gap")
    ].iterrows():
        point_rows.append(
            {
                "meter": row["meter"],
                "context_rows": 100_000,
                "quantity": "learner_gap_change_5k_to_100k",
                "metric": row["metric"],
                "estimate": row["estimate"],
            }
        )
    point = pd.DataFrame(point_rows)
    summary = draws.groupby(
        ["meter", "context_rows", "quantity", "metric"], as_index=False
    )["estimate"].agg(
        bootstrap_q025=lambda x: x.quantile(0.025),
        bootstrap_q975=lambda x: x.quantile(0.975),
        valid_draws="count",
    )
    return point.merge(
        summary, how="left", on=["meter", "context_rows", "quantity", "metric"]
    )


def leave_one_building(
    scores: dict[str, dict[int, np.ndarray]],
    base: pd.DataFrame,
    ranks: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    gaps: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y = base["anomaly"].to_numpy()
    meter = base["meter"].to_numpy()
    building = base["building_id"].to_numpy()
    for code, name in METER_NAMES.items():
        mask = meter == code
        for building_id in np.unique(building[mask]):
            keep = mask & (building != building_id)
            details = {
                "building_id": int(building_id),
                "building_rows": int((mask & (building == building_id)).sum()),
                "building_positives": int(y[mask & (building == building_id)].sum()),
            }
            for metric in PRIMARY_GAP_METRICS:
                tab_100 = metric_values(
                    y[keep],
                    scores["tabpfn"][100_000][keep],
                    ranks["tabpfn", 100_000][0][keep],
                    ranks["tabpfn", 100_000][1][keep],
                )[metric]
                tree_100 = metric_values(
                    y[keep],
                    scores["trees"][100_000][keep],
                    ranks["trees", 100_000][0][keep],
                    ranks["trees", 100_000][1][keep],
                )[metric]
                tab_5 = metric_values(
                    y[keep],
                    scores["tabpfn"][5_000][keep],
                    ranks["tabpfn", 5_000][0][keep],
                    ranks["tabpfn", 5_000][1][keep],
                )[metric]
                tree_5 = metric_values(
                    y[keep],
                    scores["trees"][5_000][keep],
                    ranks["trees", 5_000][0][keep],
                    ranks["trees", 5_000][1][keep],
                )[metric]
                full_100 = float(
                    gaps.loc[
                        (gaps.meter == name)
                        & (gaps.context_rows == 100_000)
                        & (gaps.metric == metric),
                        "tabpfn_minus_trees",
                    ].iloc[0]
                )
                full_change = float(
                    gaps.loc[
                        (gaps.meter == name)
                        & (gaps.context_rows == 100_000)
                        & (gaps.metric == metric),
                        "tabpfn_minus_trees",
                    ].iloc[0]
                    - gaps.loc[
                        (gaps.meter == name)
                        & (gaps.context_rows == 5_000)
                        & (gaps.metric == metric),
                        "tabpfn_minus_trees",
                    ].iloc[0]
                )
                for endpoint, estimate, full in (
                    ("100k_learner_gap", tab_100 - tree_100, full_100),
                    (
                        "5k_to_100k_learner_gap_change",
                        (tab_100 - tree_100) - (tab_5 - tree_5),
                        full_change,
                    ),
                ):
                    rows.append(
                        {
                            "meter": name,
                            "metric": metric,
                            "endpoint": endpoint,
                            "full_estimate": full,
                            "leave_one_building_estimate": estimate,
                            "influence_delta": estimate - full,
                            **details,
                        }
                    )
            # Positive ranking endpoint is intentionally reported separately from AUC.
            full_rank = float(
                gaps.loc[
                    (gaps.meter == name)
                    & (gaps.context_rows == 100_000)
                    & (gaps.metric == "anomaly_within_meter_percentile_rank"),
                    "tabpfn_minus_trees",
                ].iloc[0]
            )
            rank_estimate = float(
                ranks["tabpfn", 100_000][1][keep & (y == 1)].mean()
                - ranks["trees", 100_000][1][keep & (y == 1)].mean()
            )
            rows.append(
                {
                    "meter": name,
                    "metric": "anomaly_within_meter_percentile_rank",
                    "endpoint": "100k_positive_rank_gap",
                    "full_estimate": full_rank,
                    "leave_one_building_estimate": rank_estimate,
                    "influence_delta": rank_estimate - full_rank,
                    **details,
                }
            )
    return pd.DataFrame(rows)


def segment_concentration(
    allowed_meters: set[str] | None = None, sensitivity_draws: int = 1000
) -> pd.DataFrame:
    """Use E1's frozen segment definition; no rows are resegmented in E0."""
    path = PROC / "m5_context_mechanism_137" / "m5_137_anomaly_segments.parquet"
    phase_path = (
        PROC / "m5_context_mechanism_137" / "m5_137_anomaly_segment_phases.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"frozen E1 segment artifact is absent: {path}")
    segment = pd.read_parquet(path)
    phases = pd.read_parquet(phase_path) if phase_path.exists() else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for meter, group in segment.groupby("meter_name", observed=True):
        if allowed_meters is not None and meter not in allowed_meters:
            continue
        group = group.copy()
        group["learner_gap_score_movement"] = (
            group["tabpfn_score_movement"] - group["tree_score_movement"]
        )
        group["learner_gap_global_rank_movement"] = (
            group["tabpfn_global_rank_movement"] - group["tree_global_rank_movement"]
        )
        for _, value in group.iterrows():
            rows.append(
                {
                    "meter": meter,
                    "model": "paired",
                    "context_scope": "5k_to_100k",
                    "summary": "segment_score_and_rank",
                    "segment_id": int(value.segment_id),
                    "building_id": int(value.building_id),
                    "segments": 1,
                    "rows": int(value.duration_rows),
                    "estimate": float(value.learner_gap_score_movement),
                    "rank_estimate": float(value.learner_gap_global_rank_movement),
                    "contribution_fraction": np.nan,
                    "top_building_concentration": np.nan,
                }
            )
        for model, movement in (
            ("tabpfn", "tabpfn_score_movement"),
            ("trees", "tree_score_movement"),
        ):
            ordered = group.reindex(
                group[movement].abs().sort_values(ascending=False).index
            )
            total = float(ordered[movement].sum())
            for top in (1, 5, 10):
                selected = ordered.head(top)
                rows.append(
                    {
                        "meter": meter,
                        "model": model,
                        "context_scope": "5k_to_100k",
                        "summary": f"top_{top}_absolute_score_movement",
                        "segment_id": pd.NA,
                        "building_id": pd.NA,
                        "segments": int(len(group)),
                        "rows": int(group.duration_rows.sum()),
                        "estimate": float(selected[movement].sum()),
                        "rank_estimate": np.nan,
                        "contribution_fraction": float(selected[movement].sum() / total)
                        if total
                        else np.nan,
                        "top_building_concentration": float(
                            selected.groupby("building_id").size().max() / len(selected)
                        ),
                    }
                )
            rows.append(
                {
                    "meter": meter,
                    "model": model,
                    "context_scope": "5k_to_100k",
                    "summary": "segment_duration_distribution",
                    "segment_id": pd.NA,
                    "building_id": pd.NA,
                    "segments": int(len(group)),
                    "rows": int(group.duration_rows.sum()),
                    "estimate": float(group.duration_rows.median()),
                    "rank_estimate": np.nan,
                    "contribution_fraction": np.nan,
                    "top_building_concentration": float(
                        group.groupby("building_id").size().max() / len(group)
                    ),
                }
            )
        ordered = group.reindex(
            group["learner_gap_score_movement"].abs().sort_values(ascending=False).index
        )
        for top in (1, 5, 10):
            selected = ordered.head(top)
            rows.append(
                {
                    "meter": meter,
                    "model": "tabpfn_minus_trees",
                    "context_scope": "5k_to_100k",
                    "summary": f"top_{top}_absolute_learner_gap_score_movement",
                    "segment_id": pd.NA,
                    "building_id": pd.NA,
                    "segments": int(len(group)),
                    "rows": int(group.duration_rows.sum()),
                    "estimate": float(selected.learner_gap_score_movement.sum()),
                    "rank_estimate": float(
                        selected.learner_gap_global_rank_movement.mean()
                    ),
                    "contribution_fraction": float(
                        selected.learner_gap_score_movement.sum()
                        / group.learner_gap_score_movement.sum()
                    )
                    if group.learner_gap_score_movement.sum()
                    else np.nan,
                    "top_building_concentration": float(
                        selected.groupby("building_id").size().max() / len(selected)
                    ),
                }
            )
        if meter in {"steam", "chilledwater"}:
            rng = np.random.default_rng(20260730)
            values = group.learner_gap_score_movement.to_numpy("float64")
            sampled = values[
                rng.integers(0, len(values), size=(sensitivity_draws, len(values)))
            ].mean(axis=1)
            rows.append(
                {
                    "meter": meter,
                    "model": "tabpfn_minus_trees",
                    "context_scope": "5k_to_100k",
                    "summary": "segment_cluster_sensitivity_score_gap_movement",
                    "segment_id": pd.NA,
                    "building_id": pd.NA,
                    "segments": int(len(group)),
                    "rows": int(group.duration_rows.sum()),
                    "estimate": float(values.mean()),
                    "rank_estimate": np.nan,
                    "contribution_fraction": float(np.quantile(sampled, 0.025)),
                    "top_building_concentration": float(np.quantile(sampled, 0.975)),
                }
            )
    if not phases.empty:
        for (segment_id, phase), value in phases.groupby(
            ["segment_id", "phase"], observed=True
        ):
            meter = segment.loc[segment.segment_id.eq(segment_id), "meter_name"].iloc[0]
            if allowed_meters is not None and meter not in allowed_meters:
                continue
            rows.append(
                {
                    "meter": meter,
                    "model": "tabpfn_minus_trees",
                    "context_scope": "5k_to_100k",
                    "summary": f"phase_{phase}_score_gap",
                    "segment_id": int(segment_id),
                    "building_id": pd.NA,
                    "segments": 1,
                    "rows": int(value.rows.sum()),
                    "estimate": float(value.tabpfn_tree_disagreement_movement.mean()),
                    "rank_estimate": np.nan,
                    "contribution_fraction": np.nan,
                    "top_building_concentration": np.nan,
                }
            )
    return pd.DataFrame(rows)


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout.strip()


def write_manifest(
    *, args: argparse.Namespace, inputs: pd.DataFrame, invalid: dict[str, int]
) -> None:
    outputs = {
        path.name: file_digest(path)
        for path in OUT.iterdir()
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    manifest = {
        "analysis": "M5 E0 meter-specific learner-gap characterization",
        "branch": git_output("branch", "--show-current"),
        "head_sha": git_output("rev-parse", "HEAD"),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn_version,
        },
        "inputs": inputs.to_dict(orient="records"),
        "outputs_sha256": outputs,
        "bootstrap": {
            "seed": args.seed,
            "draws": args.bootstrap_draws,
            "invalid_draws": invalid,
        },
        "metric_definitions": {
            "pr_auc": "average_precision_score within meter",
            "roc_auc": "tie-aware within-meter ROC AUC",
            "percentile_ranks": "average-tie percentile ranks; global or meter-conditioned",
            "learner_gap": "TabPFN 8.0.8 score metric minus matched-row tree metric",
            "slope": "ordinary least-squares slope over log10(context_rows)",
        },
        "working_tree_status": git_output("status", "--short"),
        "scope": {
            "fit": False,
            "inference": False,
            "gpu": False,
            "independent_192_row_query": False,
            "path_b": False,
            "site_transfer": False,
        },
    }
    (OUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _record(row: pd.Series) -> dict[str, object]:
    return json.loads(row.to_json())


def _validation_provenance(
    args: argparse.Namespace, audit: pd.DataFrame
) -> dict[str, object]:
    # Deliberate validation interruption is an orchestration control, not a
    # result-affecting input.  In particular, Run 1 supplies the stop flag
    # while Run 2/3 must be able to prove reuse of those exact units.
    return {
        "repository": "kuokuant-oss/lead-reproduction",
        "branch": git_output("branch", "--show-current"),
        "committed_source_sha": git_output("rev-parse", "HEAD"),
        "source_file_digest": file_digest(Path(__file__)),
        "execution_mode": "NON_SCIENTIFIC_VALIDATION",
        "inputs": audit.to_dict(orient="records"),
        "contexts": list(CONTEXTS),
        "meters": list(METER_NAMES.values()),
        "learners": list(MODELS),
        "bootstrap_seed": args.seed,
        "metric_definition": "sklearn average_precision_score and roc_auc_score; ranks use average ties",
        "validation_limits": {
            "bootstrap_draws": args.bootstrap_draws,
            "loo_buildings": args.loo_buildings,
            "segment_draws": args.segment_draws,
        },
    }


def formal_bootstrap_manifest() -> list[str]:
    """The immutable 4,000-unit formal bootstrap universe, in RR order."""
    return [
        f"{METER_NAMES[code]}__draw__{draw}"
        for draw in range(FORMAL_BOOTSTRAP_DRAWS_PER_METER)
        for code in METER_NAMES
    ]


def formal_tranche_units(
    store: ResearchCheckpointStore, manifest: list[str], max_new: int
) -> list[str]:
    """Select the smallest missing IDs per meter and schedule them round-robin."""
    completed = store.completed_units(manifest)
    selected: dict[str, list[str]] = {}
    for name in METER_NAMES.values():
        missing = [
            f"{name}__draw__{draw}"
            for draw in range(FORMAL_BOOTSTRAP_DRAWS_PER_METER)
            if f"{name}__draw__{draw}" not in completed
        ]
        selected[name] = missing[:max_new]
    return [
        selected[name][position]
        for position in range(
            max((len(items) for items in selected.values()), default=0)
        )
        for name in METER_NAMES.values()
        if position < len(selected[name])
    ]


def _formal_provenance(
    args: argparse.Namespace, audit: pd.DataFrame
) -> dict[str, object]:
    manifest = formal_bootstrap_manifest()
    return {
        "repository": "kuokuant-oss/lead-reproduction",
        "branch": git_output("branch", "--show-current"),
        "committed_source_sha": git_output("rev-parse", "HEAD"),
        "source_file_digest": file_digest(Path(__file__)),
        "execution_mode": "FORMAL_E0",
        "inputs": audit.to_dict(orient="records"),
        "contexts": list(CONTEXTS),
        "meters": list(METER_NAMES.values()),
        "learners": list(MODELS),
        "bootstrap_seed": args.seed,
        "draw_mapping": "numpy.SeedSequence([bootstrap_seed, meter_code, draw_id])",
        "metric_definition": "sklearn average_precision_score and roc_auc_score; ranks use average ties",
        "formal_bootstrap_draws_per_meter": FORMAL_BOOTSTRAP_DRAWS_PER_METER,
        "formal_bootstrap_expected_units": len(manifest),
        "formal_bootstrap_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, separators=(",", ":")).encode()
        ).hexdigest(),
        "wall_clock_termination": False,
        "output_root": str(args.output_root.resolve()),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "log_root": str(args.log_root.resolve()),
    }


def _formal_root_checks(args: argparse.Namespace) -> None:
    roots = [
        args.output_root.resolve(),
        args.checkpoint_root.resolve(),
        args.log_root.resolve(),
    ]
    if len(set(roots)) != len(roots):
        raise ValueError("formal output, checkpoint, and log roots must be distinct")
    validation_root = (PROC / "m5_e0_validation").resolve()
    if any(
        root == validation_root or validation_root in root.parents for root in roots
    ):
        raise ValueError(
            "FORMAL_E0 roots must be isolated from NON_SCIENTIFIC_VALIDATION"
        )


def _validate_formal_args(args: argparse.Namespace) -> None:
    if args.authorization_token != FORMAL_AUTHORIZATION_TOKEN:
        raise PermissionError(
            "--formal requires the explicit AUTHORIZE_E0_FORMAL_RUN token"
        )
    if not args.resume:
        raise PermissionError(
            "--formal requires --resume for deterministic checkpoint recovery"
        )
    if args.max_new_draws_per_meter != FORMAL_TRANCHE_DRAWS_PER_METER:
        raise ValueError("formal tranche 1 requires --max-new-draws-per-meter 42")
    if args.validation_mode or args.validation_stop_after_units is not None:
        raise PermissionError("formal execution cannot use validation controls")
    if args.phase not in (None, "bootstrap"):
        raise PermissionError(
            "formal tranche permits only identity, base_metrics, and bootstrap"
        )
    _formal_root_checks(args)


def _formal_preflight(args: argparse.Namespace) -> int:
    """Verify formal artifacts and launch conditions without computing metrics."""
    _formal_root_checks(args)
    if git_output("branch", "--show-current") != "m5-tabpfn-repro-audit":
        raise RuntimeError("formal preflight requires m5-tabpfn-repro-audit")
    if git_output("status", "--short", "--", str(Path(__file__).relative_to(ROOT))):
        raise RuntimeError("formal preflight refuses uncommitted execution source")
    for root in (args.output_root, args.checkpoint_root, args.log_root):
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".formal-preflight-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    _, _, _, audit = load_predictions()
    provenance = _formal_provenance(args, audit)
    bootstrap_store = ResearchCheckpointStore(
        args.checkpoint_root, "bootstrap", {**provenance, "phase": "bootstrap"}
    )
    manifest = formal_bootstrap_manifest()
    selected = formal_tranche_units(
        bootstrap_store, manifest, args.max_new_draws_per_meter
    )
    payload = {
        "status": "FORMAL_PREFLIGHT_PASSED",
        "execution_mode": "FORMAL_E0",
        "head_sha": provenance["committed_source_sha"],
        "source_file_digest": provenance["source_file_digest"],
        "input_count": len(provenance["inputs"]),
        "manifest_units": len(manifest),
        "manifest_sha256": provenance["formal_bootstrap_manifest_sha256"],
        "selected_units": selected,
        "selected_per_meter": {
            name: [
                int(unit.rsplit("__", 1)[-1])
                for unit in selected
                if unit.startswith(name + "__")
            ]
            for name in METER_NAMES.values()
        },
        "roots": {
            "output": str(args.output_root),
            "checkpoint": str(args.checkpoint_root),
            "log": str(args.log_root),
        },
        "no_metric_computation": True,
        "wall_clock_termination": False,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(args.output_root / "formal_preflight.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


def _run_formal_tranche(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    scores, _, base, audit = load_predictions()
    provenance = _formal_provenance(args, audit)
    controller = ValidationStopController(args.checkpoint_root, None)

    identity_store = ResearchCheckpointStore(
        args.checkpoint_root, "identity", {**provenance, "phase": "identity"}
    )
    expected_identity = [
        f"{row['model']}-{int(row['context_rows'])}" for row in provenance["inputs"]
    ]
    identity_heartbeat = PhaseHeartbeat(identity_store, expected_identity)
    for unit, row in zip(expected_identity, provenance["inputs"], strict=True):
        _execute_unit(
            store=identity_store,
            expected=expected_identity,
            unit_id=unit,
            meter=None,
            rows=int(row["rows"]),
            compute=lambda row=row: row,
            controller=controller,
            heartbeat=identity_heartbeat,
        )
    _finish_phase(identity_store, expected_identity, identity_heartbeat)

    metrics, _ = per_meter_metrics(scores, base)
    base_store = ResearchCheckpointStore(
        args.checkpoint_root, "base_metrics", {**provenance, "phase": "base_metrics"}
    )
    expected_base = [
        f"{row.model}-{int(row.context_rows)}-{row.meter}"
        for _, row in metrics.iterrows()
    ]
    base_heartbeat = PhaseHeartbeat(base_store, expected_base)
    for unit, (_, row) in zip(expected_base, metrics.iterrows(), strict=True):
        _execute_unit(
            store=base_store,
            expected=expected_base,
            unit_id=unit,
            meter=str(row.meter),
            rows=int(row.rows),
            compute=lambda row=row: _record(row),
            controller=controller,
            heartbeat=base_heartbeat,
        )
    _finish_phase(base_store, expected_base, base_heartbeat)

    bootstrap_store = ResearchCheckpointStore(
        args.checkpoint_root, "bootstrap", {**provenance, "phase": "bootstrap"}
    )
    manifest = formal_bootstrap_manifest()
    selected = formal_tranche_units(
        bootstrap_store, manifest, args.max_new_draws_per_meter
    )
    heartbeat = PhaseHeartbeat(bootstrap_store, manifest)
    for unit in selected:
        name, _, draw_text = unit.partition("__draw__")
        code = next(
            code for code, meter_name in METER_NAMES.items() if meter_name == name
        )
        draw = int(draw_text)

        def compute_bootstrap(
            code: int = code, draw: int = draw, name: str = name
        ) -> dict[str, object]:
            frame, invalid = bootstrap_unit(
                scores, base, code=code, draw=draw, seed=args.seed
            )
            return {
                "meter": name,
                "draw": draw,
                "records": json.loads(frame.to_json(orient="records")),
                "invalid": invalid,
            }

        _execute_unit(
            store=bootstrap_store,
            expected=manifest,
            unit_id=unit,
            meter=name,
            rows=int((base.meter == code).sum()),
            compute=compute_bootstrap,
            controller=controller,
            heartbeat=heartbeat,
        )
    completed = bootstrap_store.completed_units(manifest)
    marker = bootstrap_store.phase_root / "COMPLETE.json"
    if len(completed) == len(manifest):
        _finish_phase(bootstrap_store, manifest, heartbeat)
    elif marker.exists():
        raise CheckpointError(
            "bootstrap completion marker exists before all 4,000 units"
        )
    else:
        heartbeat._write(
            status="running",
            current_unit=None,
            current_meter=None,
            completion_marker=None,
        )
    summary = {
        "execution_mode": "FORMAL_E0",
        "status": "PARTIAL_BOOTSTRAP_TRANCHE"
        if len(completed) < len(manifest)
        else "BOOTSTRAP_COMPLETE",
        "expected_bootstrap_units": len(manifest),
        "completed_bootstrap_units": len(completed),
        "newly_computed_units": controller.computed_units,
        "reused_units": controller.reused_units,
        "selected_units": selected,
        "selected_draw_ids": {
            name: [
                int(unit.rsplit("__", 1)[-1])
                for unit in selected
                if unit.startswith(name + "__")
            ]
            for name in METER_NAMES.values()
        },
        "bootstrap_complete_marker": str(marker) if marker.exists() else None,
        "elapsed_seconds": time.perf_counter() - started,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(args.output_root / "formal_tranche_summary.json", summary)
    print(
        f"FORMAL_E0 tranche checkpointed {len(selected)} bootstrap units; {len(completed)}/{len(manifest)} complete",
        flush=True,
    )
    return 0


def _write_invocation_record(root: Path, args: argparse.Namespace) -> None:
    """Record each launcher invocation outside result-affecting provenance."""
    invocation = {
        "command": [Path(sys.executable).name, *sys.argv],
        "validation_stop_after_units": args.validation_stop_after_units,
        "execution_mode": "NON_SCIENTIFIC_VALIDATION",
    }
    atomic_write_json(root / "last_invocation.json", invocation)


class ExpectedValidationInterruption(RuntimeError):
    pass


class ValidationStopController:
    def __init__(self, root: Path, stop_after_units: int | None) -> None:
        self.root = root
        self.stop_after_units = stop_after_units
        self.computed_units = 0
        self.reused_units = 0

    def after_computed_unit(self, *, phase: str, unit_id: str) -> None:
        self.computed_units += 1
        if (
            self.stop_after_units is not None
            and self.computed_units >= self.stop_after_units
        ):
            atomic_write_json(
                self.root / "EXPECTED_VALIDATION_INTERRUPTION.json",
                {
                    "status": "EXPECTED_VALIDATION_INTERRUPTION",
                    "exit_code": EXPECTED_VALIDATION_INTERRUPTION_EXIT,
                    "computed_units": self.computed_units,
                    "phase": phase,
                    "unit_id": unit_id,
                },
            )
            raise ExpectedValidationInterruption(
                f"EXPECTED_VALIDATION_INTERRUPTION after {phase}/{unit_id}"
            )


class PhaseHeartbeat:
    """Atomic, monotonic phase status across fresh, resumed, and reused work."""

    def __init__(self, store: ResearchCheckpointStore, expected: list[str]) -> None:
        self.store = store
        self.expected = expected
        self.total = len(expected)
        self.started = time.perf_counter()
        self.initial_completed = store.completed_units(
            expected
        )  # validate before status is published
        self.completed = len(self.initial_completed)
        self.computed = 0
        self.reused = self.completed
        self.last_checkpoint = (
            str(store.unit_path(sorted(self.initial_completed)[-1]))
            if self.initial_completed
            else None
        )
        self._write(
            status="reusing" if self.completed == self.total else "running",
            current_unit=None,
            current_meter=None,
            completion_marker=None,
        )

    def _write(
        self,
        *,
        status: str,
        current_unit: str | None,
        current_meter: str | None,
        completion_marker: Path | None,
    ) -> None:
        verified_completed = len(self.store.completed_units(self.expected))
        if verified_completed < self.completed:
            raise CheckpointError(
                f"{self.store.phase}: heartbeat completed counter regressed"
            )
        self.completed = verified_completed
        pending = self.total - self.completed
        elapsed = time.perf_counter() - self.started
        throughput = self.completed / elapsed if elapsed > 0 else None
        eta = (
            pending / throughput
            if throughput and pending
            else (0.0 if pending == 0 else None)
        )
        self.store.heartbeat(
            status=status,
            total=self.total,
            total_units=self.total,
            completed=self.completed,
            completed_units=self.completed,
            computed=self.computed,
            reused=self.reused,
            pending=pending,
            current_unit=current_unit,
            current_meter=current_meter,
            last_completed_checkpoint=self.last_checkpoint,
            last_progress_timestamp=datetime.now(UTC).isoformat(),
            elapsed_seconds=elapsed,
            throughput_units_per_second=throughput,
            eta_seconds=eta,
            phase_completion_marker=str(completion_marker)
            if completion_marker is not None
            else None,
        )

    def reused_unit(self, unit_id: str, meter: str | None) -> None:
        self.last_checkpoint = str(self.store.unit_path(unit_id))
        self._write(
            status="reusing" if self.completed == self.total else "running",
            current_unit=unit_id,
            current_meter=meter,
            completion_marker=None,
        )

    def computed_unit(self, unit_id: str, meter: str | None) -> None:
        self.computed += 1
        self.last_checkpoint = str(self.store.unit_path(unit_id))
        self._write(
            status="running",
            current_unit=unit_id,
            current_meter=meter,
            completion_marker=None,
        )

    def complete(self, marker: Path) -> None:
        self._write(
            status="completed",
            current_unit=None,
            current_meter=None,
            completion_marker=marker,
        )


def rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _execute_unit(
    *,
    store: ResearchCheckpointStore,
    expected: list[str],
    unit_id: str,
    meter: str | None,
    rows: int | None,
    compute: callable,
    controller: ValidationStopController,
    heartbeat: PhaseHeartbeat,
) -> dict[str, object]:
    completed = store.completed_units(expected)
    if unit_id in completed:
        payload = store.read_unit(unit_id)
        controller.reused_units += 1
        runtime = {
            "computed": False,
            "reused": True,
            "meter": meter,
            "rows": rows,
            "compute_seconds": 0.0,
            "checkpoint_write_seconds": 0.0,
            "checkpoint_validation_seconds": 0.0,
            "total_seconds": 0.0,
            "rss_bytes": rss_bytes(),
        }
        store.write_runtime(unit_id, runtime)
        heartbeat.reused_unit(unit_id, meter)
        store.log(f"reused {unit_id}")
        return payload
    started = time.perf_counter()
    payload = compute()
    after_compute = time.perf_counter()
    store.write_unit(unit_id, payload, validate=False)
    after_write = time.perf_counter()
    store.read_unit(unit_id)
    after_validation = time.perf_counter()
    runtime = {
        "computed": True,
        "reused": False,
        "meter": meter,
        "rows": rows,
        "compute_seconds": after_compute - started,
        "checkpoint_write_seconds": after_write - after_compute,
        "checkpoint_validation_seconds": after_validation - after_write,
        "total_seconds": after_validation - started,
        "rss_bytes": rss_bytes(),
    }
    store.write_runtime(unit_id, runtime)
    heartbeat.computed_unit(unit_id, meter)
    store.log(f"computed {unit_id} total_seconds={runtime['total_seconds']:.6f}")
    controller.after_computed_unit(phase=store.phase, unit_id=unit_id)
    return payload


def bootstrap_unit(
    scores: dict[str, dict[int, np.ndarray]],
    base: pd.DataFrame,
    *,
    code: int,
    draw: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """One deterministic meter × draw checkpoint; contexts share this building sample."""
    meter_name = METER_NAMES[code]
    meter = base["meter"].to_numpy()
    local = np.flatnonzero(meter == code)
    building = base["building_id"].to_numpy()
    y = base["anomaly"].to_numpy()
    buildings = np.unique(building[local])
    by_building = {int(value): local[building[local] == value] for value in buildings}
    rng = np.random.default_rng(np.random.SeedSequence([seed, code, draw]))
    chosen = rng.choice(buildings, size=len(buildings), replace=True)
    sampled = np.concatenate([by_building[int(value)] for value in chosen])
    y_draw = y[sampled]
    if y_draw.min() == y_draw.max():
        return pd.DataFrame(), {"single_class": 1}
    rows: list[dict[str, object]] = []
    for context in CONTEXTS:
        by_model: dict[str, dict[str, float]] = {}
        for model in MODELS:
            score = scores[model][context][sampled]
            rank = pd.Series(score).rank(method="average", pct=True).to_numpy("float64")
            stats = metric_values(y_draw, score, rank, rank)
            by_model[model] = stats
            for metric in PRIMARY_GAP_METRICS:
                rows.append(
                    {
                        "draw": draw,
                        "meter": meter_name,
                        "context_rows": context,
                        "quantity": model,
                        "metric": metric,
                        "estimate": stats[metric],
                    }
                )
        for metric in PRIMARY_GAP_METRICS:
            rows.append(
                {
                    "draw": draw,
                    "meter": meter_name,
                    "context_rows": context,
                    "quantity": "tabpfn_minus_trees",
                    "metric": metric,
                    "estimate": by_model["tabpfn"][metric] - by_model["trees"][metric],
                }
            )
    for metric in PRIMARY_GAP_METRICS:
        end_gap = (
            metric_values(
                y_draw,
                scores["tabpfn"][100_000][sampled],
                np.zeros(len(y_draw)),
                np.zeros(len(y_draw)),
            )[metric]
            - metric_values(
                y_draw,
                scores["trees"][100_000][sampled],
                np.zeros(len(y_draw)),
                np.zeros(len(y_draw)),
            )[metric]
        )
        start_gap = (
            metric_values(
                y_draw,
                scores["tabpfn"][5_000][sampled],
                np.zeros(len(y_draw)),
                np.zeros(len(y_draw)),
            )[metric]
            - metric_values(
                y_draw,
                scores["trees"][5_000][sampled],
                np.zeros(len(y_draw)),
                np.zeros(len(y_draw)),
            )[metric]
        )
        rows.append(
            {
                "draw": draw,
                "meter": meter_name,
                "context_rows": 100_000,
                "quantity": "learner_gap_change_5k_to_100k",
                "metric": metric,
                "estimate": end_gap - start_gap,
            }
        )
    return pd.DataFrame(rows), {"single_class": 0}


def leave_one_building_unit(
    scores: dict[str, dict[int, np.ndarray]],
    base: pd.DataFrame,
    ranks: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    gaps: pd.DataFrame,
    *,
    code: int,
    building_id: int,
) -> pd.DataFrame:
    """One exact meter × building omission, without evaluating other omissions."""
    name = METER_NAMES[code]
    y, meter, building = (
        base[column].to_numpy() for column in ("anomaly", "meter", "building_id")
    )
    mask = meter == code
    keep = mask & (building != building_id)
    details = {
        "building_id": int(building_id),
        "building_rows": int((mask & (building == building_id)).sum()),
        "building_positives": int(y[mask & (building == building_id)].sum()),
    }
    rows: list[dict[str, object]] = []
    for metric in PRIMARY_GAP_METRICS:
        values: dict[tuple[str, int], float] = {}
        for model in MODELS:
            for context in (5_000, 100_000):
                global_rank, within_rank = ranks[model, context]
                values[model, context] = metric_values(
                    y[keep],
                    scores[model][context][keep],
                    global_rank[keep],
                    within_rank[keep],
                )[metric]
        full_100 = float(
            gaps.loc[
                (gaps.meter == name)
                & (gaps.context_rows == 100_000)
                & (gaps.metric == metric),
                "tabpfn_minus_trees",
            ].iloc[0]
        )
        full_change = full_100 - float(
            gaps.loc[
                (gaps.meter == name)
                & (gaps.context_rows == 5_000)
                & (gaps.metric == metric),
                "tabpfn_minus_trees",
            ].iloc[0]
        )
        for endpoint, estimate, full in (
            (
                "100k_learner_gap",
                values["tabpfn", 100_000] - values["trees", 100_000],
                full_100,
            ),
            (
                "5k_to_100k_learner_gap_change",
                (values["tabpfn", 100_000] - values["trees", 100_000])
                - (values["tabpfn", 5_000] - values["trees", 5_000]),
                full_change,
            ),
        ):
            rows.append(
                {
                    "meter": name,
                    "metric": metric,
                    "endpoint": endpoint,
                    "full_estimate": full,
                    "leave_one_building_estimate": estimate,
                    "influence_delta": estimate - full,
                    **details,
                }
            )
    full_rank = float(
        gaps.loc[
            (gaps.meter == name)
            & (gaps.context_rows == 100_000)
            & (gaps.metric == "anomaly_within_meter_percentile_rank"),
            "tabpfn_minus_trees",
        ].iloc[0]
    )
    rank_estimate = float(
        ranks["tabpfn", 100_000][1][keep & (y == 1)].mean()
        - ranks["trees", 100_000][1][keep & (y == 1)].mean()
    )
    rows.append(
        {
            "meter": name,
            "metric": "anomaly_within_meter_percentile_rank",
            "endpoint": "100k_positive_rank_gap",
            "full_estimate": full_rank,
            "leave_one_building_estimate": rank_estimate,
            "influence_delta": rank_estimate - full_rank,
            **details,
        }
    )
    return pd.DataFrame(rows)


def loo_unit_manifest(
    base: pd.DataFrame, loo_buildings: int
) -> tuple[dict[int, np.ndarray], list[str]]:
    selected = {
        code: np.unique(base.loc[base.meter.eq(code), "building_id"].to_numpy())[
            :loo_buildings
        ]
        for code in METER_NAMES
    }
    expected = [
        f"{METER_NAMES[code]}__building__{int(building_id)}"
        for code, values in selected.items()
        for building_id in values
    ]
    return selected, expected


def _finish_phase(
    store: ResearchCheckpointStore, expected: list[str], heartbeat: PhaseHeartbeat
) -> None:
    started = time.perf_counter()
    store.complete_phase(expected)
    elapsed = time.perf_counter() - started
    store.write_runtime(
        "finalization",
        {
            "computed": True,
            "reused": False,
            "meter": None,
            "rows": None,
            "compute_seconds": 0.0,
            "checkpoint_write_seconds": elapsed,
            "checkpoint_validation_seconds": 0.0,
            "total_seconds": elapsed,
            "rss_bytes": rss_bytes(),
            "finalization": True,
        },
    )
    heartbeat.complete(store.phase_root / "COMPLETE.json")


def _run_validation_phase(
    phase: str,
    store: ResearchCheckpointStore,
    scores: dict[str, dict[int, np.ndarray]],
    base: pd.DataFrame,
    metrics: pd.DataFrame,
    ranks: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    gaps: pd.DataFrame,
    args: argparse.Namespace,
    controller: ValidationStopController,
) -> None:
    if phase == "identity":
        expected = [
            f"{row['model']}-{int(row['context_rows'])}"
            for row in store.provenance["inputs"]
        ]
        heartbeat = PhaseHeartbeat(store, expected)
        for unit, row in zip(expected, store.provenance["inputs"], strict=True):
            _execute_unit(
                store=store,
                expected=expected,
                unit_id=unit,
                meter=None,
                rows=int(row["rows"]),
                compute=lambda row=row: row,
                controller=controller,
                heartbeat=heartbeat,
            )
        _finish_phase(store, expected, heartbeat)
        return
    if phase == "base_metrics":
        expected = [
            f"{row.model}-{int(row.context_rows)}-{row.meter}"
            for _, row in metrics.iterrows()
        ]
        heartbeat = PhaseHeartbeat(store, expected)
        for unit, (_, row) in zip(expected, metrics.iterrows(), strict=True):
            _execute_unit(
                store=store,
                expected=expected,
                unit_id=unit,
                meter=str(row.meter),
                rows=int(row.rows),
                compute=lambda row=row: _record(row),
                controller=controller,
                heartbeat=heartbeat,
            )
        _finish_phase(store, expected, heartbeat)
        return
    if phase == "bootstrap":
        expected = [
            f"{name}__draw__{draw}"
            for name in METER_NAMES.values()
            for draw in range(args.bootstrap_draws)
        ]
        heartbeat = PhaseHeartbeat(store, expected)
        for code, meter_name in METER_NAMES.items():
            for draw in range(args.bootstrap_draws):
                unit = f"{meter_name}__draw__{draw}"

                def compute_bootstrap(
                    code: int = code, draw: int = draw, meter_name: str = meter_name
                ) -> dict[str, object]:
                    draw_frame, invalid = bootstrap_unit(
                        scores, base, code=code, draw=draw, seed=args.seed
                    )
                    return {
                        "meter": meter_name,
                        "draw": draw,
                        "records": json.loads(draw_frame.to_json(orient="records")),
                        "invalid": invalid,
                    }

                _execute_unit(
                    store=store,
                    expected=expected,
                    unit_id=unit,
                    meter=meter_name,
                    rows=int((base.meter == code).sum()),
                    compute=compute_bootstrap,
                    controller=controller,
                    heartbeat=heartbeat,
                )
        _finish_phase(store, expected, heartbeat)
        return
    if phase == "leave_one_building":
        selected, expected = loo_unit_manifest(base, args.loo_buildings)
        heartbeat = PhaseHeartbeat(store, expected)
        for code, meter_name in METER_NAMES.items():
            positions = np.flatnonzero(base.meter.to_numpy() == code)
            buildings = selected[code]
            positions = positions[
                np.isin(base.building_id.to_numpy()[positions], buildings)
            ]
            unit_base = base.iloc[positions].reset_index(drop=True)
            unit_scores = {
                model: {context: value[positions] for context, value in values.items()}
                for model, values in scores.items()
            }
            unit_metrics, unit_ranks = per_meter_metrics(unit_scores, unit_base)
            unit_gaps = learner_gaps(unit_metrics)
            for building_id in buildings:
                unit = f"{meter_name}__building__{int(building_id)}"
                _execute_unit(
                    store=store,
                    expected=expected,
                    unit_id=unit,
                    meter=meter_name,
                    rows=int((unit_base.building_id == building_id).sum()),
                    compute=lambda building_id=int(building_id),
                    code=code,
                    meter_name=meter_name: {
                        "meter": meter_name,
                        "building_id": building_id,
                        "records": json.loads(
                            leave_one_building_unit(
                                unit_scores,
                                unit_base,
                                unit_ranks,
                                unit_gaps,
                                code=code,
                                building_id=building_id,
                            ).to_json(orient="records")
                        ),
                    },
                    controller=controller,
                    heartbeat=heartbeat,
                )
        _finish_phase(store, expected, heartbeat)
        return
    if phase == "segment":
        expected = [
            f"{meter_name}__segment_summary" for meter_name in METER_NAMES.values()
        ]
        heartbeat = PhaseHeartbeat(store, expected)
        for meter_name in expected:
            name = meter_name.removesuffix("__segment_summary")
            _execute_unit(
                store=store,
                expected=expected,
                unit_id=meter_name,
                meter=name,
                rows=None,
                compute=lambda name=name: {
                    "meter": name,
                    "records": json.loads(
                        segment_concentration(
                            {name}, sensitivity_draws=args.segment_draws
                        ).to_json(orient="records")
                    ),
                },
                controller=controller,
                heartbeat=heartbeat,
            )
        _finish_phase(store, expected, heartbeat)
        return
    raise ValueError(f"unknown phase: {phase}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "identity",
            "base_metrics",
            "bootstrap",
            "leave_one_building",
            "segment",
            "all",
        ),
        default=None,
    )
    parser.add_argument("--validation-mode", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--authorization-token")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-draws-per-meter", type=int)
    parser.add_argument("--formal-preflight", action="store_true")
    parser.add_argument("--output-root", type=Path, default=PROC / "m5_e0_validation")
    parser.add_argument(
        "--checkpoint-root", type=Path, default=PROC / "m5_e0_validation"
    )
    parser.add_argument("--log-root", type=Path, default=PROC / "m5_e0_validation")
    parser.add_argument("--bootstrap-draws", type=int, default=None)
    parser.add_argument("--loo-buildings", type=int, default=None)
    parser.add_argument("--segment-draws", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=1)
    parser.add_argument("--provenance-only", action="store_true")
    parser.add_argument("--validation-stop-after-units", type=int)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    if args.formal:
        _validate_formal_args(args)
        if args.formal_preflight:
            return _formal_preflight(args)
        return _run_formal_tranche(args)
    if (
        args.formal_preflight
        or args.authorization_token
        or args.resume
        or args.max_new_draws_per_meter is not None
    ):
        raise PermissionError("formal controls require --formal")
    if not args.validation_mode:
        raise PermissionError(
            "default invocation is safe: specify --validation-mode with deterministic limits"
        )
    if (
        args.phase is None
        or args.bootstrap_draws is None
        or args.loo_buildings is None
        or args.segment_draws is None
    ):
        raise ValueError(
            "validation requires --phase and explicit limits for bootstrap, LOO, and segment work"
        )
    if min(args.bootstrap_draws, args.loo_buildings, args.segment_draws) < 1:
        raise ValueError("all deterministic validation limits must be positive")
    if (
        args.validation_stop_after_units is not None
        and args.validation_stop_after_units < 1
    ):
        raise ValueError("--validation-stop-after-units must be positive")
    if args.provenance_only:
        print(
            json.dumps(
                {
                    "mode": "NON_SCIENTIFIC_VALIDATION",
                    "source_digest": file_digest(Path(__file__)),
                },
                indent=2,
            )
        )
        return 0
    run_started = time.perf_counter()
    scores, _, base, audit = load_predictions()
    artifact_load_seconds = time.perf_counter() - run_started
    metrics, ranks = per_meter_metrics(scores, base)
    gaps = learner_gaps(metrics)
    provenance = _validation_provenance(args, audit)
    _write_invocation_record(args.checkpoint_root, args)
    phases = (
        ("identity", "base_metrics", "bootstrap", "leave_one_building", "segment")
        if args.phase == "all"
        else (args.phase,)
    )
    controller = ValidationStopController(
        args.checkpoint_root, args.validation_stop_after_units
    )
    try:
        for phase in phases:
            store = ResearchCheckpointStore(
                args.checkpoint_root, phase, {**provenance, "phase": phase}
            )
            _run_validation_phase(
                phase, store, scores, base, metrics, ranks, gaps, args, controller
            )
    except ExpectedValidationInterruption as error:
        events = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (args.checkpoint_root / "checkpoints").glob("*/runtime/*.json")
        ]
        atomic_write_json(
            args.checkpoint_root / "runtime_summary.json",
            {
                "execution_mode": "NON_SCIENTIFIC_VALIDATION",
                "status": "EXPECTED_VALIDATION_INTERRUPTION",
                "artifact_load_seconds": artifact_load_seconds,
                "computed_units": controller.computed_units,
                "reused_units": controller.reused_units,
                "peak_rss_bytes": max(
                    (
                        event["rss_bytes"]
                        for event in events
                        if event.get("rss_bytes") is not None
                    ),
                    default=None,
                ),
                "unit_events": events,
            },
        )
        # This is an expected validation control-flow result, not an error.
        # Keep it on stdout so Windows PowerShell does not promote stderr into
        # a terminating NativeCommandError before the launcher can inspect 75.
        print(str(error), flush=True)
        return EXPECTED_VALIDATION_INTERRUPTION_EXIT
    runtime_files = list(
        (args.checkpoint_root / "checkpoints").glob("*/runtime/*.json")
    )
    events = [json.loads(path.read_text(encoding="utf-8")) for path in runtime_files]

    def distribution(values: list[float]) -> dict[str, object]:
        if not values:
            return {
                "sample_count": 0,
                "min": None,
                "median": None,
                "max": None,
                "p95": None,
            }
        return {
            "sample_count": len(values),
            "min": min(values),
            "median": float(np.median(values)),
            "max": max(values),
            "p95": float(np.quantile(values, 0.95)) if len(values) >= 20 else None,
        }

    phase_summary: dict[str, dict[str, object]] = {}
    for phase in phases:
        phase_events = [event for event in events if event["phase"] == phase]
        computational = [
            event for event in phase_events if not event.get("finalization")
        ]
        phase_summary[phase] = {
            "computed_units": sum(event["computed"] for event in computational),
            "reused_units": sum(event["reused"] for event in computational),
            "unit_total_seconds": distribution(
                [float(event["total_seconds"]) for event in computational]
            ),
            "checkpoint_overhead_seconds": distribution(
                [
                    float(
                        event["checkpoint_write_seconds"]
                        + event["checkpoint_validation_seconds"]
                    )
                    for event in computational
                ]
            ),
            "finalization_seconds": sum(
                float(event["total_seconds"])
                for event in phase_events
                if event.get("finalization")
            ),
        }
    summary = {
        "execution_mode": "NON_SCIENTIFIC_VALIDATION",
        "artifact_load_seconds": artifact_load_seconds,
        "resume_startup_seconds": artifact_load_seconds,
        "computed_units": controller.computed_units,
        "reused_units": controller.reused_units,
        "peak_rss_bytes": max(
            (
                event["rss_bytes"]
                for event in events
                if event.get("rss_bytes") is not None
            ),
            default=None,
        ),
        "phases": phase_summary,
        "unit_events": events,
    }
    atomic_write_json(args.checkpoint_root / "runtime_summary.json", summary)
    print(
        f"completed bounded NON_SCIENTIFIC_VALIDATION checkpoints under {args.checkpoint_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
