"""CPU-only mechanism analysis for existing M5 F4 full-holdout scores.

The script never fits a model.  It joins the five pre-existing context-size
prediction files for TabPFN and matched-row trees, creates auditable row-level
score/rank movements, and resolves aggregate effects into meter, building, and
anomaly-segment diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = PROC / "m5_context_mechanism_137"
CONTEXTS = (5_000, 10_000, 20_000, 50_000, 100_000)
METER_NAMES = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}
FPRS = (0.001, 0.01, 0.05)
TPRS = (0.50, 0.80, 0.95)


def prediction_path(model: str, context: int) -> Path:
    if model == "tabpfn":
        suffix = (
            "n8_predictions.npz"
            if context == 100_000
            else f"context{context}_n8_predictions.npz"
        )
        return PROC / f"m5_tabpfn_137_full_test_{suffix}"
    return PROC / f"m5_tree_ensemble_f137_context{context}_predictions.npz"


def load_scores(model: str) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
    """Load one model's five scores and prove its fixed query identity."""
    scores: dict[int, np.ndarray] = {}
    metadata: dict[str, np.ndarray] | None = None
    for context in CONTEXTS:
        with np.load(prediction_path(model, context)) as payload:
            raw_key = "raw_index" if model == "tabpfn" else "validation_raw_index"
            score_key = "tabpfn" if model == "tabpfn" else "ensemble"
            current = {
                "raw_index": np.asarray(payload[raw_key], dtype="int64"),
                "anomaly": np.asarray(payload["anomaly"], dtype="int8"),
                "site_id": np.asarray(payload["site_id"], dtype="int8"),
                "building_id": np.asarray(payload["building_id"], dtype="int16"),
            }
            if metadata is None:
                metadata = current
            elif any(
                not np.array_equal(metadata[key], current[key]) for key in metadata
            ):
                raise AssertionError(f"{model}/{context}: query rows or labels drifted")
            scores[context] = np.asarray(payload[score_key], dtype="float32")
    assert metadata is not None
    return scores, metadata


def load_raw_fields(raw_index: np.ndarray) -> pd.DataFrame:
    """Read only fields needed for CPU diagnostics and join by frozen position."""
    raw = pd.read_csv(
        ROOT / "data" / "raw" / "m3" / "train.csv",
        usecols=["meter", "timestamp", "meter_reading"],
        parse_dates=["timestamp"],
    )
    selected = raw.iloc[raw_index].reset_index(drop=True)
    selected.insert(0, "raw_index", raw_index)
    return selected


def reading_regimes(frame: pd.DataFrame) -> pd.Series:
    """Use label-blind meter-specific reading regimes; preserve hotwater 0--1."""
    regime = pd.Series(index=frame.index, dtype="object")
    for meter, group in frame.groupby("meter", sort=False):
        values = group["meter_reading"].to_numpy(dtype="float64")
        if meter == 3:
            regime.loc[group.index] = np.where(
                values <= 1.0, "hotwater_0_to_1", "hotwater_gt_1"
            )
            continue
        q25, q75 = np.nanquantile(values, [0.25, 0.75])
        labels = np.where(
            values <= q25, "low", np.where(values >= q75, "high", "middle")
        )
        regime.loc[group.index] = labels
    return regime.astype("string")


def percentile_ranks(
    score: np.ndarray, meter: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    global_rank = pd.Series(score).rank(method="average", pct=True).to_numpy("float32")
    within_rank = (
        pd.DataFrame({"score": score, "meter": meter})
        .groupby("meter", sort=False)["score"]
        .rank(method="average", pct=True)
        .to_numpy(dtype="float32")
    )
    return global_rank, within_rank


def pair_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Tie-aware AUC without materialising positive × negative pairs."""
    if not len(positive) or not len(negative):
        return float("nan")
    ordered_negative = np.sort(negative)
    below = np.searchsorted(ordered_negative, positive, side="left")
    at_or_below = np.searchsorted(ordered_negative, positive, side="right")
    return float((below + 0.5 * (at_or_below - below)).mean() / len(negative))


def pairwise_decomposition(
    model: str, scores: dict[int, np.ndarray], base: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    meter = base["meter"].to_numpy()
    label = base["anomaly"].to_numpy()
    positives_total = int(label.sum())
    negatives_total = int((label == 0).sum())
    for context, score in scores.items():
        for pos_meter in METER_NAMES:
            positive = score[(label == 1) & (meter == pos_meter)]
            for neg_meter in METER_NAMES:
                negative = score[(label == 0) & (meter == neg_meter)]
                weight = (
                    len(positive) * len(negative) / (positives_total * negatives_total)
                )
                rows.append(
                    {
                        "model": model,
                        "context_rows": context,
                        "positive_meter": METER_NAMES[pos_meter],
                        "negative_meter": METER_NAMES[neg_meter],
                        "positive_rows": len(positive),
                        "negative_rows": len(negative),
                        "within_meter": pos_meter == neg_meter,
                        "pair_weight": weight,
                        "pair_auc": pair_auc(positive, negative),
                        "weighted_auc_contribution": weight
                        * pair_auc(positive, negative),
                    }
                )
    result = pd.DataFrame(rows)
    summary = result.groupby(
        ["model", "context_rows", "within_meter"], as_index=False
    ).agg(
        pair_weight=("pair_weight", "sum"),
        weighted_auc_contribution=("weighted_auc_contribution", "sum"),
    )
    summary["decomposition"] = np.where(
        summary["within_meter"], "within_meter", "cross_meter"
    )
    summary["pair_auc"] = summary["weighted_auc_contribution"] / summary["pair_weight"]
    summary["positive_meter"] = "all"
    summary["negative_meter"] = "all"
    summary["positive_rows"] = pd.NA
    summary["negative_rows"] = pd.NA
    summary["decomposition"] = np.where(
        summary["within_meter"], "within_meter", "cross_meter"
    )
    result["decomposition"] = "meter_pair"
    return pd.concat([result, summary[result.columns]], ignore_index=True)


def score_distributions(
    model: str, scores: dict[int, np.ndarray], base: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["meter_name", "anomaly", "reading_regime"]
    for context, score in scores.items():
        work = base[group_columns].copy()
        work["score"] = score
        for keys, group in work.groupby(group_columns, observed=True, sort=True):
            values = group["score"].to_numpy()
            rows.append(
                {
                    "model": model,
                    "context_rows": context,
                    **dict(zip(group_columns, keys, strict=True)),
                    "rows": len(values),
                    "mean_score": values.mean(),
                    "median_score": np.median(values),
                    "q10_score": np.quantile(values, 0.10),
                    "q90_score": np.quantile(values, 0.90),
                }
            )
    return pd.DataFrame(rows)


def operating_points(
    model: str, scores: dict[int, np.ndarray], base: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [("global", np.ones(len(base), dtype=bool))]
    groups.extend(
        (name, base["meter"].to_numpy() == meter) for meter, name in METER_NAMES.items()
    )
    y_all = base["anomaly"].to_numpy()
    for context, score in scores.items():
        for group_name, mask in groups:
            y, values = y_all[mask], score[mask]
            pos, neg = values[y == 1], values[y == 0]
            for fpr in FPRS:
                threshold = float(np.quantile(neg, 1.0 - fpr))
                rows.append(
                    {
                        "model": model,
                        "context_rows": context,
                        "group": group_name,
                        "operating_point": f"fpr_{fpr:g}",
                        "threshold": threshold,
                        "recall": float((pos >= threshold).mean()),
                        "fpr": float((neg >= threshold).mean()),
                    }
                )
            for tpr in TPRS:
                threshold = float(np.quantile(pos, 1.0 - tpr))
                rows.append(
                    {
                        "model": model,
                        "context_rows": context,
                        "group": group_name,
                        "operating_point": f"tpr_{tpr:g}",
                        "threshold": threshold,
                        "recall": float((pos >= threshold).mean()),
                        "fpr": float((neg >= threshold).mean()),
                    }
                )
    return pd.DataFrame(rows)


def threshold_crossings(
    model: str, scores: dict[int, np.ndarray], base: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y_all, meter = base["anomaly"].to_numpy(), base["meter"].to_numpy()
    for context in CONTEXTS[1:]:
        for group_name, mask in [
            ("global", np.ones(len(base), dtype=bool)),
            *[(name, meter == code) for code, name in METER_NAMES.items()],
        ]:
            for fpr in FPRS:
                threshold = float(
                    np.quantile(scores[context][mask & (y_all == 0)], 1.0 - fpr)
                )
                before = scores[5_000][mask] >= threshold
                after = scores[context][mask] >= threshold
                labels = y_all[mask]
                for label in (0, 1):
                    select = labels == label
                    rows.append(
                        {
                            "model": model,
                            "context_rows": context,
                            "group": group_name,
                            "label": label,
                            "fixed_fpr": fpr,
                            "threshold_at_context": threshold,
                            "rows": int(select.sum()),
                            "below_to_above": int(
                                (~before[select] & after[select]).sum()
                            ),
                            "above_to_below": int(
                                (before[select] & ~after[select]).sum()
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def building_uncertainty(
    model: str,
    score_5k: np.ndarray,
    score_100k: np.ndarray,
    base: pd.DataFrame,
    seed: int,
    draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_5k, within_5k = percentile_ranks(score_5k, base["meter"].to_numpy())
    global_100k, within_100k = percentile_ranks(score_100k, base["meter"].to_numpy())
    work = base[["building_id", "meter_name", "anomaly"]].copy()
    work["score_delta"] = score_100k - score_5k
    work["global_rank_delta"] = global_100k - global_5k
    work["within_meter_rank_delta"] = within_100k - within_5k
    rows: list[dict[str, object]] = []
    influence: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for (meter_name, label), group in work.groupby(
        ["meter_name", "anomaly"], observed=True
    ):
        by_building = group.groupby("building_id", observed=True)[
            ["score_delta", "global_rank_delta", "within_meter_rank_delta"]
        ].mean()
        values = by_building.to_numpy(dtype="float64")
        point = values.mean(axis=0)
        sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(
            axis=1
        )
        for column, estimate, samples in zip(
            by_building.columns, point, sampled.T, strict=True
        ):
            rows.append(
                {
                    "model": model,
                    "meter_name": meter_name,
                    "anomaly": label,
                    "statistic": column,
                    "buildings": len(values),
                    "estimate": estimate,
                    "bootstrap_q025": np.quantile(samples, 0.025),
                    "bootstrap_q975": np.quantile(samples, 0.975),
                    "bootstrap_draws": draws,
                }
            )
            leave_one_out = (
                values.sum(axis=0)[list(by_building.columns).index(column)]
                - by_building[column].to_numpy()
            ) / (len(values) - 1)
            for building_id, loo in zip(by_building.index, leave_one_out, strict=True):
                influence.append(
                    {
                        "model": model,
                        "meter_name": meter_name,
                        "anomaly": label,
                        "statistic": column,
                        "building_id": building_id,
                        "leave_one_building_estimate": loo,
                        "influence_vs_full": loo - estimate,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(influence)


def add_temporal_statistics(base: pd.DataFrame) -> pd.DataFrame:
    """Derive raw-reading temporal diagnostics within the frozen holdout stream."""
    ordered = base.sort_values(
        ["building_id", "meter", "timestamp", "raw_index"], kind="stable"
    ).copy()
    group = ordered.groupby(["building_id", "meter"], observed=True)["meter_reading"]
    timestamp_group = ordered.groupby(["building_id", "meter"], observed=True)[
        "timestamp"
    ]
    for hours in (1, 24, 168):
        previous = group.shift(hours)
        previous_timestamp = timestamp_group.shift(hours)
        contiguous = (
            (ordered["timestamp"] - previous_timestamp)
            .dt.total_seconds()
            .eq(hours * 3600)
        )
        previous = previous.where(contiguous)
        ordered[f"diff_{hours}h"] = ordered["meter_reading"] - previous
        ordered[f"ratio_{hours}h"] = ordered["meter_reading"] / previous.replace(
            0, np.nan
        )
    return ordered


def segment_outputs(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge adjacent hourly anomaly rows and summarize morphology plus score movement."""
    anomaly = base.loc[base["anomaly"] == 1].copy()
    anomaly.sort_values(
        ["building_id", "meter", "timestamp", "raw_index"], inplace=True, kind="stable"
    )
    gap = (
        anomaly.groupby(["building_id", "meter"], observed=True)["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )
    starts = gap.isna() | (gap > 1.0)
    anomaly["segment_id"] = starts.cumsum().astype("int64")
    anomaly["position"] = anomaly.groupby("segment_id", observed=True).cumcount()
    anomaly["segment_rows"] = anomaly.groupby("segment_id", observed=True)[
        "segment_id"
    ].transform("size")
    fraction = anomaly["position"] / np.maximum(anomaly["segment_rows"] - 1, 1)
    anomaly["phase"] = np.where(
        fraction <= 0.25, "onset", np.where(fraction >= 0.75, "recovery", "middle")
    )
    aggregate = (
        anomaly.groupby("segment_id", observed=True)
        .agg(
            building_id=("building_id", "first"),
            meter_name=("meter_name", "first"),
            site_id=("site_id", "first"),
            start_timestamp=("timestamp", "min"),
            end_timestamp=("timestamp", "max"),
            duration_rows=("segment_id", "size"),
            reading_mean=("meter_reading", "mean"),
            reading_min=("meter_reading", "min"),
            reading_max=("meter_reading", "max"),
            reading_slope=(
                "meter_reading",
                lambda x: (x.iloc[-1] - x.iloc[0]) / max(len(x) - 1, 1),
            ),
            deviation_24h=("diff_24h", "mean"),
            deviation_168h=("diff_168h", "mean"),
            diff_1h_mean=("diff_1h", "mean"),
            ratio_1h_mean=("ratio_1h", "mean"),
            tabpfn_score_movement=("tabpfn_score_delta_100000", "mean"),
            tree_score_movement=("trees_score_delta_100000", "mean"),
            tabpfn_global_rank_movement=("tabpfn_global_rank_delta_100000", "mean"),
            tree_global_rank_movement=("trees_global_rank_delta_100000", "mean"),
            tabpfn_tree_disagreement_5k=("tabpfn_minus_tree_5000", "mean"),
            tabpfn_tree_disagreement_100k=("tabpfn_minus_tree_100000", "mean"),
        )
        .reset_index()
    )
    aggregate["tabpfn_tree_disagreement_movement"] = (
        aggregate["tabpfn_tree_disagreement_100k"]
        - aggregate["tabpfn_tree_disagreement_5k"]
    )
    phases = (
        anomaly.groupby(["segment_id", "phase"], observed=True)
        .agg(
            rows=("raw_index", "size"),
            reading_mean=("meter_reading", "mean"),
            reading_slope=(
                "meter_reading",
                lambda x: (x.iloc[-1] - x.iloc[0]) / max(len(x) - 1, 1),
            ),
            tabpfn_score_movement=("tabpfn_score_delta_100000", "mean"),
            tree_score_movement=("trees_score_delta_100000", "mean"),
            tabpfn_tree_disagreement_movement=("tabpfn_minus_tree_movement", "mean"),
        )
        .reset_index()
    )
    return aggregate, phases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    all_scores: dict[str, dict[int, np.ndarray]] = {}
    reference: dict[str, np.ndarray] | None = None
    for model in ("tabpfn", "trees"):
        scores, metadata = load_scores(model)
        if reference is None:
            reference = metadata
        elif any(
            not np.array_equal(reference[key], metadata[key]) for key in reference
        ):
            raise AssertionError(f"{model}: differs from TabPFN frozen query")
        all_scores[model] = scores
        print(f"loaded {model}", flush=True)
    assert reference is not None

    base = pd.DataFrame(reference)
    base["prediction_position"] = np.arange(len(base), dtype="int64")
    raw = load_raw_fields(reference["raw_index"])
    base = base.merge(raw, on="raw_index", validate="one_to_one")
    base["meter_name"] = base["meter"].map(METER_NAMES).astype("string")
    base["reading_regime"] = reading_regimes(base)
    base = add_temporal_statistics(base)
    base.sort_values("prediction_position", inplace=True, kind="stable")
    base.reset_index(drop=True, inplace=True)

    tables: dict[str, list[pd.DataFrame]] = {
        "pairwise": [],
        "distributions": [],
        "operating": [],
        "crossings": [],
        "bootstrap": [],
        "influence": [],
    }
    rank_endpoints: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model, scores in all_scores.items():
        tables["pairwise"].append(pairwise_decomposition(model, scores, base))
        tables["distributions"].append(score_distributions(model, scores, base))
        tables["operating"].append(operating_points(model, scores, base))
        tables["crossings"].append(threshold_crossings(model, scores, base))
        bootstrap, influence = building_uncertainty(
            model, scores[5_000], scores[100_000], base, args.seed, args.bootstrap_draws
        )
        tables["bootstrap"].append(bootstrap)
        tables["influence"].append(influence)

        row = base[
            [
                "raw_index",
                "building_id",
                "site_id",
                "meter",
                "meter_name",
                "timestamp",
                "meter_reading",
                "anomaly",
                "reading_regime",
            ]
        ].copy()
        for context, score in scores.items():
            row[f"score_{context}"] = score
        global_5k, within_5k = percentile_ranks(scores[5_000], base["meter"].to_numpy())
        global_100k, within_100k = percentile_ranks(
            scores[100_000], base["meter"].to_numpy()
        )
        rank_endpoints[model] = (global_5k, global_100k)
        for context in CONTEXTS[1:]:
            global_rank, within_rank = percentile_ranks(
                scores[context], base["meter"].to_numpy()
            )
            row[f"score_delta_{context}"] = scores[context] - scores[5_000]
            row[f"global_rank_delta_{context}"] = global_rank - global_5k
            row[f"within_meter_rank_delta_{context}"] = within_rank - within_5k
        row.to_parquet(
            OUT / f"m5_137_row_score_rank_movement_{model}.parquet", index=False
        )
        print(f"wrote {model} row movements", flush=True)

    for name, frames in tables.items():
        pd.concat(frames, ignore_index=True).to_csv(
            OUT / f"m5_137_{name}.csv", index=False
        )

    for model, scores in all_scores.items():
        base[f"{model}_score_delta_100000"] = scores[100_000] - scores[5_000]
        base[f"{model}_minus_tree_5000"] = (
            scores[5_000] - all_scores["trees"][5_000] if model == "tabpfn" else 0.0
        )
        base[f"{model}_minus_tree_100000"] = (
            scores[100_000] - all_scores["trees"][100_000] if model == "tabpfn" else 0.0
        )
    for model, (rank_5k, rank_100k) in rank_endpoints.items():
        base[f"{model}_global_rank_delta_100000"] = rank_100k - rank_5k
    base["tabpfn_minus_tree_movement"] = (
        base["tabpfn_minus_tree_100000"] - base["tabpfn_minus_tree_5000"]
    )
    segments, phases = segment_outputs(base)
    segments.to_parquet(OUT / "m5_137_anomaly_segments.parquet", index=False)
    phases.to_parquet(OUT / "m5_137_anomaly_segment_phases.parquet", index=False)
    print(f"wrote {len(segments):,} segments to {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
