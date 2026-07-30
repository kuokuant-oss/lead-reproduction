"""Analyse pre-registered fixed-query estimands for the M5 hotwater factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lead import ROOT


OUT = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUT)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def pair_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if not len(positive) or not len(negative):
        return float("nan")
    negative = np.sort(negative)
    below = np.searchsorted(negative, positive, side="left")
    at = np.searchsorted(negative, positive, side="right")
    return float((below + 0.5 * (at - below)).mean() / len(negative))


def metrics(scores: np.ndarray, query: pd.DataFrame) -> dict[str, float]:
    work = query.copy()
    work["score"] = scores
    work["global_rank"] = work["score"].rank(method="average", pct=True)
    work["within_meter_rank"] = work.groupby("meter", sort=False)["score"].rank(
        method="average", pct=True
    )
    hw = (work["meter"] == 3) & (work["meter_reading"] <= 1.0)
    hw_pos = work.loc[hw & (work["anomaly"] == 1)]
    hw_neg = work.loc[hw & (work["anomaly"] == 0)]
    steam_pos = work.loc[
        (work["meter"] == 2) & (work["anomaly"] == 1), "score"
    ].to_numpy()
    hotwater_neg = work.loc[
        (work["meter"] == 3) & (work["anomaly"] == 0), "score"
    ].to_numpy()
    normal = work.loc[work["anomaly"] == 0, "score"].to_numpy()
    positive = work.loc[work["anomaly"] == 1, "score"].to_numpy()
    threshold = np.quantile(normal, 0.999)
    return {
        "hw01_within_rank_gap": float(
            hw_pos["within_meter_rank"].mean() - hw_neg["within_meter_rank"].mean()
        ),
        "hw01_pair_auc": pair_auc(
            hw_pos["score"].to_numpy(), hw_neg["score"].to_numpy()
        ),
        "steam_pos_vs_hw_neg_auc": pair_auc(steam_pos, hotwater_neg),
        "global_recall_at_fpr_0.001": float((positive >= threshold).mean()),
        "hw01_anomaly_mean_score": float(hw_pos["score"].mean()),
        "hw01_normal_mean_score": float(hw_neg["score"].mean()),
        "hw01_anomaly_mean_rank": float(hw_pos["within_meter_rank"].mean()),
        "hw01_normal_mean_rank": float(hw_neg["within_meter_rank"].mean()),
    }


def factor_effect(values: dict[tuple[bool, bool], float]) -> dict[str, float]:
    y00, y01, y10, y11 = (
        values[(False, False)],
        values[(False, True)],
        values[(True, False)],
        values[(True, True)],
    )
    return {
        "positive_support_main_effect": (y10 + y11 - y00 - y01) / 2,
        "negative_support_main_effect": (y01 + y11 - y00 - y10) / 2,
        "positive_x_negative_interaction": y11 - y10 - y01 + y00,
    }


def query_frame() -> pd.DataFrame:
    qpath = (
        ROOT
        / "data"
        / "processed"
        / "m5_context_stories"
        / "queries"
        / "screening"
        / "queries.npz"
    )
    with np.load(qpath) as payload:
        raw = np.asarray(payload["raw_index"], dtype="int64")
        query = pd.DataFrame(
            {
                "raw_index": raw,
                "anomaly": payload["anomaly"],
                "meter": payload["meter"],
                "site_id": payload["site_id"],
                "building_id": payload["building_id"],
            }
        )
    source = pd.read_csv(
        ROOT / "data" / "raw" / "m3" / "train.csv",
        usecols=["timestamp", "meter_reading"],
        parse_dates=["timestamp"],
    )
    fields = source.iloc[raw].reset_index(drop=True)
    return pd.concat([query, fields], axis=1)


def segment_clusters(query: pd.DataFrame) -> pd.Series:
    """Map query anomalies to full holdout consecutive-hour segment ids."""
    train = pd.read_csv(
        ROOT / "data" / "raw" / "m3" / "train.csv",
        usecols=["building_id", "meter", "timestamp"],
        parse_dates=["timestamp"],
    )
    labels = (
        pd.read_csv(ROOT / "data" / "raw" / "m3" / "bad_meter_readings.csv")
        .iloc[:, 0]
        .to_numpy(dtype="int8")
    )
    train["anomaly"] = labels
    train["raw_index"] = np.arange(len(train), dtype="int64")
    anomaly = train.loc[
        (train["building_id"] % 2 == 1) & (train["anomaly"] == 1)
    ].sort_values(["building_id", "meter", "timestamp"], kind="stable")
    gap = (
        anomaly.groupby(["building_id", "meter"], observed=True)["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )
    anomaly["segment_cluster"] = (gap.isna() | (gap > 1)).cumsum().astype("int64")
    mapping = anomaly.set_index("raw_index")["segment_cluster"]
    clusters = query["raw_index"].map(mapping).astype("object")
    clusters.loc[clusters.isna()] = "normal_" + query.loc[
        clusters.isna(), "raw_index"
    ].astype(str)
    return clusters.astype(str)


def load_scores(
    root: Path, query: pd.DataFrame
) -> tuple[pd.DataFrame, dict[tuple[str, str, int, bool, bool], np.ndarray]]:
    rows: list[dict[str, object]] = []
    score_map: dict[tuple[str, str, int, bool, bool], np.ndarray] = {}
    for result_path in sorted((root / "predictions").glob("*/*/*/*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        prediction_path = result_path.with_name("predictions.npz")
        with np.load(prediction_path) as payload:
            raw, score = (
                np.asarray(payload["raw_index"], dtype="int64"),
                np.asarray(payload["score"], dtype="float64"),
            )
        if not np.array_equal(raw, query["raw_index"].to_numpy()):
            raise AssertionError(f"query order drifted: {prediction_path}")
        name = result["factorial_cell_id"]
        positive_present = "hw_pos_present" in name
        negative_present = "hw_neg_present" in name
        key = (
            str(result["model"]),
            str(result["scaler_arm"]),
            int(result["context_seed"]),
            positive_present,
            negative_present,
        )
        score_map[key] = score
        for metric, value in metrics(score, query).items():
            rows.append(
                {
                    "model": key[0],
                    "scaler_arm": key[1],
                    "context_seed": key[2],
                    "positive_present": positive_present,
                    "negative_present": negative_present,
                    "factorial_cell_id": name,
                    "metric": metric,
                    "value": value,
                }
            )
    return pd.DataFrame(rows), score_map


def effects_table(metric_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in metric_rows.groupby(
        ["model", "scaler_arm", "context_seed", "metric"], sort=True
    ):
        lookup = {
            (bool(row.positive_present), bool(row.negative_present)): float(row.value)
            for row in group.itertuples()
        }
        if len(lookup) != 4:
            raise AssertionError(f"incomplete factorial cell set: {keys}")
        for effect, value in factor_effect(lookup).items():
            rows.append(
                {
                    "model": keys[0],
                    "scaler_arm": keys[1],
                    "context_seed": keys[2],
                    "metric": keys[3],
                    "effect": effect,
                    "estimate": value,
                }
            )
    effects = pd.DataFrame(rows)
    frozen = effects.loc[effects["scaler_arm"] == "frozen_reference"].rename(
        columns={"estimate": "frozen_estimate"}
    )
    cell = effects.loc[effects["scaler_arm"] == "cell_specific"].merge(
        frozen[["model", "context_seed", "metric", "effect", "frozen_estimate"]],
        on=["model", "context_seed", "metric", "effect"],
        how="left",
    )
    cell["scaler_arm_interaction"] = cell["estimate"] - cell["frozen_estimate"]
    return effects.merge(
        cell[
            [
                "model",
                "scaler_arm",
                "context_seed",
                "metric",
                "effect",
                "scaler_arm_interaction",
            ]
        ],
        on=["model", "scaler_arm", "context_seed", "metric", "effect"],
        how="left",
    )


def bootstrap(
    score_map: dict[tuple[str, str, int, bool, bool], np.ndarray],
    query: pd.DataFrame,
    clusters: pd.Series,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for model, arm, context_seed in sorted({key[:3] for key in score_map}):
        for cluster_kind, cluster_values in (
            ("building", query["building_id"].astype(str)),
            ("segment", clusters),
        ):
            local_units = {
                name: np.flatnonzero(cluster_values.to_numpy() == name)
                for name in np.unique(cluster_values)
            }
            local_names = list(local_units)
            samples: dict[str, list[float]] = {
                name: []
                for name in (
                    "hw01_within_rank_gap",
                    "hw01_pair_auc",
                    "steam_pos_vs_hw_neg_auc",
                    "global_recall_at_fpr_0.001",
                )
            }
            for _ in range(draws):
                chosen = rng.choice(local_names, size=len(local_names), replace=True)
                index = np.concatenate([local_units[name] for name in chosen])
                q = query.iloc[index].reset_index(drop=True)
                for metric_name in samples:
                    values = {
                        pair: metrics(
                            score_map[(model, arm, context_seed, *pair)][index], q
                        )[metric_name]
                        for pair in (
                            (False, False),
                            (False, True),
                            (True, False),
                            (True, True),
                        )
                    }
                    value = factor_effect(values)["positive_x_negative_interaction"]
                    if np.isfinite(value):
                        samples[metric_name].append(value)
            for metric_name, values in samples.items():
                if not values:
                    raise AssertionError(
                        f"bootstrap lost every valid {metric_name} draw"
                    )
                rows.append(
                    {
                        "model": model,
                        "scaler_arm": arm,
                        "context_seed": context_seed,
                        "cluster": cluster_kind,
                        "metric": metric_name,
                        "effect": "positive_x_negative_interaction",
                        "bootstrap_draws": draws,
                        "bootstrap_q025": np.quantile(values, 0.025),
                        "bootstrap_q975": np.quantile(values, 0.975),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    query = query_frame()
    metric_rows, score_map = load_scores(args.root, query)
    if len(score_map) != 48:
        raise AssertionError(
            f"expected 48 model/scaler/seed/cell predictions, found {len(score_map)}"
        )
    effects = effects_table(metric_rows)
    clusters = segment_clusters(query)
    bootstrap_rows = bootstrap(
        score_map, query, clusters, args.bootstrap_draws, args.seed
    )
    reports = args.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    metric_rows.to_csv(reports / "factorial_cell_estimands.csv", index=False)
    effects.to_csv(reports / "factorial_effects.csv", index=False)
    bootstrap_rows.to_csv(reports / "factorial_cluster_bootstrap.csv", index=False)
    print(
        f"wrote {len(metric_rows)} cell estimands, {len(effects)} factorial effects, and {len(bootstrap_rows)} clustered intervals"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
