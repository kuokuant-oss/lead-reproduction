"""Retrospectively decompose the existing M5 context-size predictions.

This does not fit models. It reads the already-scored 5k--100k full-holdout
artifacts and resolves their site/meter aggregates into meter x site cells,
class-conditional score movement, and positive-meter x negative-meter AUCs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


CONTEXTS = (5_000, 10_000, 20_000, 50_000, 100_000)
FEATURES = ("17", "137")
MODELS = ("tabpfn", "trees")
METER_NAMES = {
    0: "electricity",
    1: "chilledwater",
    2: "steam",
    3: "hotwater",
}


def prediction_path(model: str, feature: str, context: int) -> Path:
    if model == "tabpfn":
        suffix = (
            "n8_predictions.npz"
            if context == 100_000
            else f"context{context}_n8_predictions.npz"
        )
        return PROC / f"m5_tabpfn_{feature}_full_test_{suffix}"
    return PROC / f"m5_tree_ensemble_f{feature}_context{context}_predictions.npz"


def load_prediction(
    model: str, feature: str, context: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = prediction_path(model, feature, context)
    with np.load(path) as payload:
        raw_index = np.asarray(
            payload["raw_index" if model == "tabpfn" else "validation_raw_index"],
            dtype="int64",
        )
        anomaly = np.asarray(payload["anomaly"], dtype="int8")
        site_id = np.asarray(payload["site_id"], dtype="int8")
        score = np.asarray(
            payload["tabpfn" if model == "tabpfn" else "ensemble"],
            dtype="float32",
        )
    return raw_index, anomaly, site_id, score


def load_meter(raw_index: np.ndarray) -> np.ndarray:
    train_path = ROOT / "data" / "raw" / "m3" / "train.csv"
    frame = pd.read_csv(train_path, usecols=["meter"], dtype={"meter": "int8"})
    if int(raw_index.max()) >= len(frame):
        raise AssertionError("raw_index exceeds frozen M3 frame")
    meter = frame["meter"].to_numpy()[raw_index]
    if not set(np.unique(meter)).issubset(METER_NAMES):
        raise AssertionError("unexpected meter code")
    return meter


def cell_bounds(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(cell, kind="stable")
    values, starts, counts = np.unique(
        cell[order], return_index=True, return_counts=True
    )
    return order, values, np.column_stack((starts, starts + counts))


def cell_metrics(
    *,
    model: str,
    feature: str,
    context: int,
    anomaly: np.ndarray,
    score: np.ndarray,
    order: np.ndarray,
    cells: np.ndarray,
    bounds: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell, (start, stop) in zip(cells, bounds, strict=True):
        idx = order[start:stop]
        y = anomaly[idx]
        values = score[idx]
        positives = int(y.sum())
        if positives == 0 or positives == len(y):
            continue
        meter, site_id = divmod(int(cell), 16)
        rows.append(
            {
                "model": model,
                "feature_tag": f"F{0 if feature == '17' else 4}",
                "context_rows": context,
                "meter": meter,
                "meter_name": METER_NAMES[meter],
                "site_id": site_id,
                "rows": len(idx),
                "positive": positives,
                "pr_auc": float(average_precision_score(y, values)),
                "roc_auc": float(roc_auc_score(y, values)),
                "positive_mean_score": float(values[y == 1].mean()),
                "negative_mean_score": float(values[y == 0].mean()),
                "mean_score_separation": float(
                    values[y == 1].mean() - values[y == 0].mean()
                ),
            }
        )
    return rows


def score_contrasts(
    *,
    model: str,
    feature: str,
    anomaly: np.ndarray,
    score_5k: np.ndarray,
    score_100k: np.ndarray,
    order: np.ndarray,
    cells: np.ndarray,
    bounds: np.ndarray,
) -> list[dict[str, object]]:
    delta = score_100k.astype("float64") - score_5k
    rows: list[dict[str, object]] = []
    for cell, (start, stop) in zip(cells, bounds, strict=True):
        idx = order[start:stop]
        meter, site_id = divmod(int(cell), 16)
        for label in (0, 1):
            values = delta[idx[anomaly[idx] == label]]
            if not len(values):
                continue
            rows.append(
                {
                    "model": model,
                    "feature_tag": f"F{0 if feature == '17' else 4}",
                    "contrast": "100k_minus_5k",
                    "meter": meter,
                    "meter_name": METER_NAMES[meter],
                    "site_id": site_id,
                    "anomaly": label,
                    "rows": len(values),
                    "mean_delta": float(values.mean()),
                    "median_delta": float(np.median(values)),
                    "q10_delta": float(np.quantile(values, 0.10)),
                    "q90_delta": float(np.quantile(values, 0.90)),
                    "mean_abs_delta": float(np.abs(values).mean()),
                    "fraction_increased": float((values > 0).mean()),
                }
            )
    return rows


def pair_auc(positive: np.ndarray, negative_sorted: np.ndarray) -> float:
    below = np.searchsorted(negative_sorted, positive, side="left")
    at_or_below = np.searchsorted(negative_sorted, positive, side="right")
    wins = below.astype("float64") + 0.5 * (at_or_below - below)
    return float(wins.mean() / len(negative_sorted))


def meter_pair_auc(
    *,
    model: str,
    feature: str,
    context: int,
    anomaly: np.ndarray,
    meter: np.ndarray,
    score: np.ndarray,
) -> list[dict[str, object]]:
    positives = {
        group: score[(anomaly == 1) & (meter == group)] for group in METER_NAMES
    }
    negatives = {
        group: np.sort(score[(anomaly == 0) & (meter == group)])
        for group in METER_NAMES
    }
    total_pairs = int((anomaly == 1).sum()) * int((anomaly == 0).sum())
    rows: list[dict[str, object]] = []
    for positive_group, positive in positives.items():
        for negative_group, negative in negatives.items():
            weight = len(positive) * len(negative) / total_pairs
            auc = pair_auc(positive, negative)
            rows.append(
                {
                    "model": model,
                    "feature_tag": f"F{0 if feature == '17' else 4}",
                    "context_rows": context,
                    "positive_meter": positive_group,
                    "positive_meter_name": METER_NAMES[positive_group],
                    "negative_meter": negative_group,
                    "negative_meter_name": METER_NAMES[negative_group],
                    "positive_rows": len(positive),
                    "negative_rows": len(negative),
                    "pair_weight": weight,
                    "pair_auc": auc,
                    "weighted_auc_contribution": weight * auc,
                    "within_meter": positive_group == negative_group,
                }
            )
    return rows


def meter_site_pair_auc(
    *,
    model: str,
    feature: str,
    context: int,
    anomaly: np.ndarray,
    cell: np.ndarray,
    score: np.ndarray,
) -> list[dict[str, object]]:
    groups = sorted(int(value) for value in np.unique(cell))
    positives = {group: score[(anomaly == 1) & (cell == group)] for group in groups}
    negatives = {
        group: np.sort(score[(anomaly == 0) & (cell == group)]) for group in groups
    }
    total_pairs = int((anomaly == 1).sum()) * int((anomaly == 0).sum())
    rows: list[dict[str, object]] = []
    for positive_group, positive in positives.items():
        if not len(positive):
            continue
        positive_meter, positive_site = divmod(positive_group, 16)
        for negative_group, negative in negatives.items():
            if not len(negative):
                continue
            negative_meter, negative_site = divmod(negative_group, 16)
            weight = len(positive) * len(negative) / total_pairs
            auc = pair_auc(positive, negative)
            rows.append(
                {
                    "model": model,
                    "feature_tag": f"F{0 if feature == '17' else 4}",
                    "context_rows": context,
                    "positive_meter": positive_meter,
                    "positive_meter_name": METER_NAMES[positive_meter],
                    "positive_site_id": positive_site,
                    "negative_meter": negative_meter,
                    "negative_meter_name": METER_NAMES[negative_meter],
                    "negative_site_id": negative_site,
                    "positive_rows": len(positive),
                    "negative_rows": len(negative),
                    "pair_weight": weight,
                    "pair_auc": auc,
                    "weighted_auc_contribution": weight * auc,
                    "within_cell": positive_group == negative_group,
                }
            )
    return rows


def main() -> int:
    report_root = PROC / "m5_context_stories" / "reports"
    report_root.mkdir(parents=True, exist_ok=True)

    reference_index, reference_y, reference_site, _ = load_prediction(
        "tabpfn", "17", 5_000
    )
    meter = load_meter(reference_index)
    cell = meter.astype("int16") * 16 + reference_site
    order, cells, bounds = cell_bounds(cell)

    metric_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    cell_pair_rows: list[dict[str, object]] = []
    for model in MODELS:
        for feature in FEATURES:
            endpoint_scores: dict[int, np.ndarray] = {}
            for context in CONTEXTS:
                raw_index, anomaly, site_id, score = load_prediction(
                    model, feature, context
                )
                if not np.array_equal(raw_index, reference_index):
                    raise AssertionError(f"{model}/{feature}/{context}: row mismatch")
                if not np.array_equal(anomaly, reference_y):
                    raise AssertionError(f"{model}/{feature}/{context}: label mismatch")
                if not np.array_equal(site_id, reference_site):
                    raise AssertionError(f"{model}/{feature}/{context}: site mismatch")
                metric_rows.extend(
                    cell_metrics(
                        model=model,
                        feature=feature,
                        context=context,
                        anomaly=anomaly,
                        score=score,
                        order=order,
                        cells=cells,
                        bounds=bounds,
                    )
                )
                if context in (5_000, 100_000):
                    endpoint_scores[context] = score
                    pair_rows.extend(
                        meter_pair_auc(
                            model=model,
                            feature=feature,
                            context=context,
                            anomaly=anomaly,
                            meter=meter,
                            score=score,
                        )
                    )
                    cell_pair_rows.extend(
                        meter_site_pair_auc(
                            model=model,
                            feature=feature,
                            context=context,
                            anomaly=anomaly,
                            cell=cell,
                            score=score,
                        )
                    )
                print(f"processed {model}/{feature}/{context:,}")
            contrast_rows.extend(
                score_contrasts(
                    model=model,
                    feature=feature,
                    anomaly=reference_y,
                    score_5k=endpoint_scores[5_000],
                    score_100k=endpoint_scores[100_000],
                    order=order,
                    cells=cells,
                    bounds=bounds,
                )
            )

    outputs = {
        "m5_context_curve_meter_site_metrics.csv": pd.DataFrame(metric_rows),
        "m5_context_curve_meter_site_label_score_contrasts.csv": pd.DataFrame(
            contrast_rows
        ),
        "m5_context_curve_pairwise_meter_auc.csv": pd.DataFrame(pair_rows),
        "m5_context_curve_pairwise_meter_site_auc.csv": pd.DataFrame(cell_pair_rows),
    }
    for filename, frame in outputs.items():
        path = report_root / filename
        frame.to_csv(path, index=False)
        print(f"{filename}: {len(frame):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
