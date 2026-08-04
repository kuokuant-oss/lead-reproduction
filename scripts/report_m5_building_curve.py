"""Aggregate building-budget predictions into detailed metrics and curve data.

Each input is a cell JSON with ``building_budget``, ``features``,
``sampling_profile`` and ``predictions``.  Prediction NPZ files must carry
``validation_raw_index``, ``anomaly``, ``building_id``, ``site_id``, ``meter``
and one or more probability arrays named in the JSON ``score_names`` field.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from lead import PROC


METER_NAMES = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}
REQUIRED_ARRAYS = {
    "validation_raw_index",
    "anomaly",
    "building_id",
    "site_id",
    "meter",
}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _threshold_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    label = (score >= 0.5).astype("int8")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, label, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, label, labels=[0, 1]).ravel()
    return {
        "precision_0_5": float(precision),
        "recall_0_5": float(recall),
        "f1_0_5": float(f1),
        "tn_0_5": int(tn),
        "fp_0_5": int(fp),
        "fn_0_5": int(fn),
        "tp_0_5": int(tp),
    }


def metric_and_curve_rows(
    y: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    *,
    base: dict[str, Any],
    max_curve_points: int = 2_000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_group = y[mask]
    score_group = score[mask]
    positives = int(y_group.sum())
    metric = {
        **base,
        "rows": int(len(y_group)),
        "anomalies": positives,
        "anomaly_rate": float(y_group.mean()) if len(y_group) else float("nan"),
    }
    if not 0 < positives < len(y_group):
        metric.update({"roc_auc": float("nan"), "pr_auc": float("nan")})
        return metric, []
    metric.update(
        {
            "roc_auc": float(roc_auc_score(y_group, score_group)),
            "pr_auc": float(average_precision_score(y_group, score_group)),
            **_threshold_metrics(y_group, score_group),
        }
    )
    fpr, tpr, roc_threshold = roc_curve(y_group, score_group)
    precision, recall, pr_threshold = precision_recall_curve(y_group, score_group)
    curves: list[dict[str, Any]] = []
    roc_points = np.unique(
        np.linspace(0, len(fpr) - 1, min(len(fpr), max_curve_points)).astype(int)
    )
    pr_points = np.unique(
        np.linspace(0, len(recall) - 1, min(len(recall), max_curve_points)).astype(int)
    )
    for point in roc_points:
        x, y_value = fpr[point], tpr[point]
        curves.append(
            {
                **base,
                "curve": "roc",
                "point": point,
                "x": float(x),
                "y": float(y_value),
                "threshold": (
                    float(roc_threshold[point]) if point < len(roc_threshold) else None
                ),
            }
        )
    for point in pr_points:
        x, y_value = recall[point], precision[point]
        curves.append(
            {
                **base,
                "curve": "precision_recall",
                "point": point,
                "x": float(x),
                "y": float(y_value),
                "threshold": (
                    float(pr_threshold[point]) if point < len(pr_threshold) else None
                ),
            }
        )
    return metric, curves


def aggregate_cell(
    metadata: dict[str, Any], payload: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    missing = REQUIRED_ARRAYS - set(payload)
    if missing:
        raise ValueError(f"prediction artifact missing {sorted(missing)}")
    raw = np.asarray(payload["validation_raw_index"], dtype="int64")
    if len(np.unique(raw)) != len(raw):
        raise AssertionError("prediction raw indices are not unique")
    y = np.asarray(payload["anomaly"], dtype="int8")
    for key in REQUIRED_ARRAYS - {"validation_raw_index"}:
        if len(payload[key]) != len(raw):
            raise AssertionError(f"prediction array length differs: {key}")

    score_names = list(metadata["score_names"])
    metrics: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    groupings: list[tuple[str, np.ndarray, dict[int, str]]] = [
        ("overall", np.zeros(len(raw), dtype="int8"), {0: "all"}),
        (
            "meter",
            np.asarray(payload["meter"], dtype="int64"),
            METER_NAMES,
        ),
        (
            "site",
            np.asarray(payload["site_id"], dtype="int64"),
            {
                int(value): f"site {int(value)}"
                for value in np.unique(payload["site_id"])
            },
        ),
    ]
    for score_name in score_names:
        score = np.asarray(payload[score_name], dtype="float64")
        if len(score) != len(raw) or not np.isfinite(score).all():
            raise AssertionError(f"invalid prediction scores: {score_name}")
        for grouping, keys, labels in groupings:
            for group in sorted(int(value) for value in np.unique(keys)):
                mask = keys == group
                base = {
                    "sampling_profile": metadata["sampling_profile"],
                    "building_budget": int(metadata["building_budget"]),
                    "features": int(metadata["features"]),
                    "model": score_name,
                    "grouping": grouping,
                    "group": group,
                    "group_label": labels[group],
                    "buildings": int(np.unique(payload["building_id"][mask]).size),
                }
                metric, curve = metric_and_curve_rows(y, score, mask, base=base)
                metrics.append(metric)
                curves.extend(curve)
    return metrics, curves


def load_cell(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    prediction_path = Path(metadata["predictions"])
    if not prediction_path.is_absolute():
        prediction_path = path.parent / prediction_path
    with np.load(prediction_path) as stored:
        payload = {name: np.asarray(stored[name]) for name in stored.files}
    return metadata, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs="+", type=Path)
    parser.add_argument(
        "--out-root", type=Path, default=PROC / "m5_building_curve" / "aggregate"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    canonical_identity: tuple[np.ndarray, np.ndarray] | None = None
    cells: list[dict[str, Any]] = []
    for path in args.cells:
        metadata, payload = load_cell(path)
        identity = (
            np.asarray(payload["validation_raw_index"], dtype="int64"),
            np.asarray(payload["anomaly"], dtype="int8"),
        )
        if canonical_identity is None:
            canonical_identity = identity
        elif not (
            np.array_equal(canonical_identity[0], identity[0])
            and np.array_equal(canonical_identity[1], identity[1])
        ):
            raise AssertionError(
                f"cell {path} does not use the canonical row/label order"
            )
        cell_metrics, cell_curves = aggregate_cell(metadata, payload)
        metrics.extend(cell_metrics)
        curves.extend(cell_curves)
        cells.append(
            {
                "metadata": str(path),
                "sampling_profile": metadata["sampling_profile"],
                "building_budget": int(metadata["building_budget"]),
                "features": int(metadata["features"]),
                "score_names": list(metadata["score_names"]),
            }
        )

    metrics_frame = pd.DataFrame(metrics).sort_values(
        [
            "sampling_profile",
            "features",
            "building_budget",
            "model",
            "grouping",
            "group",
        ]
    )
    curve_frame = pd.DataFrame(curves).sort_values(
        [
            "sampling_profile",
            "features",
            "building_budget",
            "model",
            "grouping",
            "group",
            "curve",
            "point",
        ]
    )
    metrics_path = args.out_root / "metrics.csv"
    curves_path = args.out_root / "curves.csv"
    _atomic_csv(metrics_frame, metrics_path)
    _atomic_csv(curve_frame, curves_path)
    summary = {
        "schema_version": 1,
        "experiment": "m5_building_count_curve_breakdown",
        "cells": cells,
        "identity_gate": "validation_raw_index and anomaly are byte-identical across cells",
        "outputs": {"metrics": str(metrics_path), "curves": str(curves_path)},
    }
    _atomic_json(summary, args.out_root / "summary.json")
    print(f"Wrote {metrics_path}, {curves_path}, and summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
