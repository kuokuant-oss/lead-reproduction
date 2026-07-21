"""Aggregate completed M6 cells into one plot-ready JSON artifact.

This script performs no model fitting and does not copy the large exact-score
arrays. It indexes their NPZ paths while flattening metrics, curves, per-site
slices, learning-curve points, matched-support cells, oracle gaps, and timing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lead import ROOT, write_json_with_provenance


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def artifact_path(path: Path) -> str:
    resolved = resolve_path(path)
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def cell_id(payload: dict[str, Any]) -> str:
    split = payload.get("split", {})
    parts = [
        str(payload.get("cell", "unknown")),
        str(payload.get("variant", "canonical")),
        str(split.get("name", "split")),
    ]
    for key in ("fold", "site_id", "b2_positive_budget", "b2_selection_seed"):
        if key in split:
            parts.append(f"{key}-{split[key]}")
    selection = payload.get("selection") or {}
    for key in ("budget", "seed"):
        if key in selection:
            parts.append(f"{key}-{selection[key]}")
    return "__".join(parts)


def flatten_cell(payload: dict[str, Any], source: str) -> dict[str, Any]:
    if payload.get("status") != "completed":
        raise ValueError(f"M6 aggregate requires completed cells: {source}")
    identifier = cell_id(payload)
    split = payload.get("split", {})
    model_metrics = []
    threshold_rows = []
    curve_rows = []
    for model, metrics in payload.get("metrics", {}).items():
        model_metrics.append(
            {
                "cell_id": identifier,
                "model": model,
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
            }
        )
        threshold = metrics.get("threshold_0_5")
        if threshold is not None:
            threshold_rows.append(
                {
                    "cell_id": identifier,
                    "model": model,
                    "threshold": 0.5,
                    **threshold,
                }
            )
        for curve_type, curve in payload.get("curves", {}).get(model, {}).items():
            curve_rows.append(
                {
                    "cell_id": identifier,
                    "model": model,
                    "curve_type": curve_type,
                    "x": curve.get("x", []),
                    "y": curve.get("y", []),
                }
            )

    site_rows = []
    for site, site_payload in payload.get("slices", {}).get("by_site_id", {}).items():
        for model, metrics in site_payload.get("models", {}).items():
            site_rows.append(
                {
                    "cell_id": identifier,
                    "site_id": int(site),
                    "model": model,
                    **metrics,
                }
            )

    fixed_recall_rows = [
        {
            "cell_id": identifier,
            "model": model,
            **operating_point,
        }
        for model, operating_point in (
            payload.get("operating_points", {})
            .get("source_calibrated_recall_0_90", {})
            .items()
        )
    ]

    selection = payload.get("selection") or {}
    fit = payload.get("fit", {})
    learning_curve = None
    if payload.get("cell") == "b1":
        learning_curve = {
            "cell_id": identifier,
            "split_name": split.get("name"),
            "budget": selection.get("budget"),
            "seed": selection.get("seed"),
            "site_allocation": selection.get("site_allocation"),
            "model_metrics": model_metrics,
            "macro_site_metrics": payload.get("macro_site_metrics"),
        }
    matched_support = None
    if payload.get("cell") == "b2":
        matched_support = {
            "cell_id": identifier,
            "split_name": split.get("name"),
            "unique_anomaly_rows": fit.get("unique_anomaly_rows"),
            "selection_seed": fit.get("selection_seed"),
            "fit_rows": fit.get("fit_rows"),
            "model_metrics": model_metrics,
            "macro_site_metrics": payload.get("macro_site_metrics"),
        }

    return {
        "cell": {
            "cell_id": identifier,
            "source": source,
            "cell": payload.get("cell"),
            "variant": payload.get("variant", "canonical"),
            "split": split,
            "selection": payload.get("selection"),
            "support": payload.get("support"),
            "macro_site_metrics": payload.get("macro_site_metrics"),
            "score_histograms": payload.get("score_histograms"),
            "predictions": payload.get("artifacts", {}).get("predictions"),
            "prepared_manifest": payload.get("artifacts", {}).get("prepared_manifest"),
        },
        "model_metrics": model_metrics,
        "threshold_0_5": threshold_rows,
        "fixed_recall_0_90": fixed_recall_rows,
        "curves": curve_rows,
        "site_metrics": site_rows,
        "learning_curve": learning_curve,
        "matched_support": matched_support,
        "runtime": {
            "cell_id": identifier,
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "timing_breakdown": payload.get("timing_breakdown"),
            "model_fit_seconds": fit.get("model_fit_seconds"),
            "prediction_seconds": payload.get("prediction_seconds"),
            "matrix_profile": payload.get("matrix_profile"),
        },
    }


def aggregate_payloads(
    cells: list[tuple[str, dict[str, Any]]],
    oracle_comparisons: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    flattened = [flatten_cell(payload, source) for source, payload in cells]
    identifiers = [item["cell"]["cell_id"] for item in flattened]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("M6 aggregate received duplicate cell IDs")
    oracle_rows = []
    for source, payload in oracle_comparisons:
        if payload.get("experiment") != "m6_paired_in_site_oracle":
            raise ValueError(f"Not an M6 oracle comparison: {source}")
        for model, metrics in payload.get("models", {}).items():
            oracle_rows.append(
                {
                    "source": source,
                    "site_id": payload["site_id"],
                    "model": model,
                    "n_rows": payload["n_rows"],
                    "n_anomalies": payload["n_anomalies"],
                    "paired_eval_key_sha256": payload["paired_eval_key_sha256"],
                    "cross_site": metrics["cross_site"],
                    "in_site_oracle": metrics["in_site_oracle"],
                    "oracle_minus_cross_pr_auc": metrics["oracle_minus_cross_pr_auc"],
                }
            )
    return {
        "schema_version": 1,
        "experiment": "m6_site_transfer_plot_data",
        "cells": [item["cell"] for item in flattened],
        "model_metrics": [row for item in flattened for row in item["model_metrics"]],
        "threshold_0_5": [row for item in flattened for row in item["threshold_0_5"]],
        "fixed_recall_0_90": [
            row for item in flattened for row in item["fixed_recall_0_90"]
        ],
        "curves": [row for item in flattened for row in item["curves"]],
        "site_metrics": [row for item in flattened for row in item["site_metrics"]],
        "learning_curves": [
            item["learning_curve"]
            for item in flattened
            if item["learning_curve"] is not None
        ],
        "matched_anomaly_support": [
            item["matched_support"]
            for item in flattened
            if item["matched_support"] is not None
        ],
        "paired_oracle": oracle_rows,
        "runtime": [item["runtime"] for item in flattened],
        "plot_data_contract": {
            "families": {
                "pooled_roc_pr": ["model_metrics", "curves"],
                "split_robustness": ["model_metrics", "cells"],
                "per_site_forest": ["site_metrics"],
                "site_by_model_heatmap": ["site_metrics"],
                "pr_auc_vs_prevalence": ["site_metrics"],
                "threshold_0_5_confusion": ["threshold_0_5"],
                "source_calibrated_recall_0_90": ["fixed_recall_0_90"],
                "paired_in_site_oracle": ["paired_oracle"],
                "training_meter_learning_curve": ["learning_curves"],
                "matched_anomaly_support": ["matched_anomaly_support"],
                "runtime_and_matrix_cost": ["runtime"],
                "score_distributions": ["cells.score_histograms"],
            },
            "exact_score_arrays": "referenced by cells[].predictions",
            "fixed_recall_0_90": (
                "available for cells run with --source-calibration; threshold "
                "is learned without test labels"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--oracle-inputs", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cell_paths = [resolve_path(path) for path in args.inputs]
    oracle_paths = [resolve_path(path) for path in args.oracle_inputs]
    payload = aggregate_payloads(
        [(artifact_path(path), read_json(path)) for path in cell_paths],
        [(artifact_path(path), read_json(path)) for path in oracle_paths],
    )
    out = resolve_path(args.out)
    write_json_with_provenance(
        out,
        payload,
        root=ROOT,
        provenance={
            "command": (
                ".\\.venv\\Scripts\\python.exe "
                "scripts/aggregate_m6_site_transfer.py --inputs "
                + " ".join(artifact_path(path) for path in cell_paths)
                + (" --oracle-inputs " if oracle_paths else "")
                + " ".join(artifact_path(path) for path in oracle_paths)
                + f" --out {artifact_path(out)}"
            ),
            "note": "Plot-data aggregation only; no model fitting ran.",
        },
    )
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
