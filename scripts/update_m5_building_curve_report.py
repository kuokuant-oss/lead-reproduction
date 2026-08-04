"""Regenerate the cumulative and transparent M5 building-count report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lead import PROC, ROOT
from report_m5_building_curve import _atomic_csv, _atomic_json, aggregate_cell, load_cell

SHARED_PROC = ROOT.parent / "lead-reproduction" / "data" / "processed"
MATCHED_TABPFN = SHARED_PROC / "m5_tabpfn_137_full_test_context50000_n8_predictions.npz"
CANONICAL = SHARED_PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, default=PROC / "m5_building_curve" / "formal")
    parser.add_argument("--aggregate-root", type=Path, default=PROC / "m5_building_curve" / "aggregate")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "reports" / "m5-building-count-experiment.md")
    return parser.parse_args()


def _matched_baseline() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not MATCHED_TABPFN.is_file() or not CANONICAL.is_file():
        raise FileNotFoundError("matched-context TabPFN or canonical holdout is missing")
    with np.load(MATCHED_TABPFN) as source, np.load(CANONICAL) as canonical:
        if not np.array_equal(source["raw_index"], canonical["validation_raw_index"]):
            raise AssertionError("matched-context raw order differs from canonical")
        if not np.array_equal(source["anomaly"], canonical["anomaly"]):
            raise AssertionError("matched-context labels differ from canonical")
        payload = {
            "validation_raw_index": np.asarray(source["raw_index"]),
            "anomaly": np.asarray(source["anomaly"]),
            "building_id": np.asarray(canonical["building_id"]),
            "site_id": np.asarray(canonical["site_id"]),
            "meter": np.asarray(canonical["meter"]),
            "tabpfn_matched_50k": np.asarray(source["tabpfn"]),
        }
    metadata = {
        "sampling_profile": "matched_context_rows",
        "building_budget": 0,
        "features": 137,
        "score_names": ["tabpfn_matched_50k"],
    }
    return metadata, payload


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def _table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame.empty:
        return ["_No completed data yet._", ""]
    rows = frame.loc[:, columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for values in rows.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")
    return [*lines, ""]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    metrics: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    baseline_meta, baseline_payload = _matched_baseline()
    cell_metrics, cell_curves = aggregate_cell(baseline_meta, baseline_payload)
    metrics.extend(cell_metrics)
    curves.extend(cell_curves)
    identity = (
        baseline_payload["validation_raw_index"],
        baseline_payload["anomaly"],
    )
    del baseline_payload

    for path in sorted(args.formal_root.rglob("cell.json")):
        if not (path.parent / "COMPLETE.json").is_file():
            continue
        metadata, payload = load_cell(path)
        if not (
            np.array_equal(identity[0], payload["validation_raw_index"])
            and np.array_equal(identity[1], payload["anomaly"])
        ):
            raise AssertionError(f"non-canonical holdout identity: {path}")
        cell_metrics, cell_curves = aggregate_cell(metadata, payload)
        metrics.extend(cell_metrics)
        curves.extend(cell_curves)
        cells.append({"path": str(path), "metadata": metadata})

    metric_frame = pd.DataFrame(metrics).sort_values(
        ["sampling_profile", "features", "building_budget", "model", "grouping", "group"]
    )
    curve_frame = pd.DataFrame(curves).sort_values(
        ["sampling_profile", "features", "building_budget", "model", "grouping", "group", "curve", "point"]
    )
    _atomic_csv(metric_frame, args.aggregate_root / "metrics.csv")
    _atomic_csv(curve_frame, args.aggregate_root / "curves.csv")
    _atomic_json(
        {
            "schema_version": 1,
            "completed_cells": [cell["path"] for cell in cells],
            "matched_context_tabpfn": str(MATCHED_TABPFN),
            "identity_gate": "byte-identical raw_index and anomaly",
        },
        args.aggregate_root / "summary.json",
    )

    lines = [
        "# M5 building-count experiment",
        "",
        "This report is atomically regenerated at every overnight publication gate. "
        "Training sources are restricted to building_id % 2 == 0; the complete "
        "canonical holdout contains only odd building IDs.",
        "",
        "K=10/20/50/100 uses only 137 features. Average allocation is at most 500 "
        "rows per building, so total context is bounded by 5K/10K/25K/50K. Allocation "
        "within each incremental K block is proportional to each building's available "
        "rows, so individual buildings may contribute above or below 500. Rows are "
        "selected by a seed-42 stable hash of raw "
        "identity without consulting labels. Building and row sets are strict nested "
        "prefixes. Trees use building-disjoint 80/20 fit and early-stop roles to choose "
        "iteration counts, then final-refit on every selected row. TabPFN uses the same "
        "selected rows and has no "
        "task-specific epoch or weight-update loop, so early stopping is not applicable.",
        "",
        "The direct TabPFN reference is the m5-matched-context-breakdown 50K / "
        "137-feature / n_estimators=8 prediction artifact. Different context row/class "
        "distributions are allowed. Comparability is maintained by the same feature "
        "pipeline, scaler, checkpoint/config, canonical holdout identity and evaluator.",
        "",
        "## Execution status",
        "",
    ]
    wanted = [
        ("tree full", 725, 17),
        ("tree full", 725, 137),
        *[(model, k, 137) for k in (10, 20, 50, 100) for model in ("tree", "tabpfn")],
    ]
    statuses = []
    for label, budget, features in wanted:
        found = [
            cell
            for cell in cells
            if int(cell["metadata"]["building_budget"]) == budget
            and int(cell["metadata"]["features"]) == features
            and (("tabpfn" in cell["metadata"]["experiment"]) == (label == "tabpfn"))
        ]
        statuses.append({"stage": f"{label} K={budget} f={features}", "status": "complete" if found else "pending"})
    lines.extend(_table(pd.DataFrame(statuses), ["stage", "status"]))

    lines.extend(["## Overall PR-AUC / ROC-AUC", ""])
    overall = metric_frame.loc[metric_frame["grouping"].eq("overall")].copy()
    overall["source"] = np.where(
        overall["sampling_profile"].eq("matched_context_rows"),
        "matched-context baseline",
        "building experiment",
    )
    lines.extend(_table(overall, ["source", "building_budget", "features", "model", "rows", "anomalies", "pr_auc", "roc_auc"]))

    for grouping, title in (("meter", "Meter breakdown"), ("site", "Site breakdown")):
        lines.extend([f"## {title}", ""])
        selected = metric_frame.loc[metric_frame["grouping"].eq(grouping)]
        lines.extend(_table(selected, ["sampling_profile", "building_budget", "features", "model", "group_label", "rows", "anomalies", "pr_auc", "roc_auc"]))

    lines.extend(["## Tree early-stopping audit", ""])
    fit_rows: list[dict[str, Any]] = []
    for cell in cells:
        metadata = cell["metadata"]
        for model, fit in metadata.get("fit", {}).get("records", {}).items():
            fit_rows.append(
                {
                    "K": metadata["building_budget"],
                    "features": metadata["features"],
                    "model": model,
                    "best_iteration": fit["best_iteration"],
                    "ceiling": fit["iteration_ceiling"],
                    "stop_reason": fit["stop_reason"],
                    "ES_PR_AUC": fit["early_stop_pr_auc"],
                    "ES_ROC_AUC": fit["early_stop_roc_auc"],
                }
            )
    lines.extend(_table(pd.DataFrame(fit_rows), ["K", "features", "model", "best_iteration", "ceiling", "stop_reason", "ES_PR_AUC", "ES_ROC_AUC"]))

    lines.extend(["## K composition and selected-building audit", ""])
    for budget in manifest["budgets"]:
        cell = manifest["cells"][str(budget)]
        lines.extend(
            [
                f"### K={budget}",
                "",
                f"- Selected rows: {cell['available_rows']:,}; anomalies: {cell['available_anomalies']:,}; anomaly rate: {cell['available_anomaly_rate']:.6f}.",
                f"- Site row composition: {json.dumps(cell['available_site_counts'], sort_keys=True)}.",
                f"- Meter row composition: {json.dumps(cell['available_meter_counts'], sort_keys=True)}.",
                f"- Anomalous building-meter pairs: {cell['available_anomalous_building_meter_pairs']}/{cell['available_building_meter_pairs']} ({cell['available_anomalous_building_meter_rate']:.6f}).",
                "",
            ]
        )
        audit = pd.DataFrame(cell["selected_building_audit"])
        audit["meters"] = audit["meter_types"].map(lambda values: ",".join(map(str, values)))
        lines.extend(
            _table(
                audit,
                ["position", "site_id", "building_id", "primary_use", "role", "selection_reason", "row_allocation_reason", "meters", "allocated_row_quota", "selected_rows", "selected_anomalies", "selected_anomaly_rate", "available_rows", "available_anomaly_rate"],
            )
        )

    lines.extend(
        [
            "## Curve artifacts and reproducibility",
            "",
            f"- Detailed metrics: {args.aggregate_root / 'metrics.csv'}.",
            f"- Plot-ready ROC/PR points for overall, every meter and every site: {args.aggregate_root / 'curves.csv'}.",
            "- Every model cell stores full predictions, provenance, heartbeats, model "
            "checkpoints, prediction chunks and an atomic COMPLETE.json.",
            "- Report rows are accepted only when holdout raw-index and labels are "
            "byte-identical to the matched-context baseline.",
            "",
        ]
    )
    _atomic_text(args.report, "\n".join(lines))
    print(f"Wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
