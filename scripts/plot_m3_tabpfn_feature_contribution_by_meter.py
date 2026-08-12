"""Render the M3 TabPFN 17-versus-137 feature comparison by meter."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lead import PROC, write_json_with_provenance


METER_SPECS = (
    (0, "electricity", "Electricity"),
    (1, "chilledwater", "Chilled Water"),
    (2, "steam", "Steam"),
    (3, "hotwater", "Hot Water"),
)
TABPFN_17_KEY = "tabpfn_17_features"
TABPFN_137_KEY = "tabpfn_137_features"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TABPFN_17 = "#d1580f"
TABPFN_137 = "#7a51a8"


@dataclass(frozen=True)
class Curve:
    x: np.ndarray
    y: np.ndarray
    score: float


@dataclass(frozen=True)
class MeterResult:
    meter: int
    slug: str
    label: str
    rows: int
    anomalies: int
    baseline_roc: Curve
    engineered_roc: Curve
    baseline_precision_recall: Curve
    engineered_precision_recall: Curve


def _load_tabpfn(path: Path) -> dict[str, np.ndarray]:
    required = {"anomaly", "site_id", "tabpfn"}
    with np.load(path) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path} missing TabPFN arrays: {sorted(missing)}")
        return {name: np.asarray(payload[name]) for name in required}


def _load_m3_metadata(path: Path) -> dict[str, np.ndarray]:
    required = {"anomaly", "site_id", "meter"}
    with np.load(path) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path} missing M3 metadata arrays: {sorted(missing)}")
        return {name: np.asarray(payload[name]) for name in required}


def load_aligned_tabpfn(
    tabpfn_17_path: Path,
    tabpfn_137_path: Path,
    m3_17_metadata_path: Path,
    m3_137_metadata_path: Path,
) -> dict[str, np.ndarray]:
    """Load TabPFN scores only after proving the borrowed meter order is exact."""
    tabpfn_17 = _load_tabpfn(tabpfn_17_path)
    tabpfn_137 = _load_tabpfn(tabpfn_137_path)
    m3_17 = _load_m3_metadata(m3_17_metadata_path)
    m3_137 = _load_m3_metadata(m3_137_metadata_path)

    for key in ("anomaly", "site_id"):
        if not np.array_equal(tabpfn_17[key], tabpfn_137[key]):
            raise ValueError(
                f"17- and 137-feature TabPFN artifacts do not align: {key}"
            )
        if not np.array_equal(tabpfn_17[key], m3_17[key]):
            raise ValueError(f"17-feature TabPFN and M3 metadata do not align: {key}")
        if not np.array_equal(tabpfn_137[key], m3_137[key]):
            raise ValueError(f"137-feature TabPFN and M3 metadata do not align: {key}")
    if not np.array_equal(m3_17["meter"], m3_137["meter"]):
        raise ValueError("17- and 137-feature M3 metadata do not align: meter")
    for name, payload in (("17-feature", tabpfn_17), ("137-feature", tabpfn_137)):
        if not np.isfinite(payload["tabpfn"]).all():
            raise ValueError(f"{name} TabPFN scores contain non-finite values")

    return {
        "anomaly": tabpfn_17["anomaly"].astype("int8", copy=False),
        "meter": m3_17["meter"].astype("int8", copy=False),
        TABPFN_17_KEY: tabpfn_17["tabpfn"],
        TABPFN_137_KEY: tabpfn_137["tabpfn"],
    }


def _compressed(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int = 1_200,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the same bounded line geometry as the established M3 meter renderer."""
    if len(x) <= max_points:
        return x, y
    positions = np.unique(np.linspace(0, len(x) - 1, max_points).round().astype(int))
    return x[positions], y[positions]


def compute_meter_results(arrays: dict[str, np.ndarray]) -> list[MeterResult]:
    """Compute the TabPFN 17-to-137 curves for each meter."""
    results: list[MeterResult] = []
    for meter, slug, label in METER_SPECS:
        mask = arrays["meter"] == meter
        labels = arrays["anomaly"][mask]
        if len(labels) == 0 or np.unique(labels).size != 2:
            raise ValueError(f"meter {meter} requires both label classes")
        curves: dict[str, tuple[Curve, Curve]] = {}
        for key in (TABPFN_17_KEY, TABPFN_137_KEY):
            scores = arrays[key][mask]
            fpr, tpr, _ = roc_curve(labels, scores)
            fpr, tpr = _compressed(fpr, tpr)
            precision, recall, _ = precision_recall_curve(labels, scores)
            recall, precision = _compressed(recall, precision)
            curves[key] = (
                Curve(fpr, tpr, float(roc_auc_score(labels, scores))),
                Curve(
                    recall, precision, float(average_precision_score(labels, scores))
                ),
            )
        results.append(
            MeterResult(
                meter=meter,
                slug=slug,
                label=label,
                rows=int(mask.sum()),
                anomalies=int(labels.sum()),
                baseline_roc=curves[TABPFN_17_KEY][0],
                engineered_roc=curves[TABPFN_137_KEY][0],
                baseline_precision_recall=curves[TABPFN_17_KEY][1],
                engineered_precision_recall=curves[TABPFN_137_KEY][1],
            )
        )
    return results


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=7, length=2.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.grid(color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_box_aspect(1)


def render_grid(
    results: list[MeterResult],
    output: Path,
    *,
    curve_type: str,
) -> None:
    """Render one shared-axis 2x2 TabPFN feature-engineering grid."""
    if curve_type not in {"roc", "precision_recall"}:
        raise ValueError(f"unsupported curve type: {curve_type}")
    if len(results) != len(METER_SPECS):
        raise ValueError("expected exactly four meter results")

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 10.0), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, result in zip(axes.ravel(), results, strict=True):
        _style_axis(ax)
        if curve_type == "roc":
            baseline = result.baseline_roc
            engineered = result.engineered_roc
            ax.plot([0, 1], [0, 1], color=AXIS, linewidth=0.7, linestyle=(0, (2, 3)))
            detail = f"ROC-AUC {baseline.score:.3f} → {engineered.score:.3f}"
        else:
            baseline = result.baseline_precision_recall
            engineered = result.engineered_precision_recall
            prevalence = result.anomalies / result.rows
            ax.axhline(prevalence, color=AXIS, linewidth=0.7, linestyle=(0, (2, 3)))
            detail = f"PR-AUC {baseline.score:.3f} → {engineered.score:.3f}"
        ax.plot(baseline.x, baseline.y, color=TABPFN_17, linewidth=1.15)
        ax.plot(engineered.x, engineered.y, color=TABPFN_137, linewidth=1.45)
        ax.set_title(
            result.label,
            loc="left",
            fontsize=12,
            fontweight="bold",
            color=INK,
            pad=27 if curve_type == "precision_recall" else 18,
        )
        ax.text(
            0,
            1.005,
            detail,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=MUTED,
        )

    metric = "ROC-AUC" if curve_type == "roc" else "PR-AUC"
    fig.suptitle(
        f"TabPFN Feature Engineering: {metric} by meters",
        x=0.06,
        y=0.965,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.915,
        "Final 50/50 building holdout · context 100k · n=8.",
        ha="left",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.supylabel(
        "True-positive rate" if curve_type == "roc" else "Precision",
        x=0.052,
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.supxlabel(
        "False-positive rate" if curve_type == "roc" else "Recall",
        x=0.54,
        y=0.10,
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.subplots_adjust(
        left=0.10, right=0.98, top=0.78, bottom=0.16, wspace=0.06, hspace=0.38
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=TABPFN_17,
                linewidth=1.6,
                label="TabPFN (17 features, context 100k, n=8)",
            ),
            Line2D(
                [0],
                [0],
                color=TABPFN_137,
                linewidth=1.8,
                label="TabPFN (137 features, context 100k, n=8)",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tabpfn-17-predictions",
        type=Path,
        default=PROC / "m5_tabpfn_17_full_test_n8_predictions.npz",
    )
    parser.add_argument(
        "--tabpfn-137-predictions",
        type=Path,
        default=PROC / "m5_tabpfn_137_full_test_n8_predictions.npz",
    )
    parser.add_argument(
        "--m3-17-metadata",
        type=Path,
        default=PROC / "m3_17_feature_ensemble_predictions_50_50.npz",
    )
    parser.add_argument(
        "--m3-137-metadata",
        type=Path,
        default=PROC / "m3_137_feature_ensemble_predictions_50_50.npz",
    )
    parser.add_argument(
        "--roc-output",
        type=Path,
        default=ROOT
        / "docs"
        / "reports"
        / "assets"
        / "m3"
        / "m3_tabpfn_feature_contribution_by_meter_roc.png",
    )
    parser.add_argument(
        "--pr-output",
        type=Path,
        default=ROOT
        / "docs"
        / "reports"
        / "assets"
        / "m3"
        / "m3_tabpfn_feature_contribution_by_meter_precision_recall.png",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=ROOT
        / "docs"
        / "metrics"
        / "m3_tabpfn_feature_contribution_by_meter.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = load_aligned_tabpfn(
        args.tabpfn_17_predictions,
        args.tabpfn_137_predictions,
        args.m3_17_metadata,
        args.m3_137_metadata,
    )
    results = compute_meter_results(arrays)
    render_grid(results, args.roc_output, curve_type="roc")
    render_grid(results, args.pr_output, curve_type="precision_recall")
    figures = {"roc": args.roc_output, "precision_recall": args.pr_output}
    summaries: dict[str, Any] = {
        result.slug: {
            "rows": result.rows,
            "anomalies": result.anomalies,
            "anomaly_rate": result.anomalies / result.rows,
            "metrics": {
                TABPFN_17_KEY: {
                    "roc_auc": result.baseline_roc.score,
                    "pr_auc": result.baseline_precision_recall.score,
                },
                TABPFN_137_KEY: {
                    "roc_auc": result.engineered_roc.score,
                    "pr_auc": result.engineered_precision_recall.score,
                },
            },
        }
        for result in results
    }
    write_json_with_provenance(
        args.metrics_out,
        {
            "experiment": "m3_tabpfn_feature_engineering_by_meter",
            "split": "50_50_mod2",
            "feature_comparison": "TabPFN, context 100k, n=8, 17 versus 137 features",
            "artifacts": {
                "tabpfn_17_predictions": str(
                    args.tabpfn_17_predictions.relative_to(ROOT)
                ),
                "tabpfn_137_predictions": str(
                    args.tabpfn_137_predictions.relative_to(ROOT)
                ),
                "m3_17_metadata": str(args.m3_17_metadata.relative_to(ROOT)),
                "m3_137_metadata": str(args.m3_137_metadata.relative_to(ROOT)),
                "figures": {
                    name: str(path.relative_to(ROOT)) for name, path in figures.items()
                },
            },
            "meters": summaries,
        },
        root=ROOT,
        provenance={
            "note": "Each TabPFN artifact is proven row-aligned to its M3 meter metadata."
        },
    )
    print(f"Saved {len(figures)} figures and {args.metrics_out}")


if __name__ == "__main__":
    main()
