"""Render the M3 17-versus-137 Tree Ensemble curves separately by meter."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import matplotlib
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
from plot_m3_figures import render_discrimination_curve


METER_SPECS = (
    (0, "electricity", "Electricity"),
    (1, "chilledwater", "Chilled Water"),
    (2, "steam", "Steam"),
    (3, "hotwater", "Hot Water"),
)
BASELINE_KEY = "m3_1_ensemble"
ENGINEERED_KEY = "ensemble"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BASELINE = "#898781"
ENGINEERED = "#2a78d6"


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


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    required = {"anomaly", "meter", "row_identity", "ensemble"}
    with np.load(path) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        return {name: np.asarray(payload[name]) for name in required}


def load_aligned_ensembles(
    baseline_path: Path,
    engineered_path: Path,
) -> dict[str, np.ndarray]:
    """Load both ensembles only when labels and per-row meter identity agree."""
    baseline = _load_arrays(baseline_path)
    engineered = _load_arrays(engineered_path)
    for key in ("anomaly", "meter", "row_identity"):
        if not np.array_equal(baseline[key], engineered[key]):
            raise ValueError(f"17- and 137-feature artifacts do not align: {key}")
    if not np.isfinite(baseline["ensemble"]).all():
        raise ValueError("17-feature ensemble contains non-finite scores")
    if not np.isfinite(engineered["ensemble"]).all():
        raise ValueError("137-feature ensemble contains non-finite scores")
    return {
        "anomaly": baseline["anomaly"].astype("int8", copy=False),
        "meter": baseline["meter"].astype("int8", copy=False),
        BASELINE_KEY: baseline["ensemble"],
        ENGINEERED_KEY: engineered["ensemble"],
    }


def meter_curve_data(
    arrays: dict[str, np.ndarray],
    meter: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute the existing renderer's ROC and PR inputs for one meter."""
    mask = arrays["meter"] == meter
    labels = arrays["anomaly"][mask]
    if len(labels) == 0 or np.unique(labels).size != 2:
        raise ValueError(f"meter {meter} requires both label classes")
    metrics: dict[str, dict[str, float]] = {}
    curves: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for key in (BASELINE_KEY, ENGINEERED_KEY):
        scores = arrays[key][mask]
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        metrics[key] = {
            "roc_auc": float(roc_auc_score(labels, scores)),
            "pr_auc": float(average_precision_score(labels, scores)),
        }
        curves[key] = {
            "roc": {"x": fpr, "y": tpr},
            "precision_recall": {"x": recall, "y": precision},
        }
    data = {"metrics": metrics, "curves": curves}
    summary = {
        "rows": int(mask.sum()),
        "anomalies": int(labels.sum()),
        "anomaly_rate": float(labels.mean()),
        "metrics": metrics,
    }
    return data, summary


def _compressed(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int = 1_200,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the same bounded line geometry as the established site renderer."""
    if len(x) <= max_points:
        return x, y
    positions = np.unique(np.linspace(0, len(x) - 1, max_points).round().astype(int))
    return x[positions], y[positions]


def compute_meter_results(arrays: dict[str, np.ndarray]) -> list[MeterResult]:
    """Compute 17-to-137 curves for the four meter panels."""
    results: list[MeterResult] = []
    for meter, slug, label in METER_SPECS:
        mask = arrays["meter"] == meter
        labels = arrays["anomaly"][mask]
        if len(labels) == 0 or np.unique(labels).size != 2:
            raise ValueError(f"meter {meter} requires both label classes")
        curves: dict[str, tuple[Curve, Curve]] = {}
        for key in (BASELINE_KEY, ENGINEERED_KEY):
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
                baseline_roc=curves[BASELINE_KEY][0],
                engineered_roc=curves[ENGINEERED_KEY][0],
                baseline_precision_recall=curves[BASELINE_KEY][1],
                engineered_precision_recall=curves[ENGINEERED_KEY][1],
            )
        )
    return results


def _style_axis(ax: plt.Axes) -> None:
    """Match the established site-panel axes for the consolidated meter plots."""
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
    """Render one 2x2 feature-engineering grid using the site-figure layout."""
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
            detail = (
                f"PR-AUC {baseline.score:.3f} → {engineered.score:.3f}\n"
                f"Prevalence {prevalence:.1%}"
            )
        ax.plot(baseline.x, baseline.y, color=BASELINE, linewidth=1.15)
        ax.plot(engineered.x, engineered.y, color=ENGINEERED, linewidth=1.45)
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
            1.03,
            detail,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=MUTED,
        )

    metric = "ROC-AUC" if curve_type == "roc" else "PR-AUC"
    fig.suptitle(
        f"Feature Engineering Impact on {metric}",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.930,
        "Each panel: equal-weight Tree Ensemble on the final 50/50 building holdout.",
        ha="left",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.supylabel(
        "True-positive rate" if curve_type == "roc" else "Precision",
        x=0.015,
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.supxlabel(
        "False-positive rate" if curve_type == "roc" else "Recall",
        y=0.10,
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.subplots_adjust(
        left=0.10, right=0.98, top=0.81, bottom=0.16, wspace=0.06, hspace=0.22
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=BASELINE,
                linewidth=1.6,
                label="17-feature Tree Ensemble",
            ),
            Line2D(
                [0],
                [0],
                color=ENGINEERED,
                linewidth=1.8,
                label="137-feature Tree Ensemble",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)


def render_all(
    arrays: dict[str, np.ndarray],
    asset_dir: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}
    summaries: dict[str, Any] = {}
    for meter, slug, label in METER_SPECS:
        data, summary = meter_curve_data(arrays, meter)
        subtitle = (
            f"Tree Ensemble · {label} · final 50/50 building holdout · "
            "17 versus 137 features"
        )
        series = [
            (BASELINE_KEY, "17 baseline features", "#898781"),
            (ENGINEERED_KEY, "137 features", "#2a78d6"),
        ]
        for curve_type, suffix in (
            ("precision_recall", "precision_recall"),
            ("roc", "roc"),
        ):
            output = asset_dir / f"m3_feature_engineering_{slug}_{suffix}.png"
            render_discrimination_curve(
                data,
                output,
                comparison="feature_engineering",
                curve_type=curve_type,
                custom_series=series,
                custom_title_prefix="Feature-Engineering Contribution",
                custom_subtitle=subtitle,
            )
            figures[f"{slug}_{suffix}"] = output
        summaries[slug] = summary
    return figures, summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=PROC / "m3_17_feature_ensemble_predictions_50_50.npz",
    )
    parser.add_argument(
        "--engineered-predictions",
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
        / "m3_feature_engineering_by_meter_roc.png",
    )
    parser.add_argument(
        "--pr-output",
        type=Path,
        default=ROOT
        / "docs"
        / "reports"
        / "assets"
        / "m3"
        / "m3_feature_engineering_by_meter_precision_recall.png",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=ROOT / "docs" / "metrics" / "m3_tree_ensemble_by_meter.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = load_aligned_ensembles(
        args.baseline_predictions,
        args.engineered_predictions,
    )
    results = compute_meter_results(arrays)
    render_grid(results, args.roc_output, curve_type="roc")
    render_grid(results, args.pr_output, curve_type="precision_recall")
    figures = {"roc": args.roc_output, "precision_recall": args.pr_output}
    summaries = {
        result.slug: {
            "rows": result.rows,
            "anomalies": result.anomalies,
            "anomaly_rate": result.anomalies / result.rows,
            "metrics": {
                BASELINE_KEY: {
                    "roc_auc": result.baseline_roc.score,
                    "pr_auc": result.baseline_precision_recall.score,
                },
                ENGINEERED_KEY: {
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
            "experiment": "m3_tree_ensemble_feature_engineering_by_meter",
            "split": "50_50_mod2",
            "feature_comparison": "equal-weight four-model Tree Ensemble, 17 versus 137 features",
            "artifacts": {
                "baseline_predictions": str(
                    args.baseline_predictions.relative_to(ROOT)
                ),
                "engineered_predictions": str(
                    args.engineered_predictions.relative_to(ROOT)
                ),
                "figures": {
                    name: str(path.relative_to(ROOT)) for name, path in figures.items()
                },
            },
            "meters": summaries,
        },
        root=ROOT,
        provenance={"note": "Curves use the existing M3 discrimination renderer."},
    )
    print(f"Saved {len(figures)} figures and {args.metrics_out}")


if __name__ == "__main__":
    main()
