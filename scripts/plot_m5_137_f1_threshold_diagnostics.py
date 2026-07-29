"""Apply a per-meter/model/context max-F1 threshold to new M5 diagnostics.

This script writes to a separate output directory. The existing 0.5 diagnostic
figures remain untouched. Thresholds are selected post hoc from the same
holdout scores used for the diagnostic comparison, so the output is an
operating-point sensitivity analysis rather than a deployable estimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_m5_137_anomaly_diagnostics import (
    CONTEXT_LABELS,
    MODEL_COLORS,
    MODEL_LABELS,
    REGIME_LABELS,
    load_anomaly_scores,
    load_normal_scores,
    load_reference,
    plain_reading_axis,
    regime_index,
)
from plot_m5_context_curve_rank_distributions import (
    AXIS,
    CONTEXTS,
    FIGURE_ROOT,
    GRID,
    METER_NAMES,
    PRIMARY_INK,
    SECONDARY_INK,
    SURFACE,
    add_title,
)
from plot_m5_score_covariate_scatter import (
    METER_ANOMALY_COLORS,
    METER_COLORS,
    RAW_READING_DISPLAY_CAP,
)


BASELINE_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meters", type=int, nargs="+", choices=(0, 1, 2, 3), default=[0, 1, 2, 3]
    )
    parser.add_argument(
        "--models", nargs="+", choices=("tabpfn", "trees"), default=["tabpfn", "trees"]
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=FIGURE_ROOT / "m5_137_f1_max_threshold_diagnostics",
    )
    return parser.parse_args()


def max_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return the score threshold and F1 at the maximum discrete PR point."""
    y = np.asarray(y_true, dtype="int8")
    score = np.asarray(scores, dtype="float64")
    positives = int(y.sum())
    if positives == 0 or len(np.unique(y)) < 2:
        return 1.0, 0.0

    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order]
    ends = np.flatnonzero(np.r_[sorted_score[1:] != sorted_score[:-1], True])
    tp = np.cumsum(sorted_y, dtype="int64")[ends]
    predicted = ends + 1
    precision = np.divide(
        tp, predicted, out=np.zeros_like(tp, dtype="float64"), where=predicted != 0
    )
    recall = tp / positives
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype="float64"),
        where=(precision + recall) != 0,
    )
    best = int(np.argmax(f1))
    return float(sorted_score[ends[best]]), float(f1[best])


def binary_counts(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=bool)
    predicted = np.asarray(scores) >= threshold
    tp = int(np.sum(y & predicted))
    fp = int(np.sum(~y & predicted))
    fn = int(np.sum(y & ~predicted))
    tn = int(np.sum(~y & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "anomaly_recall": recall,
        "precision": precision,
        "f1": f1,
    }


def load_scores(
    reference_index: np.ndarray,
    reference_anomaly: np.ndarray,
    meter_mask: np.ndarray,
    model: str,
    context: int,
) -> tuple[np.ndarray, np.ndarray]:
    anomaly = load_anomaly_scores(
        reference_index=reference_index,
        reference_anomaly=reference_anomaly,
        meter_mask=meter_mask,
        model=model,
        context=context,
    )
    normal = load_normal_scores(
        reference_index=reference_index,
        reference_anomaly=reference_anomaly,
        meter_mask=meter_mask,
        model=model,
        context=context,
    )
    return anomaly, normal


def style_axis(axis: plt.Axes, meter_id: int) -> None:
    axis.set_ylim(-0.01, 1.01)
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.set_facecolor(SURFACE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(AXIS)
    axis.spines["bottom"].set_color(AXIS)
    axis.tick_params(colors=SECONDARY_INK, labelsize=7)
    axis.grid(axis="both", color=GRID, linewidth=0.65)
    axis.set_axisbelow(True)
    plain_reading_axis(axis, meter_id)
    if meter_id == 2:
        axis.set_xlim(0.0, 350_000.0)


def plot_classification_grid(
    *,
    meter_id: int,
    model: str,
    reading: np.ndarray,
    scores: list[np.ndarray],
    thresholds: list[float],
    anomaly: bool,
    output: Path,
) -> None:
    meter_name = METER_NAMES[meter_id]
    x = np.minimum(reading, RAW_READING_DISPLAY_CAP) if meter_id == 2 else reading
    figure_width = 23.5 if meter_id == 2 else 19.2
    figure, axes = plt.subplots(
        1, 5, figsize=(figure_width, 5.1), sharex=True, sharey=True
    )
    figure.patch.set_facecolor(SURFACE)
    kind = "anomalies" if anomaly else "normal rows"
    add_title(
        figure,
        title=f"{meter_name} | 137 features | {MODEL_LABELS[model]} | max-F1 threshold",
        subtitle=(
            f"True {kind} only. Each panel uses its own post-hoc threshold selected for maximum F1. "
            "Point colours stay tied to the meter."
        ),
        title_y=0.985,
        subtitle_y=0.925,
    )
    for column, context in enumerate(CONTEXTS):
        axis = axes[column]
        score = scores[column]
        threshold = thresholds[column]
        positive = score >= threshold
        low_alpha = ~positive
        point_color = (
            METER_ANOMALY_COLORS[meter_id] if anomaly else METER_COLORS[meter_id]
        )
        axis.scatter(
            x[low_alpha],
            score[low_alpha],
            s=1.2,
            marker=",",
            color=point_color,
            alpha=0.20,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        axis.scatter(
            x[positive],
            score[positive],
            s=1.2,
            marker=",",
            color=point_color,
            alpha=0.72,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        axis.axhline(
            threshold, color="#d1498b", linewidth=0.9, linestyle=(0, (4, 3)), zorder=3
        )
        if anomaly:
            annotation = f"t={threshold:.3f} | missed {int(low_alpha.sum()):,} | caught {int(positive.sum()):,}"
        else:
            annotation = f"t={threshold:.3f} | excluded {int(low_alpha.sum()):,} | FP {int(positive.sum()):,}"
        axis.text(
            0.02,
            1.03,
            annotation,
            transform=axis.transAxes,
            color="#7b4770",
            fontsize=6.5,
            va="top",
            ha="left",
            clip_on=False,
        )
        axis.set_title(
            CONTEXT_LABELS[context],
            fontsize=10.5,
            fontweight="bold",
            color=PRIMARY_INK,
            pad=22,
        )
        style_axis(axis, meter_id)

    figure.supylabel(
        "Predicted anomaly score", x=0.012, fontsize=10.5, color=PRIMARY_INK
    )
    figure.supxlabel(
        "Raw meter_reading (symlog scale)", y=0.105, fontsize=10.5, color=PRIMARY_INK
    )
    point_color = METER_ANOMALY_COLORS[meter_id] if anomaly else METER_COLORS[meter_id]
    figure.legend(
        handles=[
            Patch(
                facecolor=point_color,
                edgecolor="none",
                alpha=0.20,
                label="Below max-F1 threshold",
            ),
            Patch(
                facecolor=point_color,
                edgecolor="none",
                label="At/above max-F1 threshold",
            ),
            Line2D(
                [0],
                [0],
                color="#d1498b",
                linestyle=(0, (4, 3)),
                linewidth=1.0,
                label="Panel-specific max-F1 threshold",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.016),
        ncol=3,
        frameon=False,
        fontsize=8.2,
    )
    figure.subplots_adjust(left=0.065, right=0.99, top=0.76, bottom=0.25, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(figure)


def plot_rate_comparison(
    frame: pd.DataFrame, output: Path, metric: str, ylabel: str
) -> None:
    meters = sorted(frame["meter_id"].unique())
    models = ["tabpfn", "trees"]
    figure, axes = plt.subplots(
        len(meters), 2, figsize=(12.6, 3.2 * len(meters)), sharey=True
    )
    axes = np.atleast_2d(axes)
    figure.patch.set_facecolor(SURFACE)
    title = (
        "False-positive rate" if metric == "false_positive_rate" else "Anomaly recall"
    )
    y_columns = [f"baseline_{metric}", f"f1_{metric}"]
    y_max = float(frame[y_columns].to_numpy(dtype="float64").max())
    add_title(
        figure,
        title=f"137-feature {title}: fixed 0.5 versus max-F1 threshold",
        subtitle="Each marker is one context. The max-F1 operating point is calibrated separately per meter, model, and context.",
        title_y=0.985,
        subtitle_y=0.94,
    )
    for row, meter_id in enumerate(meters):
        for column, model in enumerate(models):
            axis = axes[row, column]
            subset = frame[
                (frame.meter_id == meter_id) & (frame.model == model)
            ].sort_values("context_rows")
            axis.plot(
                subset["context_rows"],
                subset[f"baseline_{metric}"],
                color="#8a8a8a",
                marker="o",
                linewidth=1.2,
                label="0.5",
            )
            axis.plot(
                subset["context_rows"],
                subset[f"f1_{metric}"],
                color=MODEL_COLORS[model],
                marker="o",
                linewidth=1.35,
                label="max F1",
            )
            axis.set_xscale("symlog", linthresh=1000)
            axis.set_xticks(list(CONTEXTS))
            axis.set_xticklabels([CONTEXT_LABELS[c] for c in CONTEXTS], fontsize=7)
            axis.set_ylim(0, y_max * 1.12 if y_max else 1.0)
            axis.set_title(
                f"{METER_NAMES[meter_id]} | {MODEL_LABELS[model]}",
                loc="left",
                fontsize=11,
                fontweight="bold",
                color=PRIMARY_INK,
            )
            axis.tick_params(colors=SECONDARY_INK, labelsize=7)
            axis.grid(axis="y", color=GRID, linewidth=0.65)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color(AXIS)
            axis.spines["bottom"].set_color(AXIS)
            if row == 0 and column == 0:
                axis.legend(frameon=False, fontsize=8)
    figure.supylabel(ylabel, x=0.02, fontsize=10.5, color=PRIMARY_INK)
    figure.supxlabel("Training context rows", y=0.04, fontsize=10.5, color=PRIMARY_INK)
    figure.subplots_adjust(
        left=0.08, right=0.985, top=0.87, bottom=0.10, wspace=0.16, hspace=0.34
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    reference_index, reference_anomaly, meter, reading = load_reference()
    comparison: list[dict[str, object]] = []
    anomaly_regime_rows: list[dict[str, object]] = []
    normal_regime_rows: list[dict[str, object]] = []

    for meter_id in args.meters:
        meter_mask = meter == meter_id
        anomaly_mask = meter_mask & (reference_anomaly == 1)
        normal_mask = meter_mask & (reference_anomaly == 0)
        anomaly_reading = reading[anomaly_mask].copy()
        normal_reading = reading[normal_mask].copy()
        anomaly_regimes = regime_index(anomaly_reading)
        normal_regimes = regime_index(normal_reading)
        for model in args.models:
            anomaly_scores_by_context: list[np.ndarray] = []
            normal_scores_by_context: list[np.ndarray] = []
            thresholds: list[float] = []
            for context in CONTEXTS:
                anomaly_scores, normal_scores = load_scores(
                    reference_index, reference_anomaly, meter_mask, model, context
                )
                anomaly_scores_by_context.append(anomaly_scores)
                normal_scores_by_context.append(normal_scores)
                scores = np.concatenate([anomaly_scores, normal_scores])
                labels = np.concatenate(
                    [
                        np.ones(len(anomaly_scores), dtype="int8"),
                        np.zeros(len(normal_scores), dtype="int8"),
                    ]
                )
                threshold, holdout_f1 = max_f1_threshold(labels, scores)
                thresholds.append(threshold)
                baseline = binary_counts(labels, scores, BASELINE_THRESHOLD)
                f1_point = binary_counts(labels, scores, threshold)
                comparison.append(
                    {
                        "meter_id": meter_id,
                        "meter": METER_NAMES[meter_id],
                        "model": model,
                        "feature_count": 137,
                        "context_rows": context,
                        "threshold": threshold,
                        "holdout_max_f1": holdout_f1,
                        **{f"baseline_{key}": value for key, value in baseline.items()},
                        **{f"f1_{key}": value for key, value in f1_point.items()},
                        "delta_fp": f1_point["fp"] - baseline["fp"],
                        "delta_false_positive_rate": f1_point["false_positive_rate"]
                        - baseline["false_positive_rate"],
                        "delta_fn": f1_point["fn"] - baseline["fn"],
                        "delta_anomaly_recall": f1_point["anomaly_recall"]
                        - baseline["anomaly_recall"],
                    }
                )
                for regime_id in range(len(REGIME_LABELS)):
                    selected = anomaly_regimes == regime_id
                    selected_normal = normal_regimes == regime_id
                    anomaly_regime_rows.append(
                        {
                            "meter_id": meter_id,
                            "meter": METER_NAMES[meter_id],
                            "model": model,
                            "context_rows": context,
                            "regime_id": regime_id,
                            "regime": REGIME_LABELS[regime_id],
                            "anomaly_count": int(selected.sum()),
                            "baseline_detected_count": int(
                                (anomaly_scores[selected] >= BASELINE_THRESHOLD).sum()
                            )
                            if selected.any()
                            else 0,
                            "f1_detected_count": int(
                                (anomaly_scores[selected] >= threshold).sum()
                            )
                            if selected.any()
                            else 0,
                            "threshold": threshold,
                        }
                    )
                    normal_regime_rows.append(
                        {
                            "meter_id": meter_id,
                            "meter": METER_NAMES[meter_id],
                            "model": model,
                            "context_rows": context,
                            "regime_id": regime_id,
                            "regime": REGIME_LABELS[regime_id],
                            "normal_count": int(selected_normal.sum()),
                            "baseline_false_positive_count": int(
                                (
                                    normal_scores[selected_normal] >= BASELINE_THRESHOLD
                                ).sum()
                            )
                            if selected_normal.any()
                            else 0,
                            "f1_false_positive_count": int(
                                (normal_scores[selected_normal] >= threshold).sum()
                            )
                            if selected_normal.any()
                            else 0,
                            "threshold": threshold,
                        }
                    )

            plot_classification_grid(
                meter_id=meter_id,
                model=model,
                reading=anomaly_reading,
                scores=anomaly_scores_by_context,
                thresholds=thresholds,
                anomaly=True,
                output=output_root
                / f"m5_137_anomaly_only_{METER_NAMES[meter_id].lower()}_{model}_f1_max_threshold.png",
            )
            plot_classification_grid(
                meter_id=meter_id,
                model=model,
                reading=normal_reading,
                scores=normal_scores_by_context,
                thresholds=thresholds,
                anomaly=False,
                output=output_root
                / f"m5_137_normal_only_{METER_NAMES[meter_id].lower()}_{model}_f1_max_threshold.png",
            )
            print(f"wrote {meter_id}/{model}", flush=True)

    comparison_frame = pd.DataFrame(comparison)
    comparison_frame.to_csv(output_root / "m5_137_f1_threshold_vs_0_5.csv", index=False)
    anomaly_frame = pd.DataFrame(anomaly_regime_rows)
    anomaly_frame.to_csv(
        output_root / "m5_137_anomaly_regime_f1_vs_0_5.csv", index=False
    )
    normal_frame = pd.DataFrame(normal_regime_rows)
    normal_frame.to_csv(output_root / "m5_137_normal_regime_f1_vs_0_5.csv", index=False)
    plot_rate_comparison(
        comparison_frame,
        output_root / "m5_137_false_positive_rate_0_5_vs_f1_max.png",
        "false_positive_rate",
        "False-positive rate among true normal rows",
    )
    plot_rate_comparison(
        comparison_frame,
        output_root / "m5_137_anomaly_recall_0_5_vs_f1_max.png",
        "anomaly_recall",
        "Recall among true anomalies",
    )
    print(f"wrote {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
