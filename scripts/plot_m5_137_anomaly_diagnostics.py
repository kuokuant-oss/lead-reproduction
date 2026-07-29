"""Create new 137-feature anomaly-focused diagnostics without touching canonical plots.

The existing M5 raw-reading context grids remain unchanged. This script creates:

* anomaly-only context grids for selected meters and learners; and
* a raw-reading-regime detection-rate figure showing the fixed 0.5 operating rule.

The second figure is designed to expose shared blind regions such as hotwater
near-zero anomalies and steam high-reading anomalies.
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
from matplotlib.ticker import NullFormatter

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
    load_prediction,
)
from plot_m5_score_covariate_scatter import (
    METER_ANOMALY_COLORS,
    METER_COLORS,
    RAW_READING_DISPLAY_CAP,
    load_raw_meter_fields,
)


CONTEXT_LABELS = {
    5_000: "5K",
    10_000: "10K",
    20_000: "20K",
    50_000: "50K",
    100_000: "100K",
}
MODEL_LABELS = {"tabpfn": "TabPFN", "trees": "Tree Ensemble"}
MODEL_COLORS = {"tabpfn": "#d1498b", "trees": "#0b0b0b"}
THRESHOLD = 0.5
REGIME_EDGES = np.array(
    [0, 1, 10, 100, 1_000, 10_000, 100_000, 300_000], dtype="float64"
)
REGIME_LABELS = [
    "0–1",
    "1–10",
    "10–100",
    "100–1k",
    "1k–10k",
    "10k–100k",
    "100k–300k",
    "≥300k",
]


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
        default=FIGURE_ROOT / "m5_137_anomaly_diagnostics",
    )
    return parser.parse_args()


def plain_reading_axis(axis: plt.Axes, meter_id: int) -> None:
    axis.set_xscale("symlog", linthresh=1.0)
    axis.xaxis.set_minor_formatter(NullFormatter())
    ticks = [0, 1, 10, 100, 1_000, 10_000, 100_000]
    labels = ["0", "1", "10", "100", "1k", "10k", "100k"]
    if meter_id == 2:
        ticks.append(300_000)
        labels.append("300k")
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels, fontsize=7)


def load_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_index, anomaly, _, _ = load_prediction("tabpfn", "17", 5_000)
    meter, reading = load_raw_meter_fields(raw_index)
    return raw_index, anomaly, meter, reading


def load_anomaly_scores(
    *,
    reference_index: np.ndarray,
    reference_anomaly: np.ndarray,
    meter_mask: np.ndarray,
    model: str,
    context: int,
) -> np.ndarray:
    raw_index, anomaly, _, score = load_prediction(model, "137", context)
    if not np.array_equal(raw_index, reference_index):
        raise AssertionError(f"row order mismatch for {model}/137/{context}")
    if not np.array_equal(anomaly, reference_anomaly):
        raise AssertionError(f"label mismatch for {model}/137/{context}")
    return np.asarray(score[meter_mask & (reference_anomaly == 1)], dtype="float32")


def load_normal_scores(
    *,
    reference_index: np.ndarray,
    reference_anomaly: np.ndarray,
    meter_mask: np.ndarray,
    model: str,
    context: int,
) -> np.ndarray:
    raw_index, anomaly, _, score = load_prediction(model, "137", context)
    if not np.array_equal(raw_index, reference_index):
        raise AssertionError(f"row order mismatch for {model}/137/{context}")
    if not np.array_equal(anomaly, reference_anomaly):
        raise AssertionError(f"label mismatch for {model}/137/{context}")
    return np.asarray(score[meter_mask & (reference_anomaly == 0)], dtype="float32")


def plot_anomaly_grid(
    *,
    meter_id: int,
    model: str,
    reading: np.ndarray,
    scores: list[np.ndarray],
    output: Path,
) -> None:
    meter_name = METER_NAMES[meter_id]
    x = np.minimum(reading, RAW_READING_DISPLAY_CAP) if meter_id == 2 else reading
    figure_width = 23.5 if meter_id == 2 else 19.2
    figure, axes = plt.subplots(
        1, 5, figsize=(figure_width, 5.1), sharex=True, sharey=True
    )
    figure.patch.set_facecolor(SURFACE)
    add_title(
        figure,
        title=f"{meter_name} — 137 features — {MODEL_LABELS[model]}",
        subtitle=(
            "True anomalies only across the complete holdout. "
            "Light points are fixed-threshold misses; saturated points are detected at score ≥ 0.5."
        ),
        title_y=0.985,
        subtitle_y=0.925,
    )
    for column, context in enumerate(CONTEXTS):
        axis = axes[column]
        score = scores[column]
        missed = score < THRESHOLD
        detected = ~missed
        axis.scatter(
            x[missed],
            score[missed],
            s=1.2,
            marker=",",
            color=METER_ANOMALY_COLORS[meter_id],
            alpha=0.20,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        axis.scatter(
            x[detected],
            score[detected],
            s=1.2,
            marker=",",
            color=METER_ANOMALY_COLORS[meter_id],
            alpha=0.62,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        axis.axhline(
            THRESHOLD, color="#d1498b", linewidth=0.9, linestyle=(0, (4, 3)), zorder=3
        )
        axis.text(
            0.02,
            1.03,
            f"missed {int(missed.sum()):,} | caught {int(detected.sum()):,}",
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

    figure.supylabel(
        "Predicted anomaly score", x=0.012, fontsize=10.5, color=PRIMARY_INK
    )
    figure.supxlabel(
        "Raw meter_reading (symlog scale)", y=0.105, fontsize=10.5, color=PRIMARY_INK
    )
    figure.legend(
        handles=[
            Patch(
                facecolor=METER_ANOMALY_COLORS[meter_id],
                edgecolor="none",
                alpha=0.20,
                label="True anomaly missed at 0.5",
            ),
            Patch(
                facecolor=METER_ANOMALY_COLORS[meter_id],
                edgecolor="none",
                label="True anomaly detected at 0.5",
            ),
            Line2D(
                [0],
                [0],
                color="#d1498b",
                linestyle=(0, (4, 3)),
                linewidth=1.0,
                label="Score 0.5",
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


def plot_normal_grid(
    *,
    meter_id: int,
    model: str,
    reading: np.ndarray,
    scores: list[np.ndarray],
    output: Path,
) -> None:
    meter_name = METER_NAMES[meter_id]
    x = np.minimum(reading, RAW_READING_DISPLAY_CAP) if meter_id == 2 else reading
    figure_width = 23.5 if meter_id == 2 else 19.2
    figure, axes = plt.subplots(
        1, 5, figsize=(figure_width, 5.1), sharex=True, sharey=True
    )
    figure.patch.set_facecolor(SURFACE)
    add_title(
        figure,
        title=f"{meter_name} — 137 features — {MODEL_LABELS[model]}",
        subtitle=(
            "True normal rows only across the complete holdout. "
            "Light points are correctly excluded; saturated points are false positives at score ≥ 0.5."
        ),
        title_y=0.985,
        subtitle_y=0.925,
    )
    for column, context in enumerate(CONTEXTS):
        axis = axes[column]
        score = scores[column]
        false_positive = score >= THRESHOLD
        excluded = ~false_positive
        axis.scatter(
            x[excluded],
            score[excluded],
            s=1.2,
            marker=",",
            color=METER_COLORS[meter_id],
            alpha=0.20,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        axis.scatter(
            x[false_positive],
            score[false_positive],
            s=1.2,
            marker=",",
            color=METER_COLORS[meter_id],
            alpha=0.72,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        axis.axhline(
            THRESHOLD, color="#d1498b", linewidth=0.9, linestyle=(0, (4, 3)), zorder=3
        )
        axis.text(
            0.02,
            1.03,
            f"excluded {int(excluded.sum()):,} | false positives {int(false_positive.sum()):,}",
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

    figure.supylabel(
        "Predicted anomaly score", x=0.012, fontsize=10.5, color=PRIMARY_INK
    )
    figure.supxlabel(
        "Raw meter_reading (symlog scale)", y=0.105, fontsize=10.5, color=PRIMARY_INK
    )
    figure.legend(
        handles=[
            Patch(
                facecolor=METER_COLORS[meter_id],
                edgecolor="none",
                alpha=0.20,
                label="Normal excluded at 0.5",
            ),
            Patch(
                facecolor=METER_COLORS[meter_id],
                edgecolor="none",
                label="Normal false positive at 0.5",
            ),
            Line2D(
                [0],
                [0],
                color="#d1498b",
                linestyle=(0, (4, 3)),
                linewidth=1.0,
                label="Score 0.5",
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


def regime_index(reading: np.ndarray) -> np.ndarray:
    clipped = np.minimum(np.maximum(reading, 0.0), REGIME_EDGES[-1])
    return np.digitize(clipped, REGIME_EDGES, right=False) - 1


def plot_regime_rate(
    *,
    results: pd.DataFrame,
    output: Path,
    title: str,
    subtitle: str,
    y_label: str,
) -> None:
    meters = sorted(results["meter_id"].unique())
    models = ["tabpfn", "trees"]
    figure, axes = plt.subplots(
        len(meters), 2, figsize=(13.0, 3.65 * len(meters)), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)
    figure.patch.set_facecolor(SURFACE)
    add_title(
        figure,
        title=title,
        subtitle=subtitle,
        title_y=0.98,
        subtitle_y=0.935,
    )
    for row, meter_id in enumerate(meters):
        for column, model in enumerate(models):
            axis = axes[row, column]
            subset = results[
                (results["meter_id"] == meter_id) & (results["model"] == model)
            ]
            for context in CONTEXTS:
                line = subset[subset["context_rows"] == context].sort_values(
                    "regime_id"
                )
                axis.plot(
                    line["regime_id"],
                    line["rate"],
                    marker="o",
                    markersize=3.2,
                    linewidth=1.15,
                    color=MODEL_COLORS[model],
                    alpha=0.35 + 0.12 * CONTEXTS.index(context),
                    label=CONTEXT_LABELS[context],
                )
            axis.set_title(
                f"{METER_NAMES[meter_id]} — {MODEL_LABELS[model]}",
                loc="left",
                fontsize=11.5,
                fontweight="bold",
                color=PRIMARY_INK,
            )
            axis.set_ylim(-0.02, 1.02)
            axis.set_yticks(np.linspace(0, 1, 6))
            axis.set_xticks(range(len(REGIME_LABELS)))
            axis.set_xticklabels(REGIME_LABELS, rotation=35, ha="right", fontsize=7)
            axis.tick_params(colors=SECONDARY_INK, labelsize=7.5)
            axis.grid(axis="y", color=GRID, linewidth=0.65)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color(AXIS)
            axis.spines["bottom"].set_color(AXIS)
    figure.supylabel(y_label, x=0.02, fontsize=10.5, color=PRIMARY_INK)
    figure.supxlabel(
        "Raw meter_reading regime", y=0.06, fontsize=10.5, color=PRIMARY_INK
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#5f5f5f",
            linewidth=1.25,
            alpha=0.35 + 0.12 * index,
            marker="o",
            markersize=3.2,
            label=CONTEXT_LABELS[context],
        )
        for index, context in enumerate(CONTEXTS)
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=5,
        frameon=False,
        fontsize=8.2,
    )
    figure.subplots_adjust(
        left=0.09, right=0.985, top=0.88, bottom=0.12, wspace=0.16, hspace=0.30
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    reference_index, reference_anomaly, meter, reading = load_reference()
    results: list[dict[str, object]] = []
    normal_results: list[dict[str, object]] = []

    for meter_id in args.meters:
        meter_mask = meter == meter_id
        anomaly_reading = reading[meter_mask & (reference_anomaly == 1)].copy()
        regime_ids = regime_index(anomaly_reading)
        normal_reading = reading[meter_mask & (reference_anomaly == 0)].copy()
        normal_regime_ids = regime_index(normal_reading)
        for model in args.models:
            score_by_context: list[np.ndarray] = []
            normal_score_by_context: list[np.ndarray] = []
            for context in CONTEXTS:
                score = load_anomaly_scores(
                    reference_index=reference_index,
                    reference_anomaly=reference_anomaly,
                    meter_mask=meter_mask,
                    model=model,
                    context=context,
                )
                score_by_context.append(score)
                normal_score = load_normal_scores(
                    reference_index=reference_index,
                    reference_anomaly=reference_anomaly,
                    meter_mask=meter_mask,
                    model=model,
                    context=context,
                )
                normal_score_by_context.append(normal_score)
                for regime_id in range(len(REGIME_LABELS)):
                    selected = regime_ids == regime_id
                    count = int(selected.sum())
                    results.append(
                        {
                            "meter_id": meter_id,
                            "meter": METER_NAMES[meter_id],
                            "model": model,
                            "feature_count": 137,
                            "context_rows": context,
                            "regime_id": regime_id,
                            "regime": REGIME_LABELS[regime_id],
                            "anomaly_count": count,
                            "detected_count": int((score[selected] >= THRESHOLD).sum())
                            if count
                            else 0,
                            "detection_rate": float(
                                (score[selected] >= THRESHOLD).mean()
                            )
                            if count
                            else np.nan,
                        }
                    )
                for regime_id in range(len(REGIME_LABELS)):
                    selected = normal_regime_ids == regime_id
                    count = int(selected.sum())
                    normal_results.append(
                        {
                            "meter_id": meter_id,
                            "meter": METER_NAMES[meter_id],
                            "model": model,
                            "feature_count": 137,
                            "context_rows": context,
                            "regime_id": regime_id,
                            "regime": REGIME_LABELS[regime_id],
                            "normal_count": count,
                            "false_positive_count": int(
                                (normal_score[selected] >= THRESHOLD).sum()
                            )
                            if count
                            else 0,
                            "rate": float((normal_score[selected] >= THRESHOLD).mean())
                            if count
                            else np.nan,
                        }
                    )
            output = (
                output_root
                / f"m5_137_anomaly_only_{METER_NAMES[meter_id].lower()}_{model}_raw_reading_context_grid.png"
            )
            plot_anomaly_grid(
                meter_id=meter_id,
                model=model,
                reading=anomaly_reading,
                scores=score_by_context,
                output=output,
            )
            print(f"wrote {output}", flush=True)
            normal_output = output_root / (
                f"m5_137_normal_only_{METER_NAMES[meter_id].lower()}_{model}_raw_reading_context_grid.png"
            )
            plot_normal_grid(
                meter_id=meter_id,
                model=model,
                reading=normal_reading,
                scores=normal_score_by_context,
                output=normal_output,
            )
            print(f"wrote {normal_output}", flush=True)

    result_frame = pd.DataFrame(results)
    csv_path = output_root / "m5_137_anomaly_detection_by_raw_regime.csv"
    result_frame.to_csv(csv_path, index=False)
    regime_path = output_root / "m5_137_anomaly_detection_by_raw_regime.png"
    plot_regime_rate(
        results=result_frame.rename(columns={"detection_rate": "rate"}),
        output=regime_path,
        title="137-feature anomaly detection by raw-reading regime",
        subtitle=(
            "True anomalies only. Each line is the fraction with score ≥ 0.5; "
            "the fixed threshold is evaluated separately within each raw-reading band."
        ),
        y_label="Anomaly detection rate at score ≥ 0.5",
    )
    normal_frame = pd.DataFrame(normal_results)
    normal_csv_path = output_root / "m5_137_normal_false_positive_by_raw_regime.csv"
    normal_frame.to_csv(normal_csv_path, index=False)
    normal_regime_path = output_root / "m5_137_normal_false_positive_by_raw_regime.png"
    plot_regime_rate(
        results=normal_frame,
        output=normal_regime_path,
        title="137-feature normal false positives by raw-reading regime",
        subtitle=(
            "True normal rows only. Each line is the fraction with score ≥ 0.5; "
            "the fixed threshold is evaluated separately within each raw-reading band."
        ),
        y_label="Normal false-positive rate at score ≥ 0.5",
    )
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {regime_path}", flush=True)
    print(f"wrote {normal_csv_path}", flush=True)
    print(f"wrote {normal_regime_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
