"""Plot one 4-by-5 context grid for each meter type.

Each output keeps one meter and one model/feature regime fixed. Its five
panels run from 5k through 100k training context left to right. Every panel
contains all rows for that meter in the 10,137,155-row holdout, with
normal/anomaly colours and the score-0.5 cutoff. Four meter types times four
regimes therefore produce 16 images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

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
    load_full_feature_columns,
    load_raw_meter_fields,
)


CONTEXT_LABELS = {
    5_000: "5K",
    10_000: "10K",
    20_000: "20K",
    50_000: "50K",
    100_000: "100K",
}
ROW_ORDER = (
    ("tabpfn", 17),
    ("trees", 17),
    ("tabpfn", 137),
    ("trees", 137),
)
ROW_LABELS = {
    ("tabpfn", 17): "17 features - TabPFN",
    ("trees", 17): "17 features - Trees",
    ("tabpfn", 137): "137 features - TabPFN",
    ("trees", 137): "137 features - Trees",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meters",
        type=int,
        nargs="+",
        choices=(0, 1, 2, 3),
        default=[0, 1, 2, 3],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=FIGURE_ROOT,
    )
    return parser.parse_args()


def _plain_reading_axis(axis: plt.Axes, meter_id: int) -> None:
    axis.set_xscale("symlog", linthresh=1.0)
    axis.xaxis.set_minor_formatter(NullFormatter())
    ticks = [0, 1, 10, 100, 1_000, 10_000, 100_000]
    labels = ["0", "1", "10", "100", "1k", "10k", "100k"]
    if meter_id == 2:
        ticks.append(300_000)
        labels.append(">=300k")
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels, fontsize=7)


def _scatter_panel(
    axis: plt.Axes,
    *,
    x: np.ndarray,
    score: np.ndarray,
    anomaly: np.ndarray,
    meter_id: int,
) -> int:
    threshold = 0.5
    normal = ~anomaly
    axis.scatter(
        x[normal],
        score[normal],
        s=1,
        marker=",",
        color=METER_COLORS[meter_id],
        alpha=0.66,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    axis.scatter(
        x[anomaly],
        score[anomaly],
        s=1,
        marker=",",
        color=METER_ANOMALY_COLORS[meter_id],
        alpha=0.92,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )
    axis.axhline(
        threshold,
        color="#d1498b",
        linewidth=0.9,
        linestyle=(0, (4, 3)),
        zorder=4,
    )
    missed = int((anomaly & (score < threshold)).sum())
    false_positives = int((normal & (score >= threshold)).sum())
    axis.text(
        0.02,
        1.03,
        f"missed {missed:,}/{int(anomaly.sum()):,} | false positives {false_positives:,}",
        transform=axis.transAxes,
        color="#9f346a",
        fontsize=6.5,
        va="top",
        ha="left",
        clip_on=False,
        zorder=5,
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
    return missed


def plot_meter_context_row(
    *,
    meter_id: int,
    key: tuple[str, int],
    raw_reading: np.ndarray,
    anomaly: np.ndarray,
    scores: dict[tuple[str, int], list[np.ndarray]],
    output: Path,
) -> None:
    label = METER_NAMES[meter_id]
    row_count = int(raw_reading.size)
    anomaly_count = int(anomaly.sum())
    figure, axes = plt.subplots(
        1,
        5,
        figsize=(19.2, 5.3),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)
    figure.patch.set_facecolor(SURFACE)
    add_title(
        figure,
        title=f"{label} — {ROW_LABELS[key]}",
        subtitle=(
            f"All {row_count:,} {label.lower()} rows in the 10,137,155-row holdout; "
            f"{anomaly_count:,} true anomalies. X = raw meter_reading; Y = predicted "
            f"anomaly score. Dark = normal; light = true anomaly; dashed = cutoff 0.5."
        ),
        title_y=0.985,
        subtitle_y=0.925,
    )
    context_note = (
        "Columns = training-context row count: 5K=5,000; 10K=10,000; "
        "20K=20,000; 50K=50,000; 100K=100,000."
    )
    if meter_id == 2:
        context_note += " Steam readings >300k are placed at the >=300k right edge."
    figure.text(
        0.065,
        0.885,
        context_note,
        color=SECONDARY_INK,
        fontsize=9.5,
        ha="left",
        va="top",
    )

    for column, context in enumerate(CONTEXTS):
        axis = axes[column]
        axis.set_title(
            CONTEXT_LABELS[context],
            fontsize=10.5,
            fontweight="bold",
            color=PRIMARY_INK,
            pad=22,
        )
        _scatter_panel(
            axis,
            x=raw_reading,
            score=scores[key][column],
            anomaly=anomaly,
            meter_id=meter_id,
        )
        _plain_reading_axis(axis, meter_id)

    figure.supylabel(
        "Predicted anomaly score",
        x=0.012,
        fontsize=10.5,
        color=PRIMARY_INK,
    )
    figure.supxlabel(
        "Raw meter_reading (symlog scale; labels use plain values)",
        y=0.105,
        fontsize=10.5,
        color=PRIMARY_INK,
    )
    legend_handles = [
        Patch(
            facecolor=METER_COLORS[meter_id],
            edgecolor="none",
            label=f"{label} normal",
        ),
        Patch(
            facecolor=METER_ANOMALY_COLORS[meter_id],
            edgecolor="none",
            label=f"{label} true anomaly",
        ),
        Line2D(
            [0],
            [0],
            linestyle=(0, (4, 3)),
            linewidth=1.0,
            color="#d1498b",
            label="Score 0.5 cutoff; light points below line are missed anomalies",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.016),
        ncol=3,
        frameon=False,
        fontsize=8.2,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.99,
        top=0.76,
        bottom=0.25,
        wspace=0.08,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    columns = load_full_feature_columns()
    reference_index, reference_y, _, _ = load_prediction("tabpfn", "17", 5_000)
    if not np.array_equal(reference_index, columns["raw_index"]):
        raise AssertionError("feature export and prediction row order differ")
    if not np.array_equal(reference_y, columns["anomaly"]):
        raise AssertionError("feature export and prediction labels differ")

    meter, raw_reading = load_raw_meter_fields(reference_index)
    scores_by_meter: dict[int, dict[tuple[str, int], list[np.ndarray]]] = {
        meter_id: {key: [] for key in ROW_ORDER} for meter_id in args.meters
    }
    masks = {meter_id: meter == meter_id for meter_id in args.meters}

    for context in CONTEXTS:
        print(f"loading context {context:,}", flush=True)
        for model, feature in ROW_ORDER:
            raw_index, anomaly_values, _, score = load_prediction(
                model, str(feature), context
            )
            if not np.array_equal(raw_index, reference_index):
                raise AssertionError(
                    f"row order mismatch for {model}/{feature}/{context}"
                )
            if not np.array_equal(anomaly_values, reference_y):
                raise AssertionError(f"labels mismatch for {model}/{feature}/{context}")
            for meter_id in args.meters:
                scores_by_meter[meter_id][(model, feature)].append(
                    score[masks[meter_id]].copy()
                )
            del score

    for meter_id in args.meters:
        selected = masks[meter_id]
        meter_reading = raw_reading[selected].copy()
        anomaly = reference_y[selected] == 1
        if meter_id == 2:
            overflow = meter_reading > RAW_READING_DISPLAY_CAP
            if int(overflow.sum()) != 3_195 or not anomaly[overflow].all():
                raise AssertionError("unexpected steam overflow composition")
            meter_reading = np.minimum(meter_reading, RAW_READING_DISPLAY_CAP)
        for model, feature in ROW_ORDER:
            output = args.output_root / (
                f"m5_meter_{METER_NAMES[meter_id].lower()}_"
                f"{model}_{feature}features_raw_reading_context_grid.png"
            )
            plot_meter_context_row(
                meter_id=meter_id,
                key=(model, feature),
                raw_reading=meter_reading,
                anomaly=anomaly,
                scores=scores_by_meter[meter_id],
                output=output,
            )
            print(f"wrote {output.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
