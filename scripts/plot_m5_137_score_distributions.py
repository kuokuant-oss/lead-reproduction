"""Plot 137-feature score geometry across meters, models, and contexts.

The script creates a separate score-distribution output directory. It does not
modify the existing raw-reading scatter plots or threshold diagnostics.
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
CONTEXT_COLORS = {
    5_000: "#2878b5",
    10_000: "#5aa469",
    20_000: "#d99a2b",
    50_000: "#b45f9f",
    100_000: "#d1495b",
}
HIST_BINS = np.linspace(0.0, 1.0, 61)
QUANTILES = np.linspace(0.0, 1.0, 401)
SUMMARY_QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
VIOLIN_SAMPLE = 4_000


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
        default=FIGURE_ROOT / "m5_137_score_distributions",
    )
    return parser.parse_args()


def load_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_index, anomaly, _, _ = load_prediction("tabpfn", "17", 5_000)
    meter, _ = load_raw_meter_fields(raw_index)
    return raw_index, anomaly, meter


def load_context_scores(
    *,
    reference_index: np.ndarray,
    reference_anomaly: np.ndarray,
    meter: np.ndarray,
    model: str,
    context: int,
) -> dict[int, dict[str, np.ndarray]]:
    raw_index, anomaly, _, score = load_prediction(model, "137", context)
    if not np.array_equal(raw_index, reference_index):
        raise AssertionError(f"row order mismatch for {model}/137/{context}")
    if not np.array_equal(anomaly, reference_anomaly):
        raise AssertionError(f"label mismatch for {model}/137/{context}")
    return {
        meter_id: {
            "normal": np.asarray(
                score[(meter == meter_id) & (anomaly == 0)], dtype="float32"
            ),
            "anomaly": np.asarray(
                score[(meter == meter_id) & (anomaly == 1)], dtype="float32"
            ),
        }
        for meter_id in sorted(METER_NAMES)
    }


def histogram_mass(values: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(np.clip(values, 0.0, 1.0), bins=HIST_BINS)
    return counts.astype("float64") / len(values)


def configure_axis(axis: plt.Axes, *, title: str, xlabel: bool = True) -> None:
    axis.set_title(
        title, loc="left", fontsize=11.5, fontweight="bold", color=PRIMARY_INK
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_xticks(np.linspace(0.0, 1.0, 6))
    axis.set_xticklabels(["0", ".2", ".4", ".6", ".8", "1"], fontsize=7)
    axis.tick_params(colors=SECONDARY_INK, labelsize=7)
    axis.grid(axis="both", color=GRID, linewidth=0.65)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(AXIS)
    axis.spines["bottom"].set_color(AXIS)
    if not xlabel:
        axis.set_xlabel("")


def plot_histograms(
    *,
    distributions: dict[tuple[str, int, int, str], np.ndarray],
    meters: list[int],
    models: list[str],
    label: str,
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        len(models),
        len(meters),
        figsize=(4.25 * len(meters), 3.25 * len(models)),
        sharex=True,
    )
    axes = np.atleast_2d(axes)
    figure.patch.set_facecolor(SURFACE)
    add_title(
        figure,
        title=f"137-feature {label} score distributions",
        subtitle="Each coloured line is a training context; y-axis is probability mass per score bin.",
        title_y=0.985,
        subtitle_y=0.93,
    )
    for row, model in enumerate(models):
        for column, meter_id in enumerate(meters):
            axis = axes[row, column]
            for context in CONTEXTS:
                mass = histogram_mass(
                    distributions[(model, context, meter_id, label.lower())]
                )
                axis.step(
                    HIST_BINS[:-1],
                    np.maximum(mass, 1e-6),
                    where="post",
                    color=CONTEXT_COLORS[context],
                    linewidth=1.2,
                    alpha=0.45 + 0.11 * CONTEXTS.index(context),
                )
            configure_axis(
                axis, title=f"{METER_NAMES[meter_id]} | {MODEL_LABELS[model]}"
            )
            axis.set_yscale("log")
            axis.set_ylim(1e-5, 1.0)
            if row == len(models) - 1:
                axis.set_xlabel("Predicted anomaly score")
    figure.supylabel(
        f"{label} probability mass", x=0.02, fontsize=10.5, color=PRIMARY_INK
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=CONTEXT_COLORS[c],
                linewidth=1.5,
                label=CONTEXT_LABELS[c],
            )
            for c in CONTEXTS
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        fontsize=8.2,
    )
    figure.subplots_adjust(
        left=0.065, right=0.99, top=0.84, bottom=0.14, wspace=0.18, hspace=0.30
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(figure)


def plot_cdf(
    *,
    distributions: dict[tuple[str, int, int, str], np.ndarray],
    meters: list[int],
    models: list[str],
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        len(models),
        len(meters),
        figsize=(4.25 * len(meters), 3.25 * len(models)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)
    figure.patch.set_facecolor(SURFACE)
    add_title(
        figure,
        title="137-feature normal/anomaly score CDF",
        subtitle="Solid lines are anomalies; dashed lines are normal rows. Colours identify context size.",
        title_y=0.985,
        subtitle_y=0.93,
    )
    for row, model in enumerate(models):
        for column, meter_id in enumerate(meters):
            axis = axes[row, column]
            for context in CONTEXTS:
                for label, linestyle in (("normal", "--"), ("anomaly", "-")):
                    values = distributions[(model, context, meter_id, label)]
                    axis.plot(
                        np.quantile(values, QUANTILES),
                        QUANTILES,
                        color=CONTEXT_COLORS[context],
                        linestyle=linestyle,
                        linewidth=1.1,
                        alpha=0.45 + 0.11 * CONTEXTS.index(context),
                    )
            axis.axvline(0.5, color="#888888", linewidth=0.8, linestyle=(0, (2, 3)))
            configure_axis(
                axis, title=f"{METER_NAMES[meter_id]} | {MODEL_LABELS[model]}"
            )
            axis.set_ylim(0.0, 1.0)
            axis.set_yticks(np.linspace(0.0, 1.0, 6))
            axis.set_yticklabels(["0", ".2", ".4", ".6", ".8", "1"], fontsize=7)
            if row == len(models) - 1:
                axis.set_xlabel("Predicted anomaly score")
    figure.supylabel(
        "Cumulative fraction of rows", x=0.02, fontsize=10.5, color=PRIMARY_INK
    )
    figure.legend(
        handles=[
            Line2D(
                [0], [0], color="#555555", linestyle="-", linewidth=1.4, label="Anomaly"
            ),
            Line2D(
                [0], [0], color="#555555", linestyle="--", linewidth=1.4, label="Normal"
            ),
            *[
                Line2D(
                    [0],
                    [0],
                    color=CONTEXT_COLORS[c],
                    linewidth=1.5,
                    label=CONTEXT_LABELS[c],
                )
                for c in CONTEXTS
            ],
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=7,
        frameon=False,
        fontsize=8.0,
    )
    figure.subplots_adjust(
        left=0.065, right=0.99, top=0.84, bottom=0.14, wspace=0.18, hspace=0.30
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(figure)


def plot_violins(
    *,
    distributions: dict[tuple[str, int, int, str], np.ndarray],
    meters: list[int],
    models: list[str],
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        len(models),
        len(meters),
        figsize=(4.25 * len(meters), 3.25 * len(models)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)
    figure.patch.set_facecolor(SURFACE)
    add_title(
        figure,
        title="137-feature score geometry by context",
        subtitle="Each context has a normal violin and an anomaly violin; samples are capped for rendering.",
        title_y=0.985,
        subtitle_y=0.93,
    )
    for row, model in enumerate(models):
        for column, meter_id in enumerate(meters):
            axis = axes[row, column]
            positions = np.arange(len(CONTEXTS), dtype="float64")
            for offset, label, color in (
                (-0.16, "normal", METER_COLORS[meter_id]),
                (0.16, "anomaly", METER_ANOMALY_COLORS[meter_id]),
            ):
                data = [
                    distributions[(model, context, meter_id, label)]
                    for context in CONTEXTS
                ]
                parts = axis.violinplot(
                    data,
                    positions=positions + offset,
                    widths=0.28,
                    showmeans=False,
                    showmedians=True,
                    showextrema=False,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.42 if label == "normal" else 0.62)
                parts["cmedians"].set_color(color)
                parts["cmedians"].set_linewidth(0.8)
            axis.axhline(0.5, color="#888888", linewidth=0.8, linestyle=(0, (2, 3)))
            axis.set_title(
                f"{METER_NAMES[meter_id]} | {MODEL_LABELS[model]}",
                loc="left",
                fontsize=11.5,
                fontweight="bold",
                color=PRIMARY_INK,
            )
            axis.set_xticks(positions)
            axis.set_xticklabels([CONTEXT_LABELS[c] for c in CONTEXTS], fontsize=7)
            axis.set_xlim(-0.55, len(CONTEXTS) - 0.45)
            axis.set_ylim(0.0, 1.0)
            axis.set_yticks(np.linspace(0.0, 1.0, 6))
            axis.tick_params(colors=SECONDARY_INK, labelsize=7)
            axis.grid(axis="y", color=GRID, linewidth=0.65)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color(AXIS)
            axis.spines["bottom"].set_color(AXIS)
    figure.supylabel(
        "Predicted anomaly score", x=0.02, fontsize=10.5, color=PRIMARY_INK
    )
    figure.supxlabel("Training context rows", y=0.04, fontsize=10.5, color=PRIMARY_INK)
    figure.legend(
        handles=[
            Line2D([0], [0], color="#555555", linewidth=5, alpha=0.42, label="Normal"),
            Line2D([0], [0], color="#555555", linewidth=5, alpha=0.62, label="Anomaly"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=8.2,
    )
    figure.subplots_adjust(
        left=0.065, right=0.99, top=0.84, bottom=0.14, wspace=0.18, hspace=0.30
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    reference_index, reference_anomaly, meter = load_reference()
    rng = np.random.default_rng(42)
    distributions: dict[tuple[str, int, int, str], np.ndarray] = {}
    summary_rows: list[dict[str, object]] = []

    for model in args.models:
        for context in CONTEXTS:
            scores_by_meter = load_context_scores(
                reference_index=reference_index,
                reference_anomaly=reference_anomaly,
                meter=meter,
                model=model,
                context=context,
            )
            for meter_id in args.meters:
                for label in ("normal", "anomaly"):
                    values = scores_by_meter[meter_id][label]
                    if len(values) > VIOLIN_SAMPLE:
                        sample = rng.choice(values, size=VIOLIN_SAMPLE, replace=False)
                    else:
                        sample = values
                    distributions[(model, context, meter_id, label)] = sample
                    quantiles = np.quantile(values, SUMMARY_QUANTILES)
                    summary_rows.append(
                        {
                            "model": model,
                            "meter_id": meter_id,
                            "meter": METER_NAMES[meter_id],
                            "context_rows": context,
                            "class": label,
                            "rows": len(values),
                            **{
                                f"q{int(q * 100):02d}": float(value)
                                for q, value in zip(
                                    SUMMARY_QUANTILES, quantiles, strict=True
                                )
                            },
                            "below_0_5_rate": float((values < 0.5).mean()),
                            "above_0_5_rate": float((values >= 0.5).mean()),
                        }
                    )
            print(f"loaded {model}/{context}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "m5_137_score_distribution_quantiles.csv", index=False)
    plot_histograms(
        distributions=distributions,
        meters=args.meters,
        models=args.models,
        label="Anomaly",
        output=output_root / "m5_137_anomaly_score_histograms.png",
    )
    plot_histograms(
        distributions=distributions,
        meters=args.meters,
        models=args.models,
        label="Normal",
        output=output_root / "m5_137_normal_score_histograms.png",
    )
    plot_cdf(
        distributions=distributions,
        meters=args.meters,
        models=args.models,
        output=output_root / "m5_137_normal_anomaly_score_cdf.png",
    )
    plot_violins(
        distributions=distributions,
        meters=args.meters,
        models=args.models,
        output=output_root / "m5_137_score_violin_by_meter_context.png",
    )
    print(f"wrote {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
