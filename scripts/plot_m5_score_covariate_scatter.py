"""Plot model anomaly score against explanatory non-model covariates.

Each point is one row in the complete 10,137,155-row holdout. Y is the model's
predicted anomaly probability, with the repository's raw operating threshold
at 0.5. X comes from source data or the frozen feature matrix.
"""

from __future__ import annotations

import argparse
import json
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
    FIGURE_ROOT,
    GRID,
    METER_NAMES,
    MODEL_STYLE,
    MODELS,
    PRIMARY_INK,
    SECONDARY_INK,
    SURFACE,
    add_title,
    load_prediction,
)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EXPORT_ROOT = PROC / "m5_tabpfn_137_distributed_context100000_n8"
CONTEXTS = (10_000, 20_000, 50_000, 100_000)
CONTEXT_LABELS = {
    5_000: "5k",
    10_000: "10k",
    20_000: "20k",
    50_000: "50k",
    100_000: "100k",
}
FEATURES = ("17", "137")
RAW_READING_DISPLAY_CAP = 300_000.0
METER_COLORS = {
    0: "#0057b8",
    1: "#b85a00",
    2: "#006b54",
    3: "#65419b",
}
METER_ANOMALY_COLORS = {
    0: "#66b5ff",
    1: "#ffaa3d",
    2: "#45c9a4",
    3: "#ae8bde",
}


def stable_jitter(raw_index: np.ndarray, width: float) -> np.ndarray:
    values = np.asarray(raw_index, dtype="uint64")
    hashed = values * np.uint64(6364136223846793005)
    hashed += np.uint64(1442695040888963407)
    unit = (hashed >> np.uint64(40)).astype("float32")
    unit /= np.float32(2**24)
    return (unit - 0.5) * width


def load_full_feature_columns() -> dict[str, np.ndarray]:
    feature_names = json.loads(
        (
            PROC / "m5_tabpfn_137_full_test_context100000_n8.work" / "fit_manifest.json"
        ).read_text(encoding="utf-8")
    )["feature_names"]
    indices = {
        name: feature_names.index(name)
        for name in (
            "meter_reading",
            "hour",
        )
    }
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in indices}
    metadata: dict[str, list[np.ndarray]] = {
        name: [] for name in ("raw_index", "anomaly", "site_id", "building_id")
    }
    expected_rows = 0
    for shard in ("head", "tail"):
        manifest = json.loads(
            (EXPORT_ROOT / shard / "manifest.json").read_text(encoding="utf-8")
        )
        matrix = np.load(EXPORT_ROOT / shard / "features.float32.npy", mmap_mode="r")
        if list(matrix.shape) != manifest["features"]["shape"]:
            raise AssertionError(f"{shard}: feature shape mismatch")
        for name, index in indices.items():
            arrays[name].append(np.asarray(matrix[:, index]).copy())
        with np.load(EXPORT_ROOT / shard / "metadata.npz") as payload:
            for name in metadata:
                metadata[name].append(np.asarray(payload[name]).copy())
        expected_rows += int(manifest["rows"])
        del matrix
    if expected_rows != 10_137_155:
        raise AssertionError("unexpected combined export size")
    result = {name: np.concatenate(parts) for name, parts in arrays.items()}
    result.update({name: np.concatenate(parts) for name, parts in metadata.items()})
    return result


def reading_percentile_within_building_meter(
    building: np.ndarray, meter: np.ndarray, reading: np.ndarray
) -> np.ndarray:
    frame = pd.DataFrame({"building": building, "meter": meter, "reading": reading})
    percentile = (
        frame.groupby(["building", "meter"], sort=False)["reading"]
        .rank(method="average", pct=True)
        .to_numpy(dtype="float32")
    )
    return percentile * 100.0


def load_raw_meter_fields(
    raw_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(
        ROOT / "data" / "raw" / "m3" / "train.csv",
        usecols=["meter", "meter_reading"],
        dtype={"meter": "int8", "meter_reading": "float32"},
    )
    if int(raw_index.max()) >= len(frame):
        raise AssertionError("raw_index exceeds frozen M3 frame")
    meter = frame["meter"].to_numpy()[raw_index]
    reading = frame["meter_reading"].to_numpy()[raw_index]
    if set(np.unique(meter)) != set(METER_NAMES):
        raise AssertionError("holdout does not contain all four meter types")
    return meter, reading


def primary_use_codes(building: np.ndarray) -> tuple[np.ndarray, list[str]]:
    metadata = pd.read_csv(
        ROOT / "data" / "raw" / "m3" / "building_metadata.csv",
        usecols=["building_id", "primary_use"],
    )
    categories = sorted(metadata["primary_use"].dropna().unique())
    category_to_code = {name: code for code, name in enumerate(categories)}
    lookup = np.full(int(metadata["building_id"].max()) + 1, -1, dtype="int16")
    for row in metadata.itertuples(index=False):
        lookup[int(row.building_id)] = category_to_code[row.primary_use]
    codes = lookup[building]
    if (codes < 0).any():
        raise AssertionError("missing primary_use mapping")
    return codes, categories


def scatter_panel(
    axis: plt.Axes,
    x: np.ndarray,
    score: np.ndarray,
    meter: np.ndarray,
    anomaly_mask: np.ndarray,
) -> None:
    threshold = 0.5
    axis.axhspan(threshold, 1.0, color="#d1498b", alpha=0.055, linewidth=0)
    for meter_id in range(4):
        selected = (meter == meter_id) & ~anomaly_mask
        axis.scatter(
            x[selected],
            score[selected],
            s=1,
            marker=",",
            color=METER_COLORS[meter_id],
            alpha=0.72,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
    for meter_id in range(4):
        selected = anomaly_mask & (meter == meter_id)
        axis.scatter(
            x[selected],
            score[selected],
            s=1,
            marker=",",
            color=METER_ANOMALY_COLORS[meter_id],
            alpha=0.96,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
    axis.axhline(
        threshold,
        color="#d1498b",
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=4,
    )
    axis.set_ylim(-0.01, 1.01)
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.set_facecolor(SURFACE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(AXIS)
    axis.spines["bottom"].set_color(AXIS)
    axis.tick_params(colors=SECONDARY_INK, labelsize=8)
    axis.grid(axis="both", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)


def plot_covariate(
    scores: dict[tuple[str, int], np.ndarray],
    *,
    context: int,
    x: np.ndarray,
    anomaly_mask: np.ndarray,
    meter: np.ndarray,
    title: str,
    subtitle: str,
    xlabel: str,
    output: Path,
    categorical_labels: list[str] | None = None,
    xscale: str | None = None,
    xticks: list[float] | None = None,
    xticklabels: list[str] | None = None,
    legend_meter_ids: tuple[int, ...] = (0, 1, 2, 3),
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), sharex=True, sharey=True)
    figure.patch.set_facecolor(SURFACE)
    add_title(
        figure,
        title=f"{title} — {CONTEXT_LABELS[context]} training context",
        subtitle=subtitle,
        title_y=0.98,
        subtitle_y=0.935,
    )
    for row, model in enumerate(MODELS):
        for column, feature_text in enumerate(FEATURES):
            feature = int(feature_text)
            axis = axes[row, column]
            scatter_panel(
                axis,
                x,
                scores[(model, feature)],
                meter,
                anomaly_mask,
            )
            axis.set_title(
                f"{MODEL_STYLE[model]['label']} — {feature} features",
                loc="left",
                fontsize=11.5,
                fontweight="bold",
                color=PRIMARY_INK,
                pad=7,
            )
            if categorical_labels is not None:
                axis.set_xticks(range(len(categorical_labels)))
                axis.set_xticklabels(
                    categorical_labels,
                    fontsize=7,
                    rotation=40,
                    ha="right",
                )
                axis.set_xlim(-0.7, len(categorical_labels) - 0.3)
            elif xscale is not None:
                axis.set_xscale(xscale, linthresh=1.0)
                axis.xaxis.set_minor_formatter(NullFormatter())
            if xticks is not None:
                axis.set_xticks(xticks)
                axis.set_xticklabels(xticklabels)
    figure.supylabel(
        "Predicted anomaly score",
        x=0.018,
        fontsize=10.5,
        color=PRIMARY_INK,
    )
    figure.supxlabel(
        xlabel,
        y=0.115 if categorical_labels is not None else 0.085,
        fontsize=10.5,
        color=PRIMARY_INK,
    )
    handles = []
    for meter_id in legend_meter_ids:
        handles.extend(
            [
                Patch(
                    facecolor=METER_COLORS[meter_id],
                    edgecolor="none",
                    label=f"{METER_NAMES[meter_id]} normal",
                ),
                Patch(
                    facecolor=METER_ANOMALY_COLORS[meter_id],
                    edgecolor="none",
                    label=f"{METER_NAMES[meter_id]} anomaly",
                ),
            ]
        )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=4,
        frameon=False,
        fontsize=7.8,
    )
    role_handles = [
        Line2D(
            [0],
            [0],
            linestyle=(0, (4, 3)),
            linewidth=1.1,
            color="#d1498b",
            label="Model cutoff: score 0.5; light-color anomaly below line = missed",
        ),
    ]
    figure.legend(
        handles=role_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.040),
        ncol=1,
        frameon=False,
        fontsize=8.2,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.985,
        top=0.84,
        bottom=0.29 if categorical_labels is not None else 0.22,
        wspace=0.10,
        hspace=0.18,
    )
    figure.savefig(output, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        choices=CONTEXTS,
        default=list(CONTEXTS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    columns = load_full_feature_columns()
    reference_index, reference_y, _, _ = load_prediction("tabpfn", "17", 5_000)
    if not np.array_equal(reference_index, columns["raw_index"]):
        raise AssertionError("feature export and prediction row order differ")
    if not np.array_equal(reference_y, columns["anomaly"]):
        raise AssertionError("feature export and prediction labels differ")
    anomaly_mask = reference_y == 1
    meter, raw_reading = load_raw_meter_fields(reference_index)
    raw_reading_overflow = raw_reading > RAW_READING_DISPLAY_CAP
    if raw_reading_overflow.any():
        overflow_meters = set(np.unique(meter[raw_reading_overflow]))
        if overflow_meters != {2} or not anomaly_mask[raw_reading_overflow].all():
            raise AssertionError(
                "raw-reading overflow is no longer exclusively anomalous steam"
            )
    raw_reading_display = np.minimum(raw_reading, RAW_READING_DISPLAY_CAP)

    print("ranking meter readings within building x meter")
    reading_percentile = reading_percentile_within_building_meter(
        columns["building_id"], meter, columns["meter_reading"]
    )
    print("mapping primary use and temporal features")
    use_code, use_labels = primary_use_codes(columns["building_id"])
    use_y = use_code.astype("float32") + stable_jitter(columns["raw_index"], 0.54)
    site_y = columns["site_id"].astype("float32") + stable_jitter(
        columns["raw_index"], 0.54
    )
    standardized_hour = columns["hour"]
    unique_hour = np.unique(standardized_hour)
    if len(unique_hour) != 24:
        raise AssertionError(
            f"expected 24 standardized hour values, got {len(unique_hour)}"
        )
    decoded_hour = np.searchsorted(unique_hour, standardized_hour).astype("float32")
    hour_y = decoded_hour + stable_jitter(columns["raw_index"], 0.54)

    for context in args.contexts:
        scores: dict[tuple[str, int], np.ndarray] = {}
        for model in MODELS:
            for feature_text in FEATURES:
                feature = int(feature_text)
                _, _, _, score = load_prediction(model, feature_text, context)
                scores[(model, feature)] = score

        common = {
            "scores": scores,
            "context": context,
            "anomaly_mask": anomaly_mask,
            "meter": meter,
        }
        plot_covariate(
            **common,
            x=raw_reading_display,
            title="Model decision by raw meter reading",
            subtitle=(
                "All 10,137,155 rows. Light = true anomaly. Values above 300k are placed at "
                "right: 3,067 steam anomalies from site 13 / building 1099."
            ),
            xlabel="Raw meter_reading (symlog scale; values above 300k at right edge)",
            output=FIGURE_ROOT
            / f"m5_context_{CONTEXT_LABELS[context]}_score_covariate_raw_reading.png",
            xscale="symlog",
            xticks=[0, 1, 10, 100, 1_000, 10_000, 100_000, 300_000],
            xticklabels=["0", "1", "10", "100", "1k", "10k", "100k", "≥300k"],
        )
        plot_covariate(
            **common,
            x=reading_percentile,
            title="Model decision by relative meter reading",
            subtitle=(
                "All 10,137,155 rows are shown; high-opacity light-shade points are ground-truth "
                "anomalies. X controls meter units and building scale."
            ),
            xlabel="Meter-reading percentile within each building and meter",
            output=FIGURE_ROOT
            / f"m5_context_{CONTEXT_LABELS[context]}_score_covariate_reading.png",
        )
        plot_covariate(
            **common,
            x=site_y,
            title="Model decision by site",
            subtitle=(
                "All 10,137,155 rows are shown; high-opacity light-shade points are ground-truth "
                "anomalies. Vertical bands reveal site-specific selection."
            ),
            xlabel="Site ID (jittered)",
            output=FIGURE_ROOT
            / f"m5_context_{CONTEXT_LABELS[context]}_score_covariate_site.png",
            categorical_labels=[str(value) for value in range(16)],
        )
        plot_covariate(
            **common,
            x=use_y,
            title="Model decision by building primary use",
            subtitle=(
                "All 10,137,155 rows are shown; high-opacity light-shade points are ground-truth "
                "anomalies. Vertical jitter only separates overlapping points."
            ),
            xlabel="Building primary use (jittered)",
            output=FIGURE_ROOT
            / f"m5_context_{CONTEXT_LABELS[context]}_score_covariate_primary_use.png",
            categorical_labels=use_labels,
        )
        plot_covariate(
            **common,
            x=hour_y,
            title="Model decision by hour of day",
            subtitle=(
                "All 10,137,155 rows are shown; high-opacity light-shade points are ground-truth "
                "anomalies. Vertical jitter only separates overlapping hours."
            ),
            xlabel="Hour of day (jittered)",
            output=FIGURE_ROOT
            / f"m5_context_{CONTEXT_LABELS[context]}_score_covariate_hour.png",
            categorical_labels=[str(value) for value in range(24)],
        )
        print(f"plotted covariate scatters for {context:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
