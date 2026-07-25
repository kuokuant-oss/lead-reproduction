"""Render canonical standalone M3 report figures from observation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from lead import PROC, ROOT, write_json_with_provenance


SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ANOMALY = "#b23a48"
MODEL_STYLE = {
    "lightgbm": {"label": "LightGBM", "color": "#2a78d6", "marker": "o"},
    "xgboost": {"label": "XGBoost", "color": "#e07a00", "marker": "s"},
    "catboost": {"label": "CatBoost", "color": "#008b6d", "marker": "^"},
    "hist_gradient_boosting": {
        "label": "HistGBT",
        "color": "#8b65c2",
        "marker": "D",
    },
    "ensemble": {"label": "Tree Ensemble", "color": INK, "marker": "p"},
}
MODEL_ORDER = (
    "lightgbm",
    "xgboost",
    "catboost",
    "hist_gradient_boosting",
    "ensemble",
)
FEATURE_BASELINE_KEY = "m3_1_ensemble"
TABPFN_KEY = "tabpfn"
TABPFN_137_KEY = "tabpfn_137"
DEFAULT_17_FEATURE_ENSEMBLE_PREDICTIONS = (
    PROC / "m3_17_feature_ensemble_predictions.npz"
)
DEFAULT_TABPFN_PREDICTIONS = (
    PROC / "m5_tabpfn_distributed_context100000_predictions.npz"
)
DEFAULT_TABPFN_137_PREDICTIONS = PROC / "m5_tabpfn_137_full_test_n8_predictions.npz"


def _style_axis(ax: plt.Axes, *, horizontal_grid: bool = True) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    if horizontal_grid:
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)


def _title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(
        title,
        x=0.06,
        y=0.965,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(0.06, 0.895, subtitle, ha="left", fontsize=10.5, color=SECONDARY)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)


def render_confusion(data: dict[str, Any], path: Path) -> None:
    matrix_data = data["metrics"]["ensemble"]["threshold_0_5"]["confusion_matrix"]
    counts = np.array(
        [
            [matrix_data["tn"], matrix_data["fp"]],
            [matrix_data["fn"], matrix_data["tp"]],
        ],
        dtype=float,
    )
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts, row_totals, out=np.zeros_like(counts), where=row_totals != 0
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.1), facecolor=SURFACE)
    fig.suptitle(
        "137-Feature Tree Ensemble Confusion Matrix",
        x=0.06,
        y=0.95,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    labels = (("TN", "FP"), ("FN", "TP"))
    for row in range(2):
        for col in range(2):
            value = normalized[row, col]
            label = labels[row][col]
            ax.text(
                col,
                row,
                f"{label}\n{int(counts[row, col]):,}\n{value:.1%}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="normal",
                color="white" if value > 0.55 else INK,
            )
    ax.set_xticks([0, 1], labels=["Normal", "Anomaly"])
    ax.set_yticks([0, 1], labels=["Normal", "Anomaly"])
    ax.tick_params(colors=SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.045)
    colorbar.set_label("Within-class share", color=SECONDARY)
    colorbar.ax.tick_params(colors=SECONDARY)
    fig.subplots_adjust(left=0.19, right=0.86, top=0.86, bottom=0.14)
    _save(fig, path)


def _render_value_change_context_curves(data: dict[str, Any], path: Path) -> None:
    segment = data["value_change_illustration"]
    frame = pd.DataFrame(segment["points"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    anchor = pd.Timestamp(segment["anchor_timestamp"])
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.4, 7.8),
        sharex=True,
        facecolor=SURFACE,
        gridspec_kw={"height_ratios": [1.5, 1, 1]},
    )
    _title(
        fig,
        "How One-Hour Value Changes Describe a Meter Reading",
        f"Building {segment['building_id']} · meter {segment['meter']} · exact timestamp merge at a one-hour lag",
    )
    axes[0].plot(
        frame["timestamp"],
        frame["meter_reading"],
        color=INK,
        linewidth=1.2,
        marker="o",
        markersize=3.2,
    )
    anomaly = frame[frame["anomaly"] == 1]
    if not anomaly.empty:
        axes[0].axvspan(
            anomaly["timestamp"].min(),
            anomaly["timestamp"].max(),
            color=ANOMALY,
            alpha=0.07,
            linewidth=0,
            label="Labeled anomaly interval",
        )
    highlighted = anomaly[
        anomaly["timestamp"].isin([anchor - pd.Timedelta(hours=1), anchor])
    ]
    axes[0].scatter(
        highlighted["timestamp"],
        highlighted["meter_reading"],
        color=ANOMALY,
        marker="X",
        s=42,
        zorder=3,
    )
    axes[0].axvline(anchor, color=ANOMALY, linewidth=1, linestyle="--")
    axes[0].set_ylabel("Meter reading")
    axes[0].set_title("Meter reading", loc="left", fontweight="bold", fontsize=11.5)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)

    axes[1].plot(
        frame["timestamp"], frame["difference"], color="#2a78d6", linewidth=1.3
    )
    axes[1].axhline(0, color=SECONDARY, linewidth=0.9)
    axes[1].set_ylabel("Difference")
    axes[1].set_title(
        "One-hour difference", loc="left", fontweight="bold", fontsize=11.5, pad=14
    )
    axes[1].text(
        0.205,
        1.01,
        "current reading − reading one hour earlier",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=SECONDARY,
    )

    axes[2].plot(frame["timestamp"], frame["ratio"], color="#008b6d", linewidth=1.3)
    axes[2].axhline(1, color=SECONDARY, linewidth=0.9)
    axes[2].set_ylabel("Ratio")
    axes[2].set_title(
        "One-hour ratio", loc="left", fontweight="bold", fontsize=11.5, pad=14
    )
    axes[2].text(
        0.145,
        1.01,
        "(current + 1) / (reading one hour earlier + 1)",
        transform=axes[2].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=SECONDARY,
    )
    axes[2].set_xlabel("Timestamp")
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b-%d\n%H:%M"))

    anchor_row = frame.loc[frame["timestamp"] == anchor]
    previous_row = frame.loc[frame["timestamp"] == anchor - pd.Timedelta(hours=1)]
    if not anchor_row.empty and not previous_row.empty:
        current = float(anchor_row.iloc[0]["meter_reading"])
        previous = float(previous_row.iloc[0]["meter_reading"])
        difference = float(anchor_row.iloc[0]["difference"])
        ratio = float(anchor_row.iloc[0]["ratio"])
        axes[0].annotate(
            f"Previous: {previous:g}",
            xy=(anchor - pd.Timedelta(hours=1), previous),
            xytext=(-76, 35),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": SECONDARY, "lw": 0.9},
            fontsize=8.5,
            color=SECONDARY,
        )
        axes[0].annotate(
            f"Anchor: {current:g}",
            xy=(anchor, current),
            xytext=(16, 29),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": ANOMALY, "lw": 0.9},
            fontsize=8.5,
            color=ANOMALY,
        )
        axes[1].annotate(
            f"{current:g} − {previous:g} = {difference:g}",
            xy=(anchor, difference),
            xytext=(15, -5),
            textcoords="offset points",
            fontsize=8.5,
            color=ANOMALY,
        )
        axes[2].annotate(
            f"({current:g} + 1) / ({previous:g} + 1) = {ratio:.2f}",
            xy=(anchor, ratio),
            xytext=(15, 9),
            textcoords="offset points",
            fontsize=8.5,
            color=ANOMALY,
        )

    missing = frame["difference"].isna() | frame["ratio"].isna()
    if missing.any():
        missing_times = frame.loc[missing, "timestamp"]
        for timestamp in missing_times:
            start = timestamp - pd.Timedelta(minutes=27)
            end = timestamp + pd.Timedelta(minutes=27)
            for ax in axes[1:]:
                ax.axvspan(start, end, color=GRID, alpha=0.55, linewidth=0)
        axes[1].text(
            missing_times.min(),
            0.94,
            "Missing exact match → NaN",
            transform=axes[1].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=MUTED,
        )

    for ax in axes:
        _style_axis(ax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    axes[2].margins(y=0.08)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.11, hspace=0.43)
    _save(fig, path)


def _render_value_change_steps(data: dict[str, Any], path: Path) -> None:
    segment = data["value_change_illustration"]
    frame = pd.DataFrame(segment["points"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    anchor = pd.Timestamp(segment["anchor_timestamp"])
    anchor_row = frame.loc[frame["timestamp"] == anchor]
    previous_row = frame.loc[frame["timestamp"] == anchor - pd.Timedelta(hours=1)]
    if anchor_row.empty or previous_row.empty:
        available = frame.loc[frame["timestamp"] <= anchor].tail(2)
        if len(available) < 2:
            raise ValueError("Value-change illustration needs two adjacent readings")
        previous_row = available.iloc[[0]]
        anchor_row = available.iloc[[1]]
        anchor = pd.Timestamp(anchor_row.iloc[0]["timestamp"])

    current = float(anchor_row.iloc[0]["meter_reading"])
    previous = float(previous_row.iloc[0]["meter_reading"])
    previous_timestamp = pd.Timestamp(previous_row.iloc[0]["timestamp"])
    difference = float(anchor_row.iloc[0]["difference"])
    ratio = float(anchor_row.iloc[0]["ratio"])

    fig = plt.figure(figsize=(11.4, 7.2), facecolor=SURFACE)
    grid = fig.add_gridspec(
        3, 2, height_ratios=[2.5, 0.9, 1.35], hspace=0.25, wspace=0.12
    )
    meter_ax = fig.add_subplot(grid[0, :])
    pair_ax = fig.add_subplot(grid[1, :])
    difference_ax = fig.add_subplot(grid[2, 0])
    ratio_ax = fig.add_subplot(grid[2, 1])
    _title(
        fig,
        "From Adjacent Readings to Value-Change Features",
        f"Building {segment['building_id']} · meter {segment['meter']} · selected pair is exactly one hour apart",
    )

    meter_ax.plot(
        frame["timestamp"],
        frame["meter_reading"],
        color=INK,
        linewidth=1.2,
        marker="o",
        markersize=3.2,
    )
    anomaly = frame[frame["anomaly"] == 1]
    if not anomaly.empty:
        meter_ax.axvspan(
            anomaly["timestamp"].min(),
            anomaly["timestamp"].max(),
            color=ANOMALY,
            alpha=0.07,
            linewidth=0,
            label="Labeled anomaly interval",
        )
    meter_ax.plot(
        [previous_timestamp, anchor],
        [previous, current],
        color=ANOMALY,
        linewidth=2.0,
        marker="o",
        markersize=6,
        zorder=4,
    )
    meter_ax.annotate(
        f"Previous reading\n{previous_timestamp:%b %d, %H:%M} · {previous:g}",
        xy=(previous_timestamp, previous),
        xytext=(-95, 45),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": SECONDARY, "lw": 0.9},
        fontsize=8.5,
        color=SECONDARY,
    )
    meter_ax.annotate(
        f"Current reading\n{anchor:%b %d, %H:%M} · {current:g}",
        xy=(anchor, current),
        xytext=(24, 28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": ANOMALY, "lw": 0.9},
        fontsize=8.5,
        color=ANOMALY,
    )
    meter_ax.set_ylabel("Meter reading")
    meter_ax.set_title(
        "Observed meter reading", loc="left", fontweight="bold", fontsize=11.5
    )
    meter_ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    meter_ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    meter_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d\n%H:%M"))
    _style_axis(meter_ax)
    meter_ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))

    pair_ax.axis("off")
    pair_ax.text(
        0.02,
        0.78,
        "Step 1 · Select two readings exactly one hour apart",
        transform=pair_ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    for x, label, value, facecolor, edgecolor in (
        (0.25, "Previous (t−1)", previous, "#f1f4f8", AXIS),
        (0.63, "Current (t)", current, "#fff4f4", ANOMALY),
    ):
        pair_ax.add_patch(
            FancyBboxPatch(
                (x, 0.08),
                0.20,
                0.52,
                transform=pair_ax.transAxes,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                linewidth=1.1,
                edgecolor=edgecolor,
                facecolor=facecolor,
            )
        )
        pair_ax.text(
            x + 0.10,
            0.34,
            f"{label}\n{value:g}",
            transform=pair_ax.transAxes,
            ha="center",
            va="center",
            fontsize=10.5,
            color=INK,
        )
    pair_ax.annotate(
        "one hour",
        xy=(0.62, 0.34),
        xytext=(0.46, 0.34),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=SECONDARY,
        arrowprops={"arrowstyle": "-|>", "color": SECONDARY, "lw": 1.1},
    )

    for ax in (difference_ax, ratio_ax):
        ax.axis("off")
    difference_ax.text(
        0.02,
        0.88,
        "Step 2 · Difference",
        transform=difference_ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    difference_ax.text(
        0.02,
        0.62,
        "current − previous",
        transform=difference_ax.transAxes,
        fontsize=9,
        color=SECONDARY,
    )
    difference_ax.text(
        0.02,
        0.37,
        f"{current:g} − {previous:g} = {difference:g}",
        transform=difference_ax.transAxes,
        fontsize=17,
        fontweight="bold",
        color="#2a78d6",
    )
    difference_ax.text(
        0.02,
        0.12,
        f"Interpretation: a {abs(difference):g}-unit decrease",
        transform=difference_ax.transAxes,
        fontsize=9,
        color=SECONDARY,
    )

    ratio_ax.text(
        0.02,
        0.88,
        "Step 3 · Ratio",
        transform=ratio_ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    ratio_ax.text(
        0.02,
        0.62,
        "(current + 1) / (previous + 1)",
        transform=ratio_ax.transAxes,
        fontsize=9,
        color=SECONDARY,
    )
    ratio_ax.text(
        0.02,
        0.37,
        f"({current:g} + 1) / ({previous:g} + 1) = {ratio:.2f}",
        transform=ratio_ax.transAxes,
        fontsize=17,
        fontweight="bold",
        color="#008b6d",
    )
    ratio_ax.text(
        0.02,
        0.12,
        f"Interpretation: current is {ratio:.0%} of previous (+1 smoothing)",
        transform=ratio_ax.transAxes,
        fontsize=9,
        color=SECONDARY,
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.07)
    _save(fig, path)


def render_value_change(data: dict[str, Any], path: Path) -> None:
    segment = data["value_change_illustration"]
    frame = pd.DataFrame(segment["points"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    by_timestamp = frame.set_index("timestamp")["meter_reading"]
    candidates: list[tuple[float, pd.Timestamp, pd.Timestamp, float, float]] = []
    for current_timestamp, current_value in by_timestamp.items():
        previous_timestamp = current_timestamp - pd.Timedelta(hours=1)
        if previous_timestamp not in by_timestamp.index:
            continue
        previous_value = float(by_timestamp.loc[previous_timestamp])
        current_value = float(current_value)
        candidates.append(
            (
                abs(current_value - previous_value),
                previous_timestamp,
                current_timestamp,
                previous_value,
                current_value,
            )
        )
    if not candidates:
        raise ValueError("Value-change illustration needs an exact one-hour pair")
    _, previous_timestamp, current_timestamp, previous, current = max(candidates)
    difference = current - previous
    ratio = (current + 1) / (previous + 1)

    fig, ax = plt.subplots(figsize=(10.4, 6.2), facecolor=SURFACE)
    _title(
        fig,
        "Difference and Ratio Between Adjacent Meter Readings",
        f"Building {segment['building_id']} · meter {segment['meter']} · selected one-hour pair has the largest absolute change",
    )
    ax.plot(
        frame["timestamp"],
        frame["meter_reading"],
        color="#2a78d6",
        linewidth=1.7,
        label=f"Building {segment['building_id']} · meter {segment['meter']}",
    )
    ax.scatter(
        [previous_timestamp, current_timestamp],
        [previous, current],
        s=70,
        facecolors=SURFACE,
        edgecolors=SECONDARY,
        linewidths=1.5,
        zorder=4,
    )

    red = "#d20f18"
    green = "#4f7f2d"
    difference_x = current_timestamp + pd.Timedelta(hours=2.0)
    ax.hlines(
        [previous, current],
        xmin=[previous_timestamp, current_timestamp],
        xmax=difference_x,
        colors=red,
        linewidth=1.8,
    )
    ax.annotate(
        "",
        xy=(difference_x, previous),
        xytext=(difference_x, current),
        arrowprops={"arrowstyle": "<->", "color": red, "lw": 1.8},
    )
    ax.text(
        difference_x + pd.Timedelta(hours=0.4),
        (previous + current) / 2,
        f"Difference\n{current:g} − {previous:g} = {difference:g}",
        color=red,
        fontsize=11,
        va="center",
    )

    ratio_previous_x = previous_timestamp - pd.Timedelta(hours=7.0)
    ratio_current_x = previous_timestamp - pd.Timedelta(hours=4.5)
    cap = pd.Timedelta(hours=0.8)
    for x, value in ((ratio_previous_x, previous), (ratio_current_x, current)):
        ax.annotate(
            "",
            xy=(x, value),
            xytext=(x, 0),
            arrowprops={"arrowstyle": "<->", "color": green, "lw": 1.6},
        )
        ax.hlines([0, value], xmin=x - cap, xmax=x + cap, colors=green, linewidth=1.6)
    ax.text(
        ratio_previous_x - pd.Timedelta(hours=0.3),
        max(previous, current) * 0.53,
        f"Ratio\n({current:g} + 1) / ({previous:g} + 1)\n= {ratio:.2f}×",
        color=green,
        fontsize=11,
        va="center",
        ha="right",
    )

    _style_axis(ax, horizontal_grid=False)
    ax.set_ylabel("Meter reading")
    ax.set_xlabel("Timestamp")
    ax.set_ylim(-3, float(frame["meter_reading"].max()) * 1.14)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d-%b\n%Y"))
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.subplots_adjust(left=0.11, right=0.94, top=0.80, bottom=0.20)
    _save(fig, path)


def _plot_curve(
    ax: plt.Axes,
    curve: dict[str, Any],
    *,
    label: str,
    color: str,
) -> None:
    x = np.asarray(curve["x"], dtype=float)
    y = np.asarray(curve["y"], dtype=float)
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    ax.plot(
        x,
        y,
        label=label,
        color=color,
        linewidth=1.45,
    )


def add_17_feature_ensemble_comparison(
    data: dict[str, Any], predictions_path: Path
) -> None:
    """Add the matched 17-feature Tree Ensemble metrics and curves."""
    with np.load(predictions_path) as predictions:
        required = {"anomaly", "ensemble"}
        missing = required - set(predictions.files)
        if missing:
            raise ValueError(f"missing 17-feature ensemble arrays: {sorted(missing)}")
        y_true = np.asarray(predictions["anomaly"])
        score = np.asarray(predictions["ensemble"])

    expected_rows = data.get("split", {}).get("validation_rows")
    if expected_rows is not None and len(y_true) != expected_rows:
        raise ValueError(
            "17-feature ensemble validation rows do not match the M3 observations: "
            f"{len(y_true)} != {expected_rows}"
        )

    fpr, tpr, _ = roc_curve(y_true, score)
    precision, recall, _ = precision_recall_curve(y_true, score)
    data["metrics"][FEATURE_BASELINE_KEY] = {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
    }
    data["curves"][FEATURE_BASELINE_KEY] = {
        "roc": {"x": fpr, "y": tpr},
        "precision_recall": {"x": recall, "y": precision},
    }


def add_tabpfn_comparison(
    data: dict[str, Any], predictions_path: Path, *, key: str = TABPFN_KEY
) -> None:
    """Add a merged TabPFN line's metrics and curves under ``key``.

    The 17-feature TabPFN (``TABPFN_KEY``) scores the same 17 baseline features as
    ``FEATURE_BASELINE_KEY``; the 137-feature TabPFN (``TABPFN_137_KEY``) scores
    the same 137 features as the blue Tree Ensemble. Both share the plotted row
    order, so each drops straight onto these figures.
    """
    with np.load(predictions_path) as predictions:
        required = {"anomaly", "tabpfn"}
        missing = required - set(predictions.files)
        if missing:
            raise ValueError(f"missing TabPFN arrays: {sorted(missing)}")
        y_true = np.asarray(predictions["anomaly"])
        score = np.asarray(predictions["tabpfn"])

    expected_rows = data.get("split", {}).get("validation_rows")
    if expected_rows is not None and len(y_true) != expected_rows:
        raise ValueError(
            "TabPFN validation rows do not match the M3 observations: "
            f"{len(y_true)} != {expected_rows}"
        )
    if not np.isfinite(score).all():
        raise ValueError("TabPFN scores contain non-finite values")

    fpr, tpr, _ = roc_curve(y_true, score)
    precision, recall, _ = precision_recall_curve(y_true, score)
    data["metrics"][key] = {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
    }
    data["curves"][key] = {
        "roc": {"x": fpr, "y": tpr},
        "precision_recall": {"x": recall, "y": precision},
    }


def render_discrimination_curve(
    data: dict[str, Any],
    path: Path,
    *,
    comparison: str,
    curve_type: str,
) -> None:
    """Render one comparison and one curve type as a standalone figure."""
    is_roc = curve_type == "roc"
    if curve_type not in {"roc", "precision_recall"}:
        raise ValueError(f"Unsupported curve type: {curve_type}")
    if comparison not in {
        "models",
        "feature_engineering",
        "feature_engineering_tabpfn",
    }:
        raise ValueError(f"Unsupported comparison: {comparison}")

    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor=SURFACE)
    if comparison == "models":
        series = [
            (
                model_name,
                MODEL_STYLE[model_name]["label"],
                MODEL_STYLE[model_name]["color"],
            )
            for model_name in MODEL_ORDER
        ]
        title_prefix = "Tree-Model Comparison"
        subtitle = (
            "Final 50/50 building holdout · zoomed to the observed performance range"
        )
    else:
        series = [
            (FEATURE_BASELINE_KEY, "17 baseline features", "#898781"),
            ("ensemble", "137 features", "#2a78d6"),
        ]
        if comparison == "feature_engineering_tabpfn":
            series.append((TABPFN_KEY, "TabPFN (17 features, context 100k)", "#d1580f"))
            if TABPFN_137_KEY in data["curves"]:
                series.append(
                    (
                        TABPFN_137_KEY,
                        "TabPFN (137 features, context 100k)",
                        "#7a51a8",
                    )
                )
        title_prefix = "Feature-Engineering Contribution"
        subtitle = "Tree Ensemble on the same final 50/50 building holdout · 17 versus 137 features"

    metric_key = "roc_auc" if is_roc else "pr_auc"
    score_name = "AUC" if is_roc else "PR-AUC"
    for key, label, color in series:
        score = data["metrics"][key][metric_key]
        _plot_curve(
            ax,
            data["curves"][key][curve_type],
            label=f"{label}  {score_name} = {score:.4f}",
            color=color,
        )

    curve_name = "ROC" if is_roc else "Precision–Recall"
    _title(fig, f"{title_prefix}: {curve_name}", subtitle)
    _style_axis(ax, horizontal_grid=False)
    ax.set_box_aspect(1)
    if is_roc:
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

    if comparison == "models" and is_roc:
        ax.set_xlim(-0.002, 0.082)
        ax.set_ylim(0.918, 1.002)
        ax.set_xticks(np.arange(0.00, 0.081, 0.02))
        ax.set_yticks(np.arange(0.92, 1.001, 0.01))
    elif comparison == "models":
        ax.set_xlim(0.645, 1.005)
        ax.set_ylim(0.645, 1.005)
        ax.set_xticks(np.arange(0.65, 1.01, 0.05))
        ax.set_yticks(np.arange(0.65, 1.01, 0.05))
    else:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))

    ax.legend(
        frameon=False,
        fontsize=8.8,
        ncol=2 if comparison == "models" else 1,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        columnspacing=1.4,
        handlelength=2.4,
    )
    # Each extra stacked legend row needs a little more room beneath the axes.
    if comparison == "models":
        legend_bottom = 0.24
    else:
        legend_bottom = {2: 0.24, 3: 0.26, 4: 0.30}.get(len(series), 0.24)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.78, bottom=legend_bottom)
    _save(fig, path)


def render_discrimination_figures(
    data: dict[str, Any], asset_dir: Path
) -> dict[str, Path]:
    figures = {
        "model_precision_recall": asset_dir
        / "m3_model_comparison_precision_recall_zoomed.png",
        "model_roc": asset_dir / "m3_model_comparison_roc_zoomed.png",
        "feature_precision_recall": asset_dir
        / "m3_feature_engineering_precision_recall.png",
        "feature_roc": asset_dir / "m3_feature_engineering_roc.png",
    }
    render_discrimination_curve(
        data,
        figures["model_precision_recall"],
        comparison="models",
        curve_type="precision_recall",
    )
    render_discrimination_curve(
        data, figures["model_roc"], comparison="models", curve_type="roc"
    )
    render_discrimination_curve(
        data,
        figures["feature_precision_recall"],
        comparison="feature_engineering",
        curve_type="precision_recall",
    )
    render_discrimination_curve(
        data,
        figures["feature_roc"],
        comparison="feature_engineering",
        curve_type="roc",
    )
    if TABPFN_KEY in data["curves"]:
        figures["feature_precision_recall_with_tabpfn"] = (
            asset_dir / "m3_feature_engineering_precision_recall_with_tabpfn.png"
        )
        figures["feature_roc_with_tabpfn"] = (
            asset_dir / "m3_feature_engineering_roc_with_tabpfn.png"
        )
        render_discrimination_curve(
            data,
            figures["feature_precision_recall_with_tabpfn"],
            comparison="feature_engineering_tabpfn",
            curve_type="precision_recall",
        )
        render_discrimination_curve(
            data,
            figures["feature_roc_with_tabpfn"],
            comparison="feature_engineering_tabpfn",
            curve_type="roc",
        )
    return figures


def _importance_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    title: str,
    value_key: str,
    color: str,
    feature_order: list[str],
    x_limit: float,
) -> None:
    by_feature = {row["feature"]: row for row in rows}
    ordered = feature_order[::-1]
    values = [
        float(by_feature.get(feature, {}).get(value_key, 0.0)) for feature in ordered
    ]
    ax.barh(
        ordered,
        values,
        color=color,
        alpha=0.9,
    )
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_xlim(0, x_limit)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11.5)
    _style_axis(ax, horizontal_grid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.tick_params(axis="y", labelsize=8)


def _importance_feature_order(rows: list[dict[str, Any]], value_key: str) -> list[str]:
    return [
        row["feature"]
        for row in sorted(
            rows,
            key=lambda row: float(row.get(value_key, 0.0)),
            reverse=True,
        )[:10]
    ]


def _importance_x_limit(data: dict[str, Any]) -> float:
    importance = data["permutation_importance"]
    comparable = [
        *[row for model in MODEL_ORDER[:4] for row in importance["models"][model]],
        *importance["ensemble"],
        *importance["consensus"],
    ]
    x_limit = (
        max(
            float(
                row.get("roc_auc_decrease_mean", row.get("mean_roc_auc_decrease", 0.0))
            )
            for row in comparable
        )
        * 1.08
    )
    return x_limit


def render_importance_figure(
    data: dict[str, Any],
    path: Path,
    *,
    source: str,
) -> None:
    importance = data["permutation_importance"]
    x_limit = _importance_x_limit(data)
    if source in MODEL_ORDER[:4]:
        style = MODEL_STYLE[source]
        rows = importance["models"][source]
        title = style["label"]
        value_key = "roc_auc_decrease_mean"
        color = style["color"]
    elif source == "ensemble":
        rows = importance["ensemble"]
        title = "Tree Ensemble"
        value_key = "roc_auc_decrease_mean"
        color = INK
    elif source == "consensus":
        rows = importance["consensus"]
        title = "Four-Model Consensus"
        value_key = "mean_roc_auc_decrease"
        color = "#52514e"
    else:
        raise ValueError(f"Unsupported importance source: {source}")
    feature_order = _importance_feature_order(rows, value_key)

    fig, ax = plt.subplots(figsize=(7.3, 5.9), facecolor=SURFACE)
    _title(
        fig,
        f"Permutation Importance: {title}",
        "Own top 10 · descending AUC contribution · shared 50/50 sample and x-scale",
    )
    _importance_panel(
        ax,
        rows,
        title="",
        value_key=value_key,
        color=color,
        feature_order=feature_order,
        x_limit=x_limit,
    )
    ax.set_xlabel("ROC-AUC decrease after permutation")
    fig.subplots_adjust(left=0.30, right=0.95, top=0.77, bottom=0.14)
    _save(fig, path)


def render_importance_figures(data: dict[str, Any], asset_dir: Path) -> dict[str, Path]:
    figures = {
        "importance_lightgbm": asset_dir / "m3_permutation_importance_lightgbm.png",
        "importance_xgboost": asset_dir / "m3_permutation_importance_xgboost.png",
        "importance_catboost": asset_dir / "m3_permutation_importance_catboost.png",
        "importance_histgbt": asset_dir / "m3_permutation_importance_histgbt.png",
        "importance_ensemble": asset_dir
        / "m3_permutation_importance_tree_ensemble.png",
        "importance_consensus": asset_dir
        / "m3_permutation_importance_four_model_consensus.png",
    }
    sources = {
        "importance_lightgbm": "lightgbm",
        "importance_xgboost": "xgboost",
        "importance_catboost": "catboost",
        "importance_histgbt": "hist_gradient_boosting",
        "importance_ensemble": "ensemble",
        "importance_consensus": "consensus",
    }
    for name, source in sources.items():
        render_importance_figure(data, figures[name], source=source)
    return figures


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "#f1f4f8",
    edgecolor: str = AXIS,
    fontsize: float = 9.5,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": SECONDARY, "lw": 1.1},
    )


def render_workflow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 8.4), facecolor=SURFACE)
    _title(
        fig,
        "M3 Anomaly-Detection Workflow",
        "Final 50/50 offline baseline · frozen feature pipeline · four-model probability ensemble",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.91, "INPUT", fontsize=8.5, fontweight="bold", color=MUTED)
    ax.text(0.03, 0.70, "FEATURES", fontsize=8.5, fontweight="bold", color=MUTED)
    ax.text(0.03, 0.46, "VALIDATION", fontsize=8.5, fontweight="bold", color=MUTED)
    ax.text(0.03, 0.29, "PREPROCESSING", fontsize=8.5, fontweight="bold", color=MUTED)
    ax.text(0.03, 0.19, "MODELS", fontsize=8.5, fontweight="bold", color=MUTED)
    _box(
        ax, 0.25, 0.84, 0.50, 0.075, "GEPIII meter readings + positional anomaly labels"
    )
    _box(
        ax,
        0.25,
        0.74,
        0.50,
        0.075,
        "Join building metadata, weather, and time features",
    )
    _box(ax, 0.15, 0.62, 0.31, 0.075, "17 baseline features", facecolor="#f1f4f8")
    _box(
        ax,
        0.54,
        0.62,
        0.31,
        0.075,
        "120 value-change features\n(exact timestamp merge)",
        facecolor="#e8f2ff",
    )
    _box(
        ax,
        0.31,
        0.51,
        0.38,
        0.07,
        "Merge into 137-feature offline table",
        facecolor="#edf3f8",
    )
    _box(ax, 0.31, 0.405, 0.38, 0.07, "50/50 building holdout · 0 building overlap")
    _box(
        ax,
        0.31,
        0.30,
        0.38,
        0.07,
        "Training-only downsampling + preserved StandardScaler",
    )
    model_x = [0.12, 0.34, 0.56, 0.78]
    model_names = ["LightGBM", "XGBoost", "CatBoost", "HistGBT\nNaN → 0"]
    model_colors = ["#e8f2ff", "#fff0de", "#e4f4ee", "#f0eafb"]
    for x, name, color in zip(model_x, model_names, model_colors, strict=True):
        _box(ax, x, 0.175, 0.18, 0.065, name, facecolor=color, fontsize=9.5)
    _box(
        ax,
        0.29,
        0.035,
        0.42,
        0.075,
        "Equal-weight probability mean\nTree Ensemble",
        facecolor="#eeeeeb",
        edgecolor=INK,
    )
    _arrow(ax, (0.5, 0.84), (0.5, 0.815))
    _arrow(ax, (0.5, 0.74), (0.31, 0.695))
    _arrow(ax, (0.5, 0.74), (0.69, 0.695))
    _arrow(ax, (0.305, 0.62), (0.42, 0.58))
    _arrow(ax, (0.695, 0.62), (0.58, 0.58))
    _arrow(ax, (0.5, 0.51), (0.5, 0.475))
    _arrow(ax, (0.5, 0.405), (0.5, 0.37))
    for x in model_x:
        _arrow(ax, (0.5, 0.30), (x + 0.09, 0.24))
        _arrow(ax, (x + 0.09, 0.175), (0.5, 0.11))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.84, bottom=0.04)
    _save(fig, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observations",
        type=Path,
        default=PROC / "m3_figure_observations_50_50.json",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=ROOT / "docs" / "reports" / "assets" / "m3",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "docs" / "metrics" / "m3-figures.json",
    )
    parser.add_argument(
        "--baseline-ensemble-predictions",
        type=Path,
        default=DEFAULT_17_FEATURE_ENSEMBLE_PREDICTIONS,
    )
    parser.add_argument(
        "--tabpfn-predictions",
        type=Path,
        default=DEFAULT_TABPFN_PREDICTIONS,
    )
    parser.add_argument(
        "--tabpfn-137-predictions",
        type=Path,
        default=DEFAULT_TABPFN_137_PREDICTIONS,
    )
    parser.add_argument(
        "--skip-tabpfn",
        action="store_true",
        help="render only the two-series feature-engineering figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.observations.read_text(encoding="utf-8"))
    if data["frozen_contract"]["split"] != "50_50_mod2":
        raise RuntimeError("M3 headline figures require the frozen 50/50 baseline")
    add_17_feature_ensemble_comparison(data, args.baseline_ensemble_predictions)
    if not args.skip_tabpfn and args.tabpfn_predictions.is_file():
        add_tabpfn_comparison(data, args.tabpfn_predictions)
        if args.tabpfn_137_predictions.is_file():
            add_tabpfn_comparison(data, args.tabpfn_137_predictions, key=TABPFN_137_KEY)
    figures = {
        "confusion": args.asset_dir / "m3_tree_ensemble_confusion_matrix.png",
        "value_change": args.asset_dir
        / "m3_value_change_difference_ratio_illustration.png",
        "workflow": args.asset_dir / "m3_anomaly_detection_workflow.png",
    }
    figures.update(render_discrimination_figures(data, args.asset_dir))
    figures.update(render_importance_figures(data, args.asset_dir))
    render_confusion(data, figures["confusion"])
    render_value_change(data, figures["value_change"])
    render_workflow(figures["workflow"])
    manifest = {
        "schema_version": 1,
        "source_observation": str(args.observations.relative_to(ROOT)),
        "source_observation_sha256": file_sha256(args.observations),
        "baseline": "50_50_mod2_offline_137_timestamp_merge",
        "feature_comparison": {
            "model": "equal-weight four-model Tree Ensemble",
            "baseline_predictions": str(
                args.baseline_ensemble_predictions.relative_to(ROOT)
            ),
            "baseline_predictions_sha256": file_sha256(
                args.baseline_ensemble_predictions
            ),
        },
        "figures": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in figures.items()
        },
        "feature_pruning": {
            "screened_candidates": len(data["permutation_importance"]["screening"]),
            "targeted_ablation": data["permutation_importance"]["targeted_ablation"],
            "canonical_feature_set_changed": False,
        },
    }
    write_json_with_provenance(
        args.manifest,
        manifest,
        root=ROOT,
        provenance={"plot_style": "docs/reference/plot-style-rules.md v0.2"},
    )
    print(f"Saved {len(figures)} figures and {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
