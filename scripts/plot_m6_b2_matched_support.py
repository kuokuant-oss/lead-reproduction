"""B1 and B2 on one axis: is it the anomaly labels, or the meters they come from?

Question answered by the figure: as the source anomaly budget grows, how does
PR-AUC on the unseen sites move — and does it matter whether the anomalies were
cut by dropping meters or by dropping anomaly rows?

B1 cuts labelled meters, which drops meter diversity and anomaly count together.
B2 keeps every meter and cuts only the anomaly rows. Two points at a similar
anomaly count but different meter counts separate the two effects.

Both arms share a split direction, so the test rows and their prevalence are
identical within a panel and the comparison is legitimate. Across panels they
are not: the two directions test at 3.60% and 10.25% prevalence.

Style contract: docs/reference/plot-style-rules.md v0.3.
Source artifacts: data/processed/m6_site_transfer_{b1,b2}_*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lead import PROC, ROOT

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

MODEL_STYLE = {
    "lightgbm": {"label": "LightGBM", "color": "#2a78d6", "marker": "o"},
    "xgboost": {"label": "XGBoost", "color": "#e07a00", "marker": "s"},
    "catboost": {"label": "CatBoost", "color": "#008b6d", "marker": "^"},
    "hist_gradient_boosting": {"label": "HistGBT", "color": "#8b65c2", "marker": "D"},
    "ensemble": {"label": "Tree Ensemble", "color": INK, "marker": "p"},
}
MODEL_ORDER = ("lightgbm", "xgboost", "catboost", "hist_gradient_boosting", "ensemble")

BUDGETS = ("50", "100", "200", "400", "all")
SEED = 42
N_POS = 410394

PANELS = (
    {"split": "a1", "flow": "Train on even sites, test on odd sites", "prev": "3.60%"},
    {"split": "a2", "flow": "Train on odd sites, test on even sites", "prev": "10.25%"},
)


def _load(name: str) -> dict[str, Any] | None:
    f = PROC / f"{name}.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text(encoding="utf-8"))
    return j if j.get("status") == "completed" else None


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def render(out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 6.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for idx, (ax, spec) in enumerate(zip(axes, PANELS, strict=True)):
        _style_axis(ax)
        split = spec["split"]

        b1 = [
            (_load(f"m6_site_transfer_b1_{split}_meters{b}_seed{SEED}"), b)
            for b in BUDGETS
        ]
        b1 = [(j, b) for j, b in b1 if j]
        b2 = _load(f"m6_site_transfer_b2_{split}_pos{N_POS}_seed{SEED}")
        if not b1 or not b2:
            raise SystemExit(f"missing cells for {split}")

        for model in MODEL_ORDER:
            st = MODEL_STYLE[model]
            x = [j["split"]["train"]["anomalies"] for j, _ in b1]
            y = [j["metrics"][model]["pr_auc"] for j, _ in b1]
            ax.plot(
                x,
                y,
                color=st["color"],
                marker=st["marker"],
                markersize=4.2,
                linewidth=1.35 if model == "ensemble" else 1.0,
                zorder=3,
            )
            # B2 is a sensitivity arm, not the canonical curve: hollow marker.
            ax.plot(
                [b2["fit"]["unique_anomaly_rows"]],
                [b2["metrics"][model]["pr_auc"]],
                color=st["color"],
                marker=st["marker"],
                markersize=6.0,
                markerfacecolor="none",
                markeredgewidth=1.2,
                linestyle="none",
                zorder=4,
            )

        ax.set_xscale("log")
        # Ticks sit on the actual B1 budgets, not on round numbers: with round
        # ticks the points float between them and the budget behind each point
        # is unreadable. B2's hollow marker is deliberately off-tick — its
        # anomaly count is what it is.
        ticks = [j["split"]["train"]["anomalies"] for j, _ in b1]
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [
                f"{x / 1000:.0f}k\n{'all' if b == 'all' else b + ' m'}"
                for x, (_, b) in zip(ticks, b1, strict=True)
            ]
        )
        ax.minorticks_off()
        ax.set_xlim(min(ticks) * 0.62, 1_400_000)
        ax.set_ylim(0.0, 1.04)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if idx == 0:
            ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])

        ax.set_title(
            spec["flow"],
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
            pad=20,
        )
        ax.text(
            0.0,
            1.015,
            f"test prevalence {spec['prev']}",
            transform=ax.transAxes,
            fontsize=8.5,
            color=MUTED,
            ha="left",
        )

    fig.suptitle(
        "Source anomalies against unseen-site PR-AUC",
        x=0.055,
        y=0.972,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.907,
        "Whether performance follows the number of anomaly labels, or the meters they come from.",
        ha="left",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.text(
        0.055,
        0.882,
        f"Seed {SEED}. Filled line = meter budgets (B1). Hollow = same anomalies, every meter (B2).",
        ha="left",
        fontsize=10,
        color=SECONDARY,
    )
    fig.text(
        0.5,
        0.062,
        "Source anomalies available for training, with the meter budget behind each tick (log scale)",
        ha="center",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.text(
        0.014,
        0.48,
        "Pooled PR-AUC on the held-out sites",
        va="center",
        rotation="vertical",
        fontsize=10.5,
        color=SECONDARY,
    )

    handles = [
        plt.Line2D(
            [],
            [],
            color=MODEL_STYLE[m]["color"],
            marker=MODEL_STYLE[m]["marker"],
            markersize=4.2,
            linewidth=1.35 if m == "ensemble" else 1.0,
            label=MODEL_STYLE[m]["label"],
        )
        for m in MODEL_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(MODEL_ORDER),
        frameon=False,
        fontsize=9.5,
        labelcolor=SECONDARY,
    )
    # Panels sit low enough that the panel titles clear the subtitle block.
    fig.subplots_adjust(left=0.09, right=0.98, top=0.775, bottom=0.155, wspace=0.06)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "m6_matched_anomaly_support_source_anomalies_pr_auc.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs" / "reports" / "assets" / "m6" / "b2-matched-support",
    )
    return p.parse_args()


def main() -> None:
    path = render(parse_args().out_dir)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
