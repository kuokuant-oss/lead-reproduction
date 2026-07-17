"""Render B1 training-meter learning curves per held-out site.

Question answered by each figure: as more source-site meters carry labels, how
does the held-out site's score change? One figure per (split direction, site
group, metric); PR-AUC is threshold-free, F1 and recall are read at the frozen
contract's canonical 0.5 operating point.

Style contract: docs/reference/plot-style-rules.md v0.3.
Source artifacts: data/processed/m6_site_transfer_b1_*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lead import PROC, ROOT
from m6_site_names import label as site_label

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
SEEDS = (42, 123, 999)

# Each figure: one split direction, four held-out sites.
FIGURES = (
    {
        "direction": "a1",
        "sites": (1, 3, 5, 7),
        "flow": "Train on even sites, test on odd sites",
        "slug": "even_to_odd_sites_1_3_5_7",
    },
    {
        "direction": "a1",
        "sites": (9, 11, 13, 15),
        "flow": "Train on even sites, test on odd sites",
        "slug": "even_to_odd_sites_9_11_13_15",
    },
    {
        "direction": "a2",
        "sites": (0, 2, 4, 6),
        "flow": "Train on odd sites, test on even sites",
        "slug": "odd_to_even_sites_0_2_4_6",
    },
    {
        "direction": "a2",
        "sites": (8, 10, 12, 14),
        "flow": "Train on odd sites, test on even sites",
        "slug": "odd_to_even_sites_8_10_12_14",
    },
)

# PR-AUC is threshold-free. F1 and recall are read at the frozen contract's
# canonical operating point, so the threshold travels in the axis title and the
# filename rather than being left implicit.
METRICS = (
    {
        "token": "pr_auc",
        "path": ("pr_auc",),
        "title": "Training-meter PR-AUC curves by held-out site",
        "purpose": "Whether labelling more source meters improves each unseen site's ranking.",
        "ylabel": "PR-AUC on held-out site",
        "panel": "PR-AUC (threshold-free)",
    },
    {
        "token": "f1_threshold_0_5",
        "path": ("threshold_0_5", "f1"),
        "title": "Training-meter F1 curves by held-out site",
        "purpose": "Whether labelling more source meters improves each unseen site's alarms.",
        "ylabel": "F1 at threshold 0.5 on held-out site",
        "panel": "F1 at threshold 0.5",
    },
    {
        "token": "recall_threshold_0_5",
        "path": ("threshold_0_5", "recall"),
        "title": "Training-meter recall curves by held-out site",
        "purpose": "How much of each unseen site's anomalies the default threshold catches.",
        "ylabel": "Recall at threshold 0.5 on held-out site",
        "panel": "Recall at threshold 0.5",
    },
)

# One figure per held-out site, faceted by metric. Each site is tested in exactly
# one split direction, so the direction follows from the site.
SITE_DIRECTION = {
    **{
        s: ("a1", "Train on even sites, test on odd sites")
        for s in (1, 3, 5, 7, 9, 11, 13, 15)
    },
    **{
        s: ("a2", "Train on odd sites, test on even sites")
        for s in (0, 2, 4, 6, 8, 10, 12, 14)
    },
}


def _load_cells(direction: str) -> dict[tuple[str, int], dict[str, Any]]:
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    for budget in BUDGETS:
        for seed in SEEDS:
            path = (
                PROC / f"m6_site_transfer_b1_{direction}_meters{budget}_seed{seed}.json"
            )
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") == "completed":
                cells[(budget, seed)] = payload
    return cells


def _meter_count(cell: dict[str, Any]) -> int:
    return len(cell["selection"]["selected_meters"])


def _complete_seeds(cells: dict[tuple[str, int], dict[str, Any]]) -> tuple[int, ...]:
    """Seeds that finished every budget.

    B1 runs seed-by-seed, so a partially finished seed would contribute to the
    small budgets only. Averaging it in would make N differ between x positions
    and the plotted seed range incomparable along the curve, so drop it until
    the whole seed lands.
    """
    return tuple(
        seed for seed in SEEDS if all((budget, seed) in cells for budget in BUDGETS)
    )


def global_xlim() -> tuple[float, float]:
    """One x range for every figure in the family.

    The two split directions hold different meter totals (1,033 for even->odd,
    1,347 for odd->even), so letting each figure scale to its own last budget
    would put "50 meters" at a different physical position in each figure and
    make the families impossible to read side by side. The budget ticks are
    identical everywhere; only the final "all" tick moves, because that value
    genuinely differs between directions.
    """
    counts: list[int] = []
    for direction in ("a1", "a2"):
        counts.extend(_meter_count(c) for c in _load_cells(direction).values())
    if not counts:
        raise SystemExit("no completed B1 cells found")
    return min(counts) * 0.85, max(counts) * 1.18


def _dig(node: dict[str, Any], path: tuple[str, ...]) -> float:
    for key in path:
        node = node[key]
    return float(node)


def _series(
    cells: dict[tuple[str, int], dict[str, Any]],
    site: int,
    model: str,
    seeds: tuple[int, ...],
    path: tuple[str, ...],
):
    """Return (x, mean, lo, hi, n_seeds) across the given seeds for one model."""
    xs, means, los, his, counts = [], [], [], [], []
    for budget in BUDGETS:
        vals, meters = [], None
        for seed in seeds:
            cell = cells.get((budget, seed))
            if cell is None:
                continue
            slice_ = cell["slices"]["by_site_id"].get(str(site))
            if slice_ is None:
                continue
            vals.append(_dig(slice_["models"][model], path))
            meters = _meter_count(cell)
        if not vals or meters is None:
            continue
        xs.append(meters)
        means.append(float(np.mean(vals)))
        los.append(float(np.min(vals)))
        his.append(float(np.max(vals)))
        counts.append(len(vals))
    return np.array(xs), np.array(means), np.array(los), np.array(his), counts


def _site_support(
    cells: dict[tuple[str, int], dict[str, Any]], site: int
) -> tuple[int, float]:
    for cell in cells.values():
        slice_ = cell["slices"]["by_site_id"].get(str(site))
        if slice_:
            e = slice_["models"]["ensemble"]
            return e["n_rows"], e["anomaly_rate"]
    raise KeyError(f"site {site} absent from B1 cells")


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def render_figure(
    spec: dict[str, Any],
    metric: dict[str, Any],
    out_dir: Path,
    xlim: tuple[float, float],
) -> Path:
    cells = _load_cells(spec["direction"])
    if not cells:
        raise SystemExit(f"no completed B1 cells for {spec['direction']}")

    seeds_done = _complete_seeds(cells)
    if not seeds_done:
        raise SystemExit(f"no seed has all budgets complete for {spec['direction']}")
    all_meters = sorted(
        {_meter_count(cells[(b, s)]) for b in BUDGETS for s in seeds_done}
    )

    # Portrait canvas so each panel's y span runs slightly longer than its x
    # span: the shared 0-1 PR-AUC axis needs the vertical resolution, since a
    # given site only occupies a narrow band of it. Diagonal stays close to the
    # 9.4x7.6 style baseline, so linewidth 1.0 / marker 4.2 / grid 0.7 hold.
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 10.4), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    # The curves carry no graphical variance encoding, so the widest seed-to-seed
    # gap on the figure is reported in the subtitle instead.
    spread = 0.0

    for idx, (ax, site) in enumerate(zip(axes.ravel(), spec["sites"], strict=True)):
        _style_axis(ax)
        rows, prevalence = _site_support(cells, site)

        # No random-ranker floor line: per the style rules it is only drawn when
        # the question is "better than random", and every model here sits far
        # above it. On the low-prevalence sites it also landed on the x axis and
        # read as an artefact. Each panel states its anomaly rate instead.

        for model in MODEL_ORDER:
            style = MODEL_STYLE[model]
            x, mean, lo, hi, _ = _series(cells, site, model, seeds_done, metric["path"])
            if x.size == 0:
                continue
            spread = max(spread, float((hi - lo).max()) if x.size else 0.0)
            ax.plot(
                x,
                mean,
                color=style["color"],
                marker=style["marker"],
                markersize=4.2,
                linewidth=1.35 if model == "ensemble" else 1.0,
                zorder=3,
            )

        ax.set_title(
            site_label(site, with_id=True),
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
            pad=23,
        )
        ax.text(
            0.0,
            1.015,
            f"{prevalence:.2%} anomalies · {rows / 1000:,.0f}k rows",
            transform=ax.transAxes,
            fontsize=8.5,
            color=MUTED,
            ha="left",
        )
        ax.set_xscale("log")
        ax.set_xticks(all_meters)
        labels = [
            f"{m:,}\n(all)" if m == all_meters[-1] else f"{m:,}" for m in all_meters
        ]
        ax.set_xticklabels(labels)
        ax.minorticks_off()
        # Log autoscale pads far beyond the budget grid and strands the data in
        # the right half of the panel; pin every figure to the same range.
        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, 1.04)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if idx % 2 == 0:
            ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])

    fig.suptitle(
        metric["title"],
        x=0.06,
        y=0.982,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    seed_note = (
        f"Mean of {len(seeds_done)} seeds, widest seed gap {spread:.2f}"
        if len(seeds_done) > 1
        else "Single seed"
    )
    fig.text(
        0.06,
        0.936,
        metric["purpose"],
        ha="left",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.text(
        0.06,
        0.918,
        f"{spec['flow']}; test rows fixed. {seed_note}.",
        ha="left",
        fontsize=10,
        color=SECONDARY,
    )
    fig.text(
        0.5,
        0.055,
        "Labelled training meters in source sites (log scale)",
        ha="center",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.text(
        0.014,
        0.48,
        metric["ylabel"],
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
        bbox_to_anchor=(0.5, 0.006),
        ncol=len(MODEL_ORDER),
        frameon=False,
        fontsize=9.5,
        labelcolor=SECONDARY,
    )

    # Panels land at ~3.2 x 3.5 in: the y span runs slightly longer than the x span.
    fig.subplots_adjust(
        left=0.095, right=0.975, top=0.848, bottom=0.105, hspace=0.25, wspace=0.07
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"m6_training_meter_curve_{spec['slug']}_{metric['token']}.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    return path


def render_site_figure(site: int, out_dir: Path, xlim: tuple[float, float]) -> Path:
    """One site, three metrics side by side."""
    direction, flow = SITE_DIRECTION[site]
    cells = _load_cells(direction)
    seeds_done = _complete_seeds(cells)
    if not seeds_done:
        raise SystemExit(f"no seed has all budgets complete for {direction}")
    all_meters = sorted(
        {_meter_count(cells[(b, s)]) for b in BUDGETS for s in seeds_done}
    )
    rows, prevalence = _site_support(cells, site)

    # Three panels across, near-square each; diagonal stays on the style baseline.
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 6.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    spread = 0.0
    for idx, (ax, metric) in enumerate(zip(axes, METRICS, strict=True)):
        _style_axis(ax)
        for model in MODEL_ORDER:
            style = MODEL_STYLE[model]
            x, mean, lo, hi, _ = _series(cells, site, model, seeds_done, metric["path"])
            if x.size == 0:
                continue
            spread = max(spread, float((hi - lo).max()))
            ax.plot(
                x,
                mean,
                color=style["color"],
                marker=style["marker"],
                markersize=4.2,
                linewidth=1.35 if model == "ensemble" else 1.0,
                zorder=3,
            )
        ax.set_title(
            metric["panel"],
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
            pad=12,
        )
        ax.set_xscale("log")
        ax.set_xticks(all_meters)
        ax.set_xticklabels(
            [f"{m:,}\n(all)" if m == all_meters[-1] else f"{m:,}" for m in all_meters]
        )
        ax.minorticks_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, 1.04)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if idx == 0:
            ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])

    fig.suptitle(
        f"Training-meter curves for {site_label(site, with_id=True)}",
        x=0.055,
        y=0.960,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.893,
        "Whether labelling more source meters improves this unseen site's ranking and its alarms.",
        ha="left",
        fontsize=10.5,
        color=SECONDARY,
    )
    seed_note = (
        f"Mean of {len(seeds_done)} seeds, widest seed gap {spread:.2f}"
        if len(seeds_done) > 1
        else "Single seed"
    )
    fig.text(
        0.055,
        0.856,
        f"{flow}; test rows fixed. {prevalence:.2%} anomalies over {rows:,} rows. {seed_note}.",
        ha="left",
        fontsize=10,
        color=SECONDARY,
    )
    fig.text(
        0.5,
        0.075,
        "Labelled training meters in source sites (log scale)",
        ha="center",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.text(
        0.012,
        0.47,
        "Score on held-out site",
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
        bbox_to_anchor=(0.5, 0.008),
        ncol=len(MODEL_ORDER),
        frameon=False,
        fontsize=9.5,
        labelcolor=SECONDARY,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.766, bottom=0.169, wspace=0.07)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"m6_training_meter_curve_site_{site}_pr_auc_f1_recall.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs" / "reports" / "assets" / "m6",
        help="directory for rendered PNGs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xlim = global_xlim()
    for metric in METRICS:
        for spec in FIGURES:
            path = render_figure(spec, metric, args.out_dir, xlim)
            print(f"wrote {path.relative_to(ROOT)}")
    for site in sorted(SITE_DIRECTION):
        path = render_site_figure(site, args.out_dir, xlim)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
