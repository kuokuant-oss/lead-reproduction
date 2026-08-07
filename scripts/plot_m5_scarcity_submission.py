"""Render the M5 scarcity figures and compact LaTeX tables for submission.

The row-scarcity arm is parsed from the authoritative matched-context report.
The building-scarcity arm is read from its aggregate metrics and frozen seed-42
manifest.  Building budgets are included only when Tree Ensemble and TabPFN
both have all four meter-level cells, so a partially completed K=100 run can
never enter a paired figure or table.

Style contract: docs/reference/plot-style-rules.md v0.3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Iterable

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
TABPFN = "#d1498b"
BAR_TREE = "#8ecae6"
BAR_TREE_EDGE = "#397a96"
BAR_TABPFN = "#f4a261"
BAR_TABPFN_EDGE = "#a95d1d"
BUDGET_COLORS = ("#443983", "#31688e", "#21918c", "#35b779", "#90d743")

MODEL_STYLE = {
    "ensemble": {
        "label": "Tree Ensemble",
        "color": INK,
        "marker": "p",
        "linewidth": 1.62,
        "markersize": 5.4,
    },
    "tabpfn": {
        "label": "TabPFN",
        "color": TABPFN,
        "marker": "X",
        "linewidth": 1.2,
        "markersize": 5.2,
    },
}
MODEL_ORDER = ("ensemble", "tabpfn")
METER_ORDER = ("electricity", "chilledwater", "steam", "hotwater")
PANEL_METERS = METER_ORDER
METER_LABEL = {
    "electricity": "Electricity",
    "chilledwater": "Chilled water",
    "steam": "Steam",
    "hotwater": "Hot water",
}

# Reconstructed and checked from the frozen seed-42 nested balanced indices.
# These are source-building coverage counts, not holdout-building counts.
MATCHED_SOURCE_BUILDINGS = {
    5_000: 703,
    10_000: 724,
    20_000: 725,
    50_000: 725,
    100_000: 725,
}

MATCHED_SECTION = {
    "tabpfn": "TabPFN by meter: 137 features, PR-AUC",
    "ensemble": "Trees by meter: 137 features, PR-AUC",
}


def _format_truncated(value: float, places: int = 3, signed: bool = False) -> str:
    """Format a value by truncating toward zero at a fixed decimal precision."""
    quantum = Decimal(1).scaleb(-places)
    truncated = Decimal(str(value)).quantize(quantum, rounding=ROUND_DOWN)
    if truncated == 0:
        truncated = abs(truncated)
    spec = f"+.{places}f" if signed else f".{places}f"
    return format(truncated, spec)


def _latex_document(table: str) -> str:
    """Wrap one generated table in a directly compilable LaTeX document."""
    preamble = r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage[paperwidth=8.5in,paperheight=11in,margin=0.55in]{geometry}
\usepackage{booktabs,tabularx,siunitx}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[active,tightpage]{preview}
\setlength\PreviewBorder{10pt}
\sisetup{detect-all}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.12}
\pagestyle{empty}
\begin{document}
"""
    body = table.strip()
    body = body.replace(
        r"\begin{table}[t]", r"\begin{preview}\begin{minipage}{\linewidth}"
    )
    body = body.replace(r"\caption{", r"\captionof{table}{")
    body = body.replace(r"\end{table}", r"\end{minipage}\end{preview}")
    return preamble + body + "\n\\end{document}\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_figure(fig: plt.Figure, stem: Path, facecolor: str = SURFACE) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, kwargs in (
        ("png", {"dpi": 180}),
        ("pdf", {}),
        ("svg", {}),
    ):
        out = stem.with_suffix(f".{suffix}")
        tmp = out.with_name(f".{out.name}.tmp")
        fig.savefig(
            tmp,
            format=suffix,
            facecolor=facecolor,
            edgecolor="none",
            **kwargs,
        )
        tmp.replace(out)
        outputs.append(out)
    plt.close(fig)
    return outputs


def _markdown_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_markdown_table(
    report: str, section: str
) -> tuple[list[str], list[list[str]]]:
    marker = f"### {section}"
    try:
        tail = report[report.index(marker) + len(marker) :]
    except ValueError as exc:
        raise ValueError(f"missing report section: {section}") from exc

    lines = tail.splitlines()
    table_start = next(
        (i for i, line in enumerate(lines) if line.startswith("| group |")), None
    )
    if table_start is None:
        raise ValueError(f"missing table under section: {section}")

    header = _markdown_cells(lines[table_start])
    rows: list[list[str]] = []
    for line in lines[table_start + 2 :]:
        if not line.startswith("|"):
            break
        cells = _markdown_cells(line)
        if len(cells) != len(header):
            raise ValueError(f"malformed row under section {section}: {line}")
        rows.append(cells)
    if not rows:
        raise ValueError(f"empty table under section: {section}")
    return header, rows


def load_matched_meter_metrics(path: Path) -> list[dict[str, Any]]:
    report = path.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    for model, section in MATCHED_SECTION.items():
        header, rows = _parse_markdown_table(report, section)
        budgets = [
            int(value.replace(",", "")) for value in header[3:] if value != "FULL"
        ]
        for cells in rows:
            meter = cells[0]
            if meter not in METER_ORDER:
                continue
            for offset, budget in enumerate(budgets, start=3):
                records.append(
                    {
                        "experiment": "matched_context",
                        "budget": budget,
                        "source_buildings": MATCHED_SOURCE_BUILDINGS[budget],
                        "source_rows": budget,
                        "model": model,
                        "meter": meter,
                        "test_rows": int(cells[1].replace(",", "")),
                        "test_anomalies": int(cells[2].replace(",", "")),
                        "pr_auc": float(cells[offset]),
                    }
                )
    expected = len(MODEL_ORDER) * len(METER_ORDER) * len(MATCHED_SOURCE_BUILDINGS)
    if len(records) != expected:
        raise ValueError(
            f"expected {expected} matched meter records, found {len(records)}"
        )
    return records


def load_building_manifest(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = payload.get("cells", {})
    parsed: dict[int, dict[str, Any]] = {}
    for raw_budget, cell in cells.items():
        budget = int(raw_budget)
        buildings = cell.get("available_buildings", [])
        if len(buildings) != budget:
            raise ValueError(f"manifest K={budget} has {len(buildings)} buildings")
        parsed[budget] = {
            "source_buildings": budget,
            "source_rows": int(cell["available_rows"]),
            "source_anomalies": int(cell["available_anomalies"]),
            "source_anomaly_rate": float(cell["available_anomaly_rate"]),
            "source_meter_counts": {
                str(k): int(v) for k, v in cell["available_meter_counts"].items()
            },
        }
    return parsed


def load_building_meter_metrics(
    metrics_path: Path,
    manifest: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    candidates: list[dict[str, Any]] = []
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["sampling_profile"] != "representative"
                or row["features"] != "137"
                or row["grouping"] != "meter"
                or row["model"] not in MODEL_ORDER
            ):
                continue
            budget = int(row["building_budget"])
            if budget not in manifest:
                continue
            meter = row["group_label"].strip().lower().replace(" ", "")
            if meter not in METER_ORDER or not row["pr_auc"]:
                continue
            candidates.append(
                {
                    "experiment": "building_count",
                    "budget": budget,
                    **manifest[budget],
                    "model": row["model"],
                    "meter": meter,
                    "test_rows": int(float(row["rows"])),
                    "test_anomalies": int(float(row["anomalies"])),
                    "pr_auc": float(row["pr_auc"]),
                }
            )

    cell_members: dict[tuple[int, str], set[str]] = defaultdict(set)
    for record in candidates:
        cell_members[(record["budget"], record["model"])].add(record["meter"])
    complete = {
        budget
        for budget in manifest
        if all(
            cell_members[(budget, model)] == set(METER_ORDER) for model in MODEL_ORDER
        )
    }
    paired_budgets = sorted(complete)
    records = [record for record in candidates if record["budget"] in complete]
    expected = len(paired_budgets) * len(MODEL_ORDER) * len(METER_ORDER)
    if len(records) != expected:
        raise ValueError(
            f"expected {expected} paired building records, found {len(records)}"
        )
    if not records:
        raise ValueError("no completed Tree Ensemble / TabPFN building-budget pairs")
    return records, paired_budgets


def macro_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    exemplars: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in records:
        key = (record["experiment"], record["budget"], record["model"])
        buckets[key].append(record["pr_auc"])
        exemplars[key] = record
    output: list[dict[str, Any]] = []
    for key, values in sorted(buckets.items()):
        if len(values) != len(METER_ORDER):
            raise ValueError(f"macro cell {key} has {len(values)} meters")
        exemplar = exemplars[key]
        output.append(
            {
                "experiment": key[0],
                "budget": key[1],
                "model": key[2],
                "source_buildings": exemplar["source_buildings"],
                "source_rows": exemplar["source_rows"],
                "pr_auc_macro": sum(values) / len(values),
            }
        )
    return output


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _model_handles() -> list[plt.Line2D]:
    return [
        plt.Line2D(
            [],
            [],
            color=MODEL_STYLE[model]["color"],
            marker=MODEL_STYLE[model]["marker"],
            linewidth=MODEL_STYLE[model]["linewidth"],
            markersize=MODEL_STYLE[model]["markersize"],
            label=MODEL_STYLE[model]["label"],
        )
        for model in MODEL_ORDER
    ]


def _plot_model_line(ax: plt.Axes, x: list[int], y: list[float], model: str) -> None:
    style = MODEL_STYLE[model]
    ax.plot(
        x,
        y,
        color=style["color"],
        marker=style["marker"],
        linewidth=style["linewidth"],
        markersize=style["markersize"],
        markeredgewidth=0.8,
        zorder=3,
    )


def _figure_heading(
    fig: plt.Figure, title: str, subtitle: str, title_y: float = 0.965
) -> None:
    fig.suptitle(
        title,
        x=0.065,
        y=title_y,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(0.065, title_y - 0.065, subtitle, ha="left", fontsize=9.5, color=SECONDARY)


def render_design(
    matched: list[dict[str, Any]],
    manifest: dict[int, dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    del matched  # Counts are validated constants; metrics are not needed here.
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    fig.patch.set_facecolor(SURFACE)
    _style_axis(ax)

    matched_x = list(MATCHED_SOURCE_BUILDINGS)
    matched_y = [MATCHED_SOURCE_BUILDINGS[x] for x in matched_x]
    building_x = [manifest[k]["source_rows"] for k in sorted(manifest)]
    building_y = sorted(manifest)
    ax.plot(
        matched_x,
        matched_y,
        color=INK,
        marker="o",
        linewidth=1.5,
        markersize=5,
        label="Row scarcity: matched context",
        zorder=3,
    )
    ax.plot(
        building_x,
        building_y,
        color=SECONDARY,
        marker="s",
        linestyle="--",
        linewidth=1.2,
        markersize=4.8,
        label="Building scarcity: representative ladder",
        zorder=3,
    )
    for x, y in zip(matched_x, matched_y, strict=True):
        ax.annotate(
            f"{y}",
            (x, y),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK,
        )
    for x, y in zip(building_x, building_y, strict=True):
        ax.annotate(
            f"K={y}",
            (x, y),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=SECONDARY,
        )

    ax.set_xscale("log")
    # Keep the 20k matched point at its true x position, but omit its tick: a
    # 20k/25k pair is illegible at journal-column width.
    ticks = [5_000, 10_000, 25_000, 50_000, 100_000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick // 1000}k" for tick in ticks])
    ax.minorticks_off()
    ax.set_xlim(4_300, 118_000)
    ax.set_ylim(0, 770)
    ax.set_yticks([0, 100, 300, 500, 700])
    ax.set_xlabel(
        "Labeled training rows (log scale)", fontsize=9.5, color=SECONDARY, labelpad=8
    )
    ax.set_ylabel(
        "Distinct source buildings", fontsize=9.5, color=SECONDARY, labelpad=8
    )
    _figure_heading(
        fig,
        "Training-data coverage under two distinct protocols",
        "Descriptive scope only; the two protocols do not define matched experimental cells.",
    )
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=8.8,
        labelcolor=SECONDARY,
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.79, bottom=0.20)
    return _atomic_figure(fig, out_dir / "m5_cross_experiment_design_source_buildings")


def render_macro(
    matched_macro: list[dict[str, Any]],
    building_macro: list[dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 5.3), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    panels = (
        (
            axes[0],
            matched_macro,
            "Experiment A — matched context",
            "Matched context rows, N",
            "rows",
        ),
        (
            axes[1],
            building_macro,
            "Experiment B — building ladder",
            "Source buildings (K)",
            "buildings",
        ),
    )
    for panel_index, (ax, records, title, xlabel, x_field) in enumerate(panels):
        _style_axis(ax)
        for model in MODEL_ORDER:
            cells = sorted(
                (r for r in records if r["model"] == model), key=lambda r: r["budget"]
            )
            x = [
                r["source_rows"] if x_field == "rows" else r["source_buildings"]
                for r in cells
            ]
            y = [r["pr_auc_macro"] for r in cells]
            _plot_model_line(ax, x, y, model)
        ax.set_xscale("log")
        ticks = sorted(
            {
                r["source_rows"] if x_field == "rows" else r["source_buildings"]
                for r in records
            }
        )
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [f"{tick // 1000}k" if x_field == "rows" else str(tick) for tick in ticks]
        )
        ax.minorticks_off()
        ax.set_ylim(0.50, 0.91)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set_xlabel(
            f"{xlabel} (log scale)", fontsize=9.5, color=SECONDARY, labelpad=8
        )
        ax.set_title(
            title, loc="left", fontsize=11.2, fontweight="bold", color=INK, pad=10
        )
        if panel_index == 0:
            ax.set_ylabel(
                "Macro PR-AUC across meter types",
                fontsize=9.5,
                color=SECONDARY,
                labelpad=8,
            )

    _figure_heading(
        fig,
        "Performance curves under two different scarcity axes",
        "Read panels separately: protocols and x-axes differ; compare trends, not point-to-point budgets.",
    )
    fig.legend(
        handles=_model_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=SECONDARY,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.78, bottom=0.19, wspace=0.07)
    return _atomic_figure(fig, out_dir / "m5_cross_experiment_macro_pr_auc")


def render_single_macro(
    records: list[dict[str, Any]],
    experiment: str,
    out_dir: Path,
) -> list[Path]:
    is_matched = experiment == "matched_context"
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    fig.patch.set_facecolor(SURFACE)
    _style_axis(ax)
    for model in MODEL_ORDER:
        cells = sorted(
            (record for record in records if record["model"] == model),
            key=lambda record: record["budget"],
        )
        x = [
            record["source_rows"] if is_matched else record["source_buildings"]
            for record in cells
        ]
        y = [record["pr_auc_macro"] for record in cells]
        _plot_model_line(ax, x, y, model)

    ticks = sorted(
        {
            record["source_rows"] if is_matched else record["source_buildings"]
            for record in records
        }
    )
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{tick // 1000}k" if is_matched else str(tick) for tick in ticks]
    )
    ax.minorticks_off()
    ax.set_ylim(0.50, 0.91)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylabel(
        "Macro PR-AUC across meter types", fontsize=9.5, color=SECONDARY, labelpad=8
    )
    if is_matched:
        title = "Experiment A — matched-context row scarcity"
        subtitle = "Only labeled context size N varies · nested 50/50 contexts · near-full building coverage."
        xlabel = "Matched context rows, N (log scale)"
        stem = "m5_exp_a_matched_context_macro_pr_auc"
    else:
        title = "Experiment B — source-building scarcity"
        subtitle = "Source-building count K varies · about 500 rows/building · natural training class mix."
        xlabel = "Source buildings, K (log scale)"
        stem = "m5_exp_b_building_count_macro_pr_auc"
    ax.set_xlabel(xlabel, fontsize=9.5, color=SECONDARY, labelpad=8)
    _figure_heading(fig, title, subtitle)
    fig.legend(
        handles=_model_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=SECONDARY,
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.76, bottom=0.20)
    return _atomic_figure(fig, out_dir / stem)


def render_meter_panels(
    records: list[dict[str, Any]],
    experiment: str,
    out_dir: Path,
) -> list[Path]:
    is_matched = experiment == "matched_context"
    subset = [
        record
        for record in records
        if record["experiment"] == experiment and record["meter"] in PANEL_METERS
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.4, 4.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for index, (ax, meter) in enumerate(zip(axes, PANEL_METERS, strict=True)):
        _style_axis(ax)
        meter_rows = [record for record in subset if record["meter"] == meter]
        for model in MODEL_ORDER:
            cells = sorted(
                (r for r in meter_rows if r["model"] == model),
                key=lambda r: r["budget"],
            )
            x = [
                r["source_rows"] if is_matched else r["source_buildings"] for r in cells
            ]
            y = [r["pr_auc"] for r in cells]
            _plot_model_line(ax, x, y, model)
        ticks = sorted(
            {
                r["source_rows"] if is_matched else r["source_buildings"]
                for r in meter_rows
            }
        )
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{x // 1000}k" if is_matched else str(x) for x in ticks])
        ax.minorticks_off()
        ax.set_ylim((0.55, 1.01) if is_matched else (0.25, 1.01))
        ax.set_title(
            METER_LABEL[meter],
            loc="left",
            fontsize=10.8,
            fontweight="bold",
            color=INK,
            pad=9,
        )
        ax.set_xlabel(
            "Context rows (log scale)"
            if is_matched
            else "Source buildings (K, log scale)",
            fontsize=8.8,
            color=SECONDARY,
            labelpad=7,
        )
        if index == 0:
            ax.set_ylabel(
                "PR-AUC on the odd-building holdout",
                fontsize=9.2,
                color=SECONDARY,
                labelpad=7,
            )

    if is_matched:
        title = "Experiment A — meter-level PR-AUC across context size"
        subtitle = "Matched-context row scarcity · 50/50 training contexts · 137 features · seed 42."
        stem = "m5_exp_a_matched_context_all_meters_pr_auc"
    else:
        title = "Experiment B — meter-level PR-AUC across building count"
        subtitle = "Representative building ladder · natural training class mix · 137 features · seed 42."
        stem = "m5_exp_b_building_count_all_meters_pr_auc"
    _figure_heading(fig, title, subtitle)
    fig.legend(
        handles=_model_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=SECONDARY,
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.75, bottom=0.21, wspace=0.08)
    return _atomic_figure(fig, out_dir / stem)


def _selected_detail_budgets(
    experiment: str, records: list[dict[str, Any]]
) -> list[int]:
    del experiment
    return sorted({record["budget"] for record in records})


def render_meter_grouped_bars(
    records: list[dict[str, Any]],
    experiment: str,
    out_dir: Path,
) -> list[Path]:
    """Compare both models across all four meters at every available budget."""
    from matplotlib.patches import Patch

    budgets = _selected_detail_budgets(experiment, records)
    is_matched = experiment == "matched_context"
    figure_size = (17.5, 5.2) if is_matched else (14.0, 5.1)
    fig, axes_grid = plt.subplots(1, len(budgets), figsize=figure_size, sharey=True)
    active_axes = list(axes_grid) if len(budgets) > 1 else [axes_grid]
    fig.patch.set_facecolor("#ffffff")
    positions = list(range(len(METER_ORDER)))
    width = 0.26
    offset = 0.20
    lookup = {
        (record["budget"], record["model"], record["meter"]): record["pr_auc"]
        for record in records
        if record["experiment"] == experiment
    }
    for panel_index, (ax, budget) in enumerate(zip(active_axes, budgets, strict=True)):
        _style_axis(ax)
        ax.set_facecolor("#ffffff")
        if panel_index != 0:
            ax.tick_params(labelleft=False)
        tree_values = [lookup[(budget, "ensemble", meter)] for meter in METER_ORDER]
        tabpfn_values = [lookup[(budget, "tabpfn", meter)] for meter in METER_ORDER]
        tree_bars = ax.bar(
            [position - offset for position in positions],
            tree_values,
            width,
            color=BAR_TREE,
            edgecolor=BAR_TREE_EDGE,
            linewidth=0.7,
            zorder=3,
        )
        tabpfn_bars = ax.bar(
            [position + offset for position in positions],
            tabpfn_values,
            width,
            color=BAR_TABPFN,
            edgecolor=BAR_TABPFN_EDGE,
            linewidth=0.7,
            zorder=3,
        )
        ax.bar_label(
            tree_bars,
            labels=[_format_truncated(value) for value in tree_values],
            padding=2,
            fontsize=7.0,
            color=INK,
        )
        ax.bar_label(
            tabpfn_bars,
            labels=[_format_truncated(value) for value in tabpfn_values],
            padding=2,
            fontsize=7.0,
            color=INK,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(["Electricity", "Chilled\nwater", "Steam", "Hot\nwater"])
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        panel_title = (
            f"N={budget // 1000}k rows" if is_matched else f"K={budget} buildings"
        )
        ax.set_title(
            panel_title,
            loc="left",
            fontsize=11.2,
            fontweight="bold",
            color=INK,
            pad=9,
        )

    if is_matched:
        title = "Experiment A — meter comparison across row budgets"
        subtitle = "All five context budgets and all four meter types; bars show Tree Ensemble and TabPFN."
        stem = "m5_exp_a_matched_context_meter_grouped_pr_auc"
        _figure_heading(fig, title, subtitle)
        legend_y = 0.005
        fig.subplots_adjust(left=0.055, right=0.995, top=0.75, bottom=0.20, wspace=0.10)
    else:
        title = "Experiment B — meter comparison by building budget"
        subtitle = (
            "All four meter types for paired K=10, K=20, K=50, and K=100 results."
        )
        stem = "m5_exp_b_building_count_meter_grouped_pr_auc"
        _figure_heading(fig, title, subtitle)
        legend_y = 0.005
        fig.subplots_adjust(left=0.07, right=0.995, top=0.75, bottom=0.20, wspace=0.08)
    fig.text(
        0.018,
        0.48,
        "PR-AUC on the odd-building holdout",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9.2,
        color=SECONDARY,
    )
    fig.legend(
        handles=[
            Patch(
                facecolor=BAR_TREE,
                edgecolor=BAR_TREE_EDGE,
                label="Tree Ensemble",
            ),
            Patch(
                facecolor=BAR_TABPFN,
                edgecolor=BAR_TABPFN_EDGE,
                label="TabPFN",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=SECONDARY,
    )
    return _atomic_figure(fig, out_dir / stem, facecolor="#ffffff")


def _budget_palette(count: int) -> list[str]:
    if count < 1 or count > len(BUDGET_COLORS):
        raise ValueError(f"unsupported budget count: {count}")
    if count == 1:
        return [BUDGET_COLORS[len(BUDGET_COLORS) // 2]]
    indices = [
        round(index * (len(BUDGET_COLORS) - 1) / (count - 1)) for index in range(count)
    ]
    return [BUDGET_COLORS[index] for index in indices]


def render_meter_budget_blocks(
    records: list[dict[str, Any]],
    experiment: str,
    out_dir: Path,
) -> list[Path]:
    """Facet by meter, split models spatially, and encode budgets by color."""
    from matplotlib.patches import Patch

    budgets = _selected_detail_budgets(experiment, records)
    colors = _budget_palette(len(budgets))
    is_matched = experiment == "matched_context"
    fig, axes = plt.subplots(1, 4, figsize=(21.0, 5.2), sharey=True)
    fig.patch.set_facecolor("#ffffff")
    group_centers = (0.0, 1.35)
    bar_width = 0.10 if len(budgets) >= 5 else 0.14
    bar_gap = 0.0605 if len(budgets) >= 5 else 0.066
    total_width = len(budgets) * bar_width + (len(budgets) - 1) * bar_gap
    offsets = [
        index * (bar_width + bar_gap) - (total_width - bar_width) / 2
        for index in range(len(budgets))
    ]
    y_min = 0.60 if is_matched else 0.28
    y_ticks = (
        [0.6, 0.7, 0.8, 0.9, 1.0]
        if is_matched
        else [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    lookup = {
        (record["budget"], record["model"], record["meter"]): record["pr_auc"]
        for record in records
        if record["experiment"] == experiment
    }

    for panel_index, (ax, meter) in enumerate(zip(axes, METER_ORDER, strict=True)):
        _style_axis(ax)
        ax.set_facecolor("#ffffff")
        if panel_index != 0:
            ax.tick_params(labelleft=False)
        for model_index, model in enumerate(MODEL_ORDER):
            values = [lookup[(budget, model, meter)] for budget in budgets]
            positions = [group_centers[model_index] + offset for offset in offsets]
            bars = ax.bar(
                positions,
                values,
                bar_width,
                color=colors,
                edgecolor="none",
                zorder=3,
            )
            ax.bar_label(
                bars,
                labels=[_format_truncated(value) for value in values],
                padding=2,
                fontsize=6.5,
                color=INK,
            )
        ax.set_xticks(group_centers)
        ax.set_xticklabels(["Tree Ensemble", "TabPFN"])
        ax.set_xlim(-0.65, 2.0)
        ax.set_ylim(y_min, 1.01)
        ax.set_yticks(y_ticks)
        ax.set_title(
            METER_LABEL[meter],
            loc="left",
            fontsize=11.2,
            fontweight="bold",
            color=INK,
            pad=9,
        )

    if is_matched:
        title = "Experiment A — budget profiles within each meter"
        labels = [f"N={budget // 1000}k" for budget in budgets]
        stem = "m5_exp_a_matched_context_meter_budget_blocks_pr_auc"
    else:
        title = "Experiment B — budget profiles within each meter"
        labels = [f"K={budget}" for budget in budgets]
        stem = "m5_exp_b_building_count_meter_budget_blocks_pr_auc"
    fig.suptitle(
        title,
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.018,
        0.48,
        "PR-AUC on the odd-building holdout",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9.2,
        color=SECONDARY,
    )
    fig.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=label)
            for color, label in zip(colors, labels, strict=True)
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(budgets),
        frameon=False,
        fontsize=9,
        labelcolor=SECONDARY,
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.82, bottom=0.20, wspace=0.08)
    return _atomic_figure(fig, out_dir / stem, facecolor="#ffffff")


def render_separate_setup_table(
    experiment: str,
    out_path: Path,
) -> None:
    is_matched = experiment == "matched_context"
    common = [
        ("Data and models", "Features", "137 engineered features"),
        ("Data and models", "Models", "Tree Ensemble; TabPFN (8 estimators)"),
        ("Evaluation", "Holdout", "Odd-ID buildings; 724 buildings; 10,137,155 rows"),
        ("Evaluation", "Metric", "PR-AUC by meter; equal-weight four-meter macro"),
        ("Reproducibility", "Seed and draws", "42; one draw"),
    ]
    if is_matched:
        caption = "Experiment A protocol: matched-context row scarcity."
        label = "tab:m5-exp-a-setup"
        rows = [
            ("Design", "Scarcity variable", "Labeled context rows, N"),
            ("Design", "Budget levels", "N=5k, 10k, 20k, 50k, 100k"),
            ("Design", "Context construction", "Nested stratified row samples"),
            ("Design", "Class ratio", "Positive:negative = 1:1"),
            ("Design", "Building coverage", "703, 724, 725, 725, 725 source buildings"),
            (
                "Design",
                "Meter composition",
                "All meter types; frequencies inherited from sampled rows",
            ),
            *common,
        ]
    else:
        caption = "Experiment B protocol: representative source-building scarcity."
        label = "tab:m5-exp-b-setup"
        rows = [
            ("Design", "Scarcity variable", "Distinct source buildings, K"),
            ("Design", "Budget levels", "K=10, 20, 50, 100"),
            (
                "Design",
                "Row allocation",
                "Approximately 500 rows/building; maximum 50k rows",
            ),
            ("Design", "Building sampling", "Representative nested building ladder"),
            (
                "Design",
                "Anomaly rate",
                "Natural anomaly rate retained from selected buildings",
            ),
            ("Design", "Meter composition", "All meter types in selected buildings"),
            *common,
        ]

    latex_rows: list[str] = []
    previous_section: str | None = None
    for section, setting, value in rows:
        if previous_section is not None and section != previous_section:
            latex_rows.append(r"\addlinespace[2pt]")
        section_cell = section if section != previous_section else ""
        latex_rows.append(f"{section_cell} & {setting} & {value} \\\\")
        previous_section = section
    body = "\n".join(latex_rows)
    latex = rf"""\begin{{table}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
\begin{{tabularx}}{{\linewidth}}{{p{{0.17\linewidth}}p{{0.23\linewidth}}X}}
\toprule
Section & Setting & Value \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\end{{table}}
"""
    _atomic_text(out_path, _latex_document(latex))


def render_exp_a_result_table(
    matched_macro: list[dict[str, Any]],
    out_path: Path,
) -> None:
    lookup = {(row["budget"], row["model"]): row for row in matched_macro}
    rows: list[str] = []
    for budget in sorted({row["budget"] for row in matched_macro}):
        tree = lookup[(budget, "ensemble")]
        tabpfn = lookup[(budget, "tabpfn")]
        tree_value = _format_truncated(tree["pr_auc_macro"])
        tabpfn_value = _format_truncated(tabpfn["pr_auc_macro"])
        delta = _format_truncated(
            tabpfn["pr_auc_macro"] - tree["pr_auc_macro"], signed=True
        )
        rows.append(
            f"N={budget // 1000}k & {tree['source_rows']} & {tree['source_buildings']} "
            f"& {tree_value} & {tabpfn_value} & {delta} \\\\"
        )
    body = "\n".join(rows)
    latex = rf"""\begin{{table}}[t]
\centering
\caption{{Experiment A macro PR-AUC across matched-context row budgets.}}
\label{{tab:m5-exp-a-results}}
\begin{{tabular}}{{lS[table-format=6.0]S[table-format=3.0]S[table-format=1.3]S[table-format=1.3]S[table-format=+1.3]}}
\toprule
Budget & {{Labeled rows}} & {{Source buildings}} & {{Tree Ensemble}} & {{TabPFN}} & {{$\Delta$}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    _atomic_text(out_path, _latex_document(latex))


def render_exp_b_result_table(
    building_macro: list[dict[str, Any]],
    manifest: dict[int, dict[str, Any]],
    out_path: Path,
) -> None:
    lookup = {(row["budget"], row["model"]): row for row in building_macro}
    rows: list[str] = []
    for budget in sorted({row["budget"] for row in building_macro}):
        tree = lookup[(budget, "ensemble")]
        tabpfn = lookup[(budget, "tabpfn")]
        anomaly_rate = 100 * manifest[budget]["source_anomaly_rate"]
        tree_value = _format_truncated(tree["pr_auc_macro"])
        tabpfn_value = _format_truncated(tabpfn["pr_auc_macro"])
        delta = _format_truncated(
            tabpfn["pr_auc_macro"] - tree["pr_auc_macro"], signed=True
        )
        rows.append(
            f"K={budget} & {tree['source_rows']} & {anomaly_rate:.2f} "
            f"& {tree_value} & {tabpfn_value} & {delta} \\\\"
        )
    body = "\n".join(rows)
    latex = rf"""\begin{{table}}[t]
\centering
\caption{{Experiment B macro PR-AUC across source-building budgets.}}
\label{{tab:m5-exp-b-results}}
\begin{{tabular}}{{lS[table-format=5.0]S[table-format=2.2]S[table-format=1.3]S[table-format=1.3]S[table-format=+1.3]}}
\toprule
Budget & {{Labeled rows}} & {{Training anomaly rate (\%)}} & {{Tree Ensemble}} & {{TabPFN}} & {{$\Delta$}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    _atomic_text(out_path, _latex_document(latex))


def render_meter_detail_table(
    records: list[dict[str, Any]],
    macro: list[dict[str, Any]],
    experiment: str,
    out_path: Path,
) -> None:
    """Render a journal-style multi-level table for every available budget."""
    budgets = _selected_detail_budgets(experiment, records)
    values = {
        (record["budget"], record["model"], record["meter"]): record["pr_auc"]
        for record in records
        if record["experiment"] == experiment
    }
    macro_values = {
        (record["budget"], record["model"]): record["pr_auc_macro"]
        for record in macro
        if record["experiment"] == experiment
    }
    rows: list[str] = []
    for meter in METER_ORDER:
        cells = [
            _format_truncated(values[(budget, model, meter)])
            for budget in budgets
            for model in MODEL_ORDER
        ]
        rows.append(f"{METER_LABEL[meter]} & " + " & ".join(cells) + r" \\")
    macro_cells = [
        _format_truncated(macro_values[(budget, model)])
        for budget in budgets
        for model in MODEL_ORDER
    ]
    rows.append(r"\midrule")
    rows.append(r"\textit{Macro} & " + " & ".join(macro_cells) + r" \\")

    if experiment == "matched_context":
        caption = (
            "Experiment A meter-level PR-AUC across all matched-context row budgets."
        )
        label = "tab:m5-exp-a-meter-detail"
        budget_labels = [f"N={budget // 1000}k" for budget in budgets]
    else:
        caption = (
            "Experiment B meter-level PR-AUC across paired source-building budgets."
        )
        label = "tab:m5-exp-b-meter-detail"
        budget_labels = [f"K={budget}" for budget in budgets]

    group_header = "Meter"
    sub_header = ""
    cmidrules: list[str] = []
    for index, budget_label in enumerate(budget_labels):
        first = 2 + 2 * index
        group_header += rf" & \multicolumn{{2}}{{c}}{{{budget_label}}}"
        sub_header += r" & {Tree} & {TabPFN}"
        cmidrules.append(rf"\cmidrule(lr){{{first}-{first + 1}}}")
    font_command = r"\scriptsize" if len(budgets) > 3 else r"\small"
    column_padding = "3.0pt" if len(budgets) > 3 else "4.5pt"
    latex = rf"""\begin{{table}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
{font_command}
\setlength{{\tabcolsep}}{{{column_padding}}}
\begin{{tabular}}{{l*{{{2 * len(budgets)}}}{{S[table-format=1.3]}}}}
\toprule
{group_header} \\
{"".join(cmidrules)}
{sub_header} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    _atomic_text(out_path, _latex_document(latex))


def write_provenance(
    path: Path,
    matched_path: Path,
    metrics_path: Path,
    manifest_path: Path,
    matched: list[dict[str, Any]],
    building: list[dict[str, Any]],
    paired_budgets: list[int],
) -> None:
    payload = {
        "schema_version": 1,
        "metric": "pr_auc",
        "features": 137,
        "seed": 42,
        "aggregation": "equal_weight_macro_across_four_meter_types",
        "paired_building_budgets": paired_budgets,
        "incomplete_building_budgets_excluded": sorted(
            set(load_building_manifest(manifest_path)) - set(paired_budgets)
        ),
        "matched_source_buildings": MATCHED_SOURCE_BUILDINGS,
        "sources": {
            str(matched_path.relative_to(ROOT)): _sha256(matched_path),
            str(metrics_path.relative_to(ROOT)): _sha256(metrics_path),
            str(manifest_path.relative_to(ROOT)): _sha256(manifest_path),
        },
        "records": {
            "matched_meter": matched,
            "building_meter_paired": building,
            "matched_macro": macro_records(matched),
            "building_macro_paired": macro_records(building),
        },
    }
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_draft(path: Path, paired_budgets: list[int]) -> None:
    paired = ", ".join(f"K={budget}" for budget in paired_budgets)
    text = f"""# M5 labeled-data scarcity experiments (submission draft)

This package contains two distinct experiments. They share the odd-building holdout, 137-feature representation, models, and reporting metric, but they implement different scarcity interventions. Each experiment is therefore presented completely before any cross-experiment synthesis.

## Experiment A — matched-context row scarcity

### Objective

Measure sensitivity to the number of labeled context rows, N, while source-building coverage remains nearly complete. Training contexts are nested with a 1:1 class ratio; evaluation uses the natural holdout distribution.

### Protocol

![Experiment A protocol](tables/m5-scarcity/m5_exp_a_matched_context_setup.png)

### Aggregate performance

![Experiment A macro PR-AUC](assets/m5-scarcity/m5_exp_a_matched_context_macro_pr_auc.png)

![Experiment A numeric results](tables/m5-scarcity/m5_exp_a_matched_context_macro_pr_auc.png)

### Meter-level performance

![Experiment A meter trend panels](assets/m5-scarcity/m5_exp_a_matched_context_all_meters_pr_auc.png)

![Experiment A grouped meter comparison](assets/m5-scarcity/m5_exp_a_matched_context_meter_grouped_pr_auc.png)

![Experiment A budget-color meter blocks](assets/m5-scarcity/m5_exp_a_matched_context_meter_budget_blocks_pr_auc.png)

![Experiment A meter detail table](tables/m5-scarcity/m5_exp_a_matched_context_meter_pr_auc.png)

## Experiment B — representative source-building scarcity

### Objective

Measure sensitivity to the number of distinct labeled source buildings, K. The ladder is nested and representative; total context size is approximately 500 rows per building and the selected rows retain their natural class mix.

### Protocol

![Experiment B protocol](tables/m5-scarcity/m5_exp_b_building_count_setup.png)

### Aggregate performance

![Experiment B macro PR-AUC](assets/m5-scarcity/m5_exp_b_building_count_macro_pr_auc.png)

![Experiment B numeric results](tables/m5-scarcity/m5_exp_b_building_count_macro_pr_auc.png)

### Meter-level performance

![Experiment B meter trend panels](assets/m5-scarcity/m5_exp_b_building_count_all_meters_pr_auc.png)

![Experiment B grouped meter comparison](assets/m5-scarcity/m5_exp_b_building_count_meter_grouped_pr_auc.png)

![Experiment B budget-color meter blocks](assets/m5-scarcity/m5_exp_b_building_count_meter_budget_blocks_pr_auc.png)

![Experiment B meter detail table](tables/m5-scarcity/m5_exp_b_building_count_meter_pr_auc.png)

Paired result figures contain {paired}.

## Cross-experiment synthesis

These experiments answer complementary questions; their cells are not matched treatments. Experiment A varies row count under near-full building coverage and forced 1:1 training balance. Experiment B varies building diversity while row count grows with K and class balance remains natural. The following figures compare curve shape and coverage only; they must not be interpreted as a point-to-point contest between N and K budgets.

![Cross-experiment training coverage](assets/m5-scarcity/m5_cross_experiment_design_source_buildings.png)

![Cross-experiment macro curves](assets/m5-scarcity/m5_cross_experiment_macro_pr_auc.png)

## Reporting conventions

- Primary measure: equal-weight macro PR-AUC across electricity, chilled water, steam, and hot water.
- Meter-level figures always retain all four meter types.
- Seed 42 represents one sampled draw.
- Protocol and numeric tables are provided as compilable LaTeX source and LaTeX-rendered PDF/PNG.
"""
    _atomic_text(path, text.replace("\n+", "\n"))


def render_all(
    matched_path: Path,
    metrics_path: Path,
    manifest_path: Path,
    out_dir: Path,
    table_dir: Path,
    provenance_path: Path,
    draft_path: Path,
) -> list[Path]:
    matched = load_matched_meter_metrics(matched_path)
    manifest = load_building_manifest(manifest_path)
    building, paired_budgets = load_building_meter_metrics(metrics_path, manifest)
    matched_macro = macro_records(matched)
    building_macro = macro_records(building)

    outputs: list[Path] = []
    outputs += render_single_macro(matched_macro, "matched_context", out_dir)
    outputs += render_meter_panels(matched, "matched_context", out_dir)
    outputs += render_meter_grouped_bars(matched, "matched_context", out_dir)
    outputs += render_meter_budget_blocks(matched, "matched_context", out_dir)
    outputs += render_single_macro(building_macro, "building_count", out_dir)
    outputs += render_meter_panels(building, "building_count", out_dir)
    outputs += render_meter_grouped_bars(building, "building_count", out_dir)
    outputs += render_meter_budget_blocks(building, "building_count", out_dir)
    outputs += render_design(matched, manifest, out_dir)
    outputs += render_macro(matched_macro, building_macro, out_dir)

    exp_a_setup = table_dir / "m5_exp_a_matched_context_setup.tex"
    exp_a_results = table_dir / "m5_exp_a_matched_context_macro_pr_auc.tex"
    exp_a_meter_detail = table_dir / "m5_exp_a_matched_context_meter_pr_auc.tex"
    exp_b_setup = table_dir / "m5_exp_b_building_count_setup.tex"
    exp_b_results = table_dir / "m5_exp_b_building_count_macro_pr_auc.tex"
    exp_b_meter_detail = table_dir / "m5_exp_b_building_count_meter_pr_auc.tex"
    render_separate_setup_table("matched_context", exp_a_setup)
    render_exp_a_result_table(matched_macro, exp_a_results)
    render_meter_detail_table(
        matched, matched_macro, "matched_context", exp_a_meter_detail
    )
    render_separate_setup_table("building_count", exp_b_setup)
    render_exp_b_result_table(building_macro, manifest, exp_b_results)
    render_meter_detail_table(
        building, building_macro, "building_count", exp_b_meter_detail
    )

    write_provenance(
        provenance_path,
        matched_path,
        metrics_path,
        manifest_path,
        matched,
        building,
        paired_budgets,
    )
    write_draft(draft_path, paired_budgets)
    outputs += [
        exp_a_setup,
        exp_a_results,
        exp_a_meter_detail,
        exp_b_setup,
        exp_b_results,
        exp_b_meter_detail,
        provenance_path,
        draft_path,
    ]
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matched-report",
        type=Path,
        default=ROOT / "docs" / "reports" / "m5-matched-context-breakdown.md",
    )
    parser.add_argument(
        "--building-metrics",
        type=Path,
        default=PROC / "m5_building_curve" / "aggregate" / "metrics.csv",
    )
    parser.add_argument(
        "--building-manifest",
        type=Path,
        default=PROC
        / "m5_building_curve"
        / "protocol"
        / "representative"
        / "seed42"
        / "building_ladder.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs" / "reports" / "assets" / "m5-scarcity",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=ROOT / "docs" / "reports" / "tables" / "m5-scarcity",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=PROC / "m5_scarcity_submission" / "figure_data.json",
    )
    parser.add_argument(
        "--draft-report",
        type=Path,
        default=ROOT / "docs" / "reports" / "m5-scarcity-submission-draft.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = render_all(
        args.matched_report,
        args.building_metrics,
        args.building_manifest,
        args.out_dir,
        args.table_dir,
        args.provenance,
        args.draft_report,
    )
    for path in outputs:
        try:
            print(f"wrote {path.relative_to(ROOT)}")
        except ValueError:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
