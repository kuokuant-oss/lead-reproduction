"""Anomaly concentration per site: how much of a site sits in one building.

Question answered by the figure: how much of each site's anomalies is held by
its single most anomalous building?

Site-level rows/buildings/prevalence hide this. Sites 4 and 11 look unremarkable
at site level (91 and 5 buildings) but nearly all of their anomalies sit in one
building each, which is what degenerates every within-site design.

Writes the numeric observation to data/processed/ first, then renders from it,
so the figure and the numbers stay traceable to one artifact.

Style contract: docs/reference/plot-style-rules.md v0.3.
Source: data/raw/m3/{train.csv,bad_meter_readings.csv,building_metadata.csv}
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

from lead import M3, PROC, ROOT, write_json_with_provenance
from m6_site_names import label as site_label
from m6_site_names import site_names

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

OBSERVATION = PROC / "m6_site_structure.json"


def observe() -> dict:
    t = pd.read_csv(
        M3 / "train.csv",
        usecols=["building_id", "meter"],
        dtype={"building_id": "int16", "meter": "int8"},
    )
    bad = pd.read_csv(M3 / "bad_meter_readings.csv")
    if len(bad) != len(t):
        raise SystemExit("bad_meter_readings.csv must align 1:1 with train.csv")
    t["anomaly"] = bad["is_bad_meter_reading"].to_numpy(np.int8)
    meta = pd.read_csv(M3 / "building_metadata.csv", usecols=["building_id", "site_id"])
    t["site_id"] = (
        t["building_id"].map(meta.set_index("building_id")["site_id"]).astype(np.int8)
    )

    names = site_names()
    sites = []
    for s, g in t.groupby("site_id"):
        per_b = g.groupby("building_id")["anomaly"].agg(["sum", "count"])
        per_b["rate"] = per_b["sum"] / per_b["count"]
        total = int(per_b["sum"].sum())
        ranked = np.sort(per_b["sum"].to_numpy())[::-1]
        shares = ranked / max(total, 1)
        n90 = int(np.searchsorted(np.cumsum(shares), 0.90) + 1) if total else 0
        top_b = int(per_b["sum"].idxmax())
        even = int(g.loc[g["building_id"] % 2 == 0, "anomaly"].sum())

        sites.append(
            {
                "site_id": int(s),
                "site_name": names[int(s)],
                "buildings": int(g["building_id"].nunique()),
                "meter_series": int(g.groupby(["building_id", "meter"]).ngroups),
                "rows": int(len(g)),
                "anomalies": total,
                "anomaly_rate": float(g["anomaly"].mean()),
                "top_building_id": top_b,
                "top_building_share": float(shares[0]) if total else 0.0,
                "top_building_rate": float(per_b.loc[top_b, "rate"]),
                "buildings_for_90pct": n90,
                "near_clean_buildings": int((per_b["rate"] < 0.001).sum()),
                "even_half_share": float(even / total) if total else 0.0,
                "odd_half_anomalies": total - even,
                # Source array for the cumulative-share curve, per the plot-data
                # contract: the figure must be rebuildable from this JSON alone.
                "building_anomalies_sorted": [int(v) for v in ranked],
            }
        )

    return {
        "schema_version": 1,
        "experiment": "m6_site_structure",
        "observation_only": True,
        "definitions": {
            "site_name": "BDG2 site name for the GEPIII site_id, read from data/raw/bdg2/metadata.csv",
            "top_building_share": "share of the site's anomalies held by its single most anomalous building",
            "buildings_for_90pct": "buildings needed, ranked by anomaly count, to cover 90% of the site's anomalies",
            "near_clean_buildings": "buildings whose anomaly rate is below 0.1%",
            "even_half_share": "share of the site's anomalies in even-numbered buildings; the A5 oracle and a0 train on even and test on odd",
        },
        "totals": {
            "sites": len(sites),
            "buildings": int(t["building_id"].nunique()),
            "rows": int(len(t)),
            "anomalies": int(t["anomaly"].sum()),
            "anomaly_rate": float(t["anomaly"].mean()),
        },
        "sites": sites,
        "note": "Read-only description of the frozen M3 inputs; no model was fitted and no M3 artifact was touched.",
    }


def render(data: dict, path: Path) -> Path:
    sites = sorted(data["sites"], key=lambda r: r["top_building_share"])
    y = np.arange(len(sites))
    share = np.array([r["top_building_share"] for r in sites])

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)

    ax.hlines(y, 0, share, color=MUTED, linewidth=0.7, zorder=1)
    ax.plot(share, y, "o", color=INK, markersize=4.2, linestyle="none", zorder=2)

    for yi, r in zip(y, sites, strict=True):
        ax.text(
            r["top_building_share"] + 0.018,
            yi,
            f"{r['top_building_share']:.1%}",
            va="center",
            fontsize=9,
            color=SECONDARY,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{site_label(r['site_id'])} · {r['buildings']} bldg" for r in sites],
        fontsize=9,
    )
    ax.tick_params(colors=SECONDARY, labelsize=9, length=0)
    ax.set_xlim(0, 1.10)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_ylim(-0.8, len(sites) - 0.2)

    fig.suptitle(
        "Anomaly concentration by site",
        x=0.06,
        y=0.968,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.903,
        "How much of each site's anomalies its single most anomalous building holds.",
        ha="left",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.text(
        0.06,
        0.874,
        "Building count is printed next to each site: it does not predict concentration.",
        ha="left",
        fontsize=10,
        color=SECONDARY,
    )
    fig.text(
        0.5,
        0.045,
        "Share of the site's anomalies in its most anomalous building",
        ha="center",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.subplots_adjust(left=0.175, right=0.975, top=0.845, bottom=0.105)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    return path


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=8.5)


def render_cumulative(data: dict, path: Path) -> Path:
    """Question: how unevenly are a site's anomalies spread across its buildings?"""
    sites = sorted(data["sites"], key=lambda r: -r["top_building_share"])

    # Both axes are 0-1 cumulative shares, so the drawing area is kept square.
    fig, axes = plt.subplots(4, 4, figsize=(8.6, 10.4), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for idx, (ax, r) in enumerate(zip(axes.ravel(), sites, strict=True)):
        _style_axis(ax)
        counts = np.array(r["building_anomalies_sorted"], dtype=float)
        n = len(counts)
        # Step from the origin: 0 buildings cover 0 anomalies.
        x = np.concatenate([[0.0], np.arange(1, n + 1) / n])
        y = np.concatenate([[0.0], np.cumsum(counts) / max(counts.sum(), 1)])
        ax.plot(
            [0, 1], [0, 1], color=MUTED, linewidth=0.7, linestyle=(0, (1, 3)), zorder=1
        )
        ax.plot(x, y, color=INK, linewidth=1.35, zorder=2)

        ax.set_title(
            site_label(r["site_id"]),
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
            pad=18,
        )
        ax.text(
            0.0,
            1.02,
            f"{r['buildings']} bldg · top {r['top_building_share']:.0%}",
            transform=ax.transAxes,
            fontsize=8,
            color=MUTED,
            ha="left",
        )
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        if idx >= 12:
            ax.set_xticklabels(["0", "50%", "100%"])
        if idx % 4 == 0:
            ax.set_yticklabels(["0", "50%", "100%"])
        ax.set_box_aspect(1)

    fig.suptitle(
        "Cumulative anomaly share by site",
        x=0.055,
        y=0.972,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.930,
        "How unevenly each site's anomalies are spread across its buildings.",
        ha="left",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.text(
        0.055,
        0.911,
        "Dotted diagonal = perfectly even. A curve hugging the top-left means one building is the site.",
        ha="left",
        fontsize=10,
        color=SECONDARY,
    )
    fig.text(
        0.5,
        0.038,
        "Share of the site's buildings, most anomalous first",
        ha="center",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.text(
        0.013,
        0.47,
        "Cumulative share of the site's anomalies",
        va="center",
        rotation="vertical",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.subplots_adjust(
        left=0.085, right=0.985, top=0.865, bottom=0.075, hspace=0.42, wspace=0.12
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    return path


def render_scatter(data: dict, path: Path) -> Path:
    """Question: does a site's building count predict how concentrated it is?"""
    sites = data["sites"]
    x = np.array([r["buildings"] for r in sites], dtype=float)
    y = np.array([r["top_building_share"] for r in sites])

    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    fig.patch.set_facecolor(SURFACE)
    _style_axis(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    ax.plot(x, y, "o", color=INK, markersize=4.2, linestyle="none", zorder=3)
    for r in sites:
        ax.annotate(
            site_label(r["site_id"]),
            xy=(r["buildings"], r["top_building_share"]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            color=SECONDARY,
            va="center",
        )

    ax.set_xscale("log")
    ax.set_xticks([5, 10, 30, 50, 100, 300])
    ax.set_xticklabels(["5", "10", "30", "50", "100", "300"])
    ax.minorticks_off()
    ax.set_xlim(4, 400)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.tick_params(labelsize=9)

    fig.suptitle(
        "Building count against anomaly concentration",
        x=0.06,
        y=0.968,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.898,
        "Whether the number of buildings in a site says anything about how concentrated its anomalies are.",
        ha="left",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.text(
        0.06,
        0.866,
        "Points are labelled with the site name.",
        ha="left",
        fontsize=10,
        color=SECONDARY,
    )
    fig.text(
        0.5,
        0.042,
        "Buildings in the site (log scale)",
        ha="center",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.text(
        0.015,
        0.47,
        "Share of the site's anomalies in its most anomalous building",
        va="center",
        rotation="vertical",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.subplots_adjust(left=0.10, right=0.975, top=0.835, bottom=0.115)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs" / "reports" / "assets" / "m6" / "site-structure",
    )
    p.add_argument(
        "--reuse", action="store_true", help="plot from the existing observation JSON"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.reuse and OBSERVATION.exists():
        data = json.loads(OBSERVATION.read_text(encoding="utf-8"))
    else:
        data = observe()
        write_json_with_provenance(OBSERVATION, data, root=ROOT)
        print(f"wrote {OBSERVATION.relative_to(ROOT)}")
    for fn, name in (
        (render, "m6_anomaly_concentration_by_site_top_building_share.png"),
        (render_cumulative, "m6_anomaly_concentration_by_site_cumulative_share.png"),
        (
            render_scatter,
            "m6_anomaly_concentration_vs_building_count_top_building_share.png",
        ),
    ):
        path = fn(data, args.out_dir / name)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
