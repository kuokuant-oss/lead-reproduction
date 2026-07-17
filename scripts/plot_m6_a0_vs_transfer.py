"""Per site: the site-unseen learning curve against the site-seen baseline.

Question answered by each figure: how far below the building-held-out baseline
does the site-unseen model sit, and does labelling more source meters close it?

A0 (building_id % 2 == 1 held out) has seen every site, so it is the ceiling
this pipeline can reach on a site. B1's transfer curve has never seen the site.
A0 scores only the odd buildings, so B1's predictions are re-scored on the same
odd-building subset before the two are put on one axis -- otherwise the curve
and the baseline would be computed over different rows.

A0 is a single cell, not a sweep, so it is drawn as a horizontal reference.

Style contract: docs/reference/plot-style-rules.md v0.3.
Source artifacts: data/processed/m6_site_transfer_b1_*_predictions.npz,
                  data/processed/m6_site_transfer_b2_a0_pos677077_seed42.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

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

BUDGETS = ("50", "100", "200", "400", "all")
SEEDS = (42, 123, 999)
A0_CELL = "m6_site_transfer_b2_a0_pos677077_seed42"
OBSERVATION = PROC / "m6_a0_vs_transfer.json"

SITE_DIRECTION = {
    **{s: "a1" for s in (1, 3, 5, 7, 9, 11, 13, 15)},
    **{s: "a2" for s in (0, 2, 4, 6, 8, 10, 12, 14)},
}
FLOW = {
    "a1": "Train on even sites, test on odd sites",
    "a2": "Train on odd sites, test on even sites",
}
METRICS = (
    {"key": "pr_auc", "panel": "PR-AUC (threshold-free)"},
    {"key": "f1", "panel": "F1 at threshold 0.5"},
    {"key": "recall", "panel": "Recall at threshold 0.5"},
)


def _score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, p >= 0.5, average="binary", zero_division=0
    )
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def observe() -> dict[str, Any]:
    a0 = json.loads((PROC / f"{A0_CELL}.json").read_text(encoding="utf-8"))
    baseline = {}
    for sid, v in a0["slices"]["by_site_id"].items():
        e = v["models"]["ensemble"]
        baseline[int(sid)] = {
            "pr_auc": e["pr_auc"],
            "f1": e["threshold_0_5"]["f1"],
            "recall": e["threshold_0_5"]["recall"],
            "precision": e["threshold_0_5"]["precision"],
            "n_rows": e["n_rows"],
            "n_anomalies": e["n_anomalies"],
            "anomaly_rate": e["anomaly_rate"],
        }

    curves: dict[str, list[dict[str, Any]]] = {}
    for direction in ("a1", "a2"):
        for budget in BUDGETS:
            for seed in SEEDS:
                stem = f"m6_site_transfer_b1_{direction}_meters{budget}_seed{seed}"
                cell = PROC / f"{stem}.json"
                npz_path = PROC / f"{stem}_predictions.npz"
                if not (cell.exists() and npz_path.exists()):
                    continue
                j = json.loads(cell.read_text(encoding="utf-8"))
                if j.get("status") != "completed":
                    continue
                # Close each archive before opening the next: the B1 NPZs are
                # ~250 MB each and 26 of them will not co-reside.
                with np.load(npz_path) as d:
                    odd = d["building_id"] % 2 == 1
                    site = d["site_id"][odd]
                    y = d["anomaly"][odd].astype(np.int8)
                    p = d["ensemble"][odd]
                meters = len(j["selection"]["selected_meters"])
                for sid in np.unique(site):
                    m = site == sid
                    rec = _score(y[m], p[m])
                    # Gate: A0 scores this site's odd buildings, so the re-scored
                    # B1 subset must land on exactly the same rows.
                    b = baseline[int(sid)]
                    if (
                        int(m.sum()) != b["n_rows"]
                        or int(y[m].sum()) != b["n_anomalies"]
                    ):
                        raise SystemExit(
                            f"{stem} site {sid}: {m.sum():,} rows / {int(y[m].sum()):,} anomalies "
                            f"vs baseline {b['n_rows']:,} / {b['n_anomalies']:,}"
                        )
                    rec.update(
                        site_id=int(sid),
                        direction=direction,
                        budget=budget,
                        seed=seed,
                        meters=meters,
                        n_rows=int(m.sum()),
                        n_anomalies=int(y[m].sum()),
                    )
                    curves.setdefault(str(int(sid)), []).append(rec)
                del odd, site, y, p
            print(f"  rescored {direction} meters{budget}")

    return {
        "schema_version": 1,
        "experiment": "m6_a0_vs_transfer",
        "note": (
            "B1 ensemble predictions re-scored on each site's odd buildings, the rows A0 "
            "scores, so the transfer curve and the building-held-out baseline share rows."
        ),
        "baseline_cell": A0_CELL,
        "baseline_by_site": {str(k): v for k, v in baseline.items()},
        "curves_by_site": curves,
    }


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def render_site(data: dict[str, Any], site: int, out_dir: Path, xlim) -> Path:
    rows = data["curves_by_site"][str(site)]
    base = data["baseline_by_site"][str(site)]
    direction = SITE_DIRECTION[site]

    seeds_done = tuple(
        s
        for s in SEEDS
        if all(any(r["seed"] == s and r["budget"] == b for r in rows) for b in BUDGETS)
    )
    if not seeds_done:
        raise SystemExit(f"site {site}: no seed has every budget")

    fig, axes = plt.subplots(1, 3, figsize=(10.7, 6.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    all_meters = sorted({r["meters"] for r in rows if r["seed"] in seeds_done})
    for idx, (ax, metric) in enumerate(zip(axes, METRICS, strict=True)):
        _style_axis(ax)
        k = metric["key"]

        xs, ys = [], []
        for m in all_meters:
            vals = [r[k] for r in rows if r["meters"] == m and r["seed"] in seeds_done]
            if vals:
                xs.append(m)
                ys.append(float(np.mean(vals)))
        ax.plot(xs, ys, color=INK, marker="p", markersize=4.2, linewidth=1.35, zorder=3)
        # A0 is the reference arm, so dashed per the style contract.
        ax.axhline(base[k], color=INK, linewidth=1.0, linestyle=(0, (5, 3)), zorder=2)

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
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if idx == 0:
            ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])

    fig.suptitle(
        f"Unseen-site penalty for {site_label(site, with_id=True)}",
        x=0.055,
        y=0.962,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    # Subtitle width budget: canvas 10.7in, text starts at 0.59in, ~9.96in usable.
    # At 10pt that is ~130 characters, at 9pt ~145. Keep every line under it.
    fig.text(
        0.055,
        0.897,
        "How far the site-unseen model sits below a baseline that has seen this site.",
        ha="left",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.text(
        0.055,
        0.866,
        f"{FLOW[direction]}. Tree Ensemble, {len(seeds_done)} seeds.",
        ha="left",
        fontsize=10,
        color=SECONDARY,
    )
    fig.text(
        0.055,
        0.840,
        f"Odd buildings only ({base['n_rows']:,} rows, {base['anomaly_rate']:.2%}) — the rows the "
        f"baseline covers. Not comparable to the whole-site curves.",
        ha="left",
        fontsize=9,
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
        "Score on this site's odd buildings only",
        va="center",
        rotation="vertical",
        fontsize=10.5,
        color=SECONDARY,
    )

    handles = [
        plt.Line2D(
            [],
            [],
            color=INK,
            marker="p",
            markersize=4.2,
            linewidth=1.35,
            label="Site unseen (meter budgets)",
        ),
        plt.Line2D(
            [],
            [],
            color=INK,
            linewidth=1.0,
            linestyle=(0, (5, 3)),
            label="Site seen (building-held-out baseline)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        labelcolor=SECONDARY,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.766, bottom=0.169, wspace=0.07)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"m6_unseen_site_penalty_site_{site}_pr_auc_f1_recall.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir", type=Path, default=ROOT / "docs" / "reports" / "assets" / "m6"
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
        OBSERVATION.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"wrote {OBSERVATION.relative_to(ROOT)}")

    meters = [r["meters"] for rows in data["curves_by_site"].values() for r in rows]
    xlim = (min(meters) * 0.85, max(meters) * 1.18)
    for site in sorted(int(s) for s in data["curves_by_site"]):
        path = render_site(data, site, args.out_dir, xlim)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
