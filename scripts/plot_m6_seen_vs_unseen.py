"""Seen against unseen, per site, on the whole site.

Question answered by each figure: at each labelled-meter budget, how much worse
is a model that has never seen this site than one that has?

  unseen = B1's own per-site numbers, read straight from its cell JSON. Not
           recomputed, not re-scored, not subset. The curve is B1's curve.
  seen   = A0's two building folds. a0odd trains on even buildings and scores
           the odd ones; a0even does the reverse. Unioned, every building carries
           a prediction from a model that never trained on it, so the arm covers
           the whole site -- the same rows B1 covers. A single fold could only
           cover half a site, which would have forced B1 onto that half.

Both arms sweep the same meter budgets, so the two curves share an x axis, share
rows, and differ only in whether the site was in training.

Style contract: docs/reference/plot-style-rules.md v0.3.
Source artifacts: data/processed/m6_site_transfer_b1_{a1,a2,a0odd,a0even}_*.json
                  and their _predictions.npz
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
FOLDS = ("a0odd", "a0even")
OBSERVATION = PROC / "m6_seen_vs_unseen.json"

SITE_DIRECTION = {
    **{s: "a1" for s in (1, 3, 5, 7, 9, 11, 13, 15)},
    **{s: "a2" for s in (0, 2, 4, 6, 8, 10, 12, 14)},
}
FLOW = {
    "a1": "Unseen arm trains on even sites, tests on odd sites",
    "a2": "Unseen arm trains on odd sites, tests on even sites",
}
METRICS = (
    {"key": "pr_auc", "panel": "PR-AUC (threshold-free)"},
    {"key": "f1", "panel": "F1 at threshold 0.5"},
    {"key": "recall", "panel": "Recall at threshold 0.5"},
)


def _load(stem: str) -> dict[str, Any] | None:
    f = PROC / f"{stem}.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text(encoding="utf-8"))
    return j if j.get("status") == "completed" else None


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
    """Read B1 as-is; union A0's two folds into whole-site scores."""
    unseen: dict[str, list[dict[str, Any]]] = {}
    for direction in ("a1", "a2"):
        for budget in BUDGETS:
            for seed in SEEDS:
                j = _load(f"m6_site_transfer_b1_{direction}_meters{budget}_seed{seed}")
                if not j:
                    continue
                meters = len(j["selection"]["selected_meters"])
                for sid, v in j["slices"]["by_site_id"].items():
                    e = v["models"]["ensemble"]
                    unseen.setdefault(sid, []).append(
                        {
                            "pr_auc": e["pr_auc"],
                            "f1": e["threshold_0_5"]["f1"],
                            "recall": e["threshold_0_5"]["recall"],
                            "precision": e["threshold_0_5"]["precision"],
                            "budget": budget,
                            "seed": seed,
                            "meters": meters,
                            "n_rows": e["n_rows"],
                            "n_anomalies": e["n_anomalies"],
                            "anomaly_rate": e["anomaly_rate"],
                        }
                    )

    seen: dict[str, list[dict[str, Any]]] = {}
    for budget in BUDGETS:
        for seed in SEEDS:
            parts = []
            meters = None
            for fold in FOLDS:
                stem = f"m6_site_transfer_b1_{fold}_meters{budget}_seed{seed}"
                j = _load(stem)
                npz = PROC / f"{stem}_predictions.npz"
                if not (j and npz.exists()):
                    parts = []
                    break
                # One archive at a time: these are ~250 MB each.
                with np.load(npz) as d:
                    parts.append(
                        (
                            d["site_id"].copy(),
                            d["anomaly"].astype(np.int8),
                            d["ensemble"].copy(),
                        )
                    )
                meters = len(j["selection"]["selected_meters"])
            if len(parts) != len(FOLDS):
                continue

            site = np.concatenate([p[0] for p in parts])
            y = np.concatenate([p[1] for p in parts])
            p_hat = np.concatenate([p[2] for p in parts])
            del parts

            for sid in np.unique(site):
                m = site == sid
                rec = _score(y[m], p_hat[m])
                rec.update(
                    budget=budget,
                    seed=seed,
                    meters=meters,
                    n_rows=int(m.sum()),
                    n_anomalies=int(y[m].sum()),
                    anomaly_rate=float(y[m].mean()),
                )
                seen.setdefault(str(int(sid)), []).append(rec)
            del site, y, p_hat
            print(f"  unioned a0 folds at meters{budget} seed{seed}")

    # Gate: the union must land on exactly the rows B1 already covers, or the
    # two curves are not on the same footing and must not share an axis.
    for sid, rows in seen.items():
        ref = unseen.get(sid)
        if not ref:
            continue
        want_rows, want_anom = ref[0]["n_rows"], ref[0]["n_anomalies"]
        for r in rows:
            if r["n_rows"] != want_rows or r["n_anomalies"] != want_anom:
                raise SystemExit(
                    f"site {sid}: seen arm has {r['n_rows']:,} rows / {r['n_anomalies']:,} "
                    f"anomalies, unseen arm has {want_rows:,} / {want_anom:,}"
                )

    return {
        "schema_version": 1,
        "experiment": "m6_seen_vs_unseen",
        "note": (
            "unseen_by_site is B1's own per-site output, unmodified. seen_by_site unions "
            "the a0odd and a0even folds so every building is scored by a model that did "
            "not train on it, giving whole-site coverage on B1's rows."
        ),
        "unseen_by_site": unseen,
        "seen_by_site": seen,
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


def _complete_seeds(rows: list[dict[str, Any]]) -> tuple[int, ...]:
    """Seeds that finished every budget; a partial seed would shift only part of a curve."""
    return tuple(
        s
        for s in SEEDS
        if all(any(r["seed"] == s and r["budget"] == b for r in rows) for b in BUDGETS)
    )


def _curve(rows: list[dict[str, Any]], key: str, seeds: tuple[int, ...]):
    xs, ys = [], []
    for m in sorted({r["meters"] for r in rows if r["seed"] in seeds}):
        vals = [r[key] for r in rows if r["meters"] == m and r["seed"] in seeds]
        if vals:
            xs.append(m)
            ys.append(float(np.mean(vals)))
    return xs, ys


def render_site(data: dict[str, Any], site: int, out_dir: Path, xlim) -> Path:
    unseen = data["unseen_by_site"][str(site)]
    seen = data["seen_by_site"].get(str(site), [])
    direction = SITE_DIRECTION[site]

    u_seeds = _complete_seeds(unseen)
    s_seeds = _complete_seeds(seen) if seen else ()
    if not u_seeds:
        raise SystemExit(f"site {site}: unseen arm has no complete seed")

    fig, axes = plt.subplots(1, 3, figsize=(10.7, 6.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    ticks = sorted({r["meters"] for r in unseen if r["seed"] in u_seeds})
    for idx, (ax, metric) in enumerate(zip(axes, METRICS, strict=True)):
        _style_axis(ax)
        k = metric["key"]

        if s_seeds:
            xs, ys = _curve(seen, k, s_seeds)
            ax.plot(
                xs,
                ys,
                color=INK,
                marker="p",
                markersize=4.2,
                linewidth=1.35,
                linestyle=(0, (5, 3)),
                zorder=2,
            )
        xs, ys = _curve(unseen, k, u_seeds)
        ax.plot(xs, ys, color=INK, marker="p", markersize=4.2, linewidth=1.35, zorder=3)

        ax.set_title(
            metric["panel"],
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
            pad=12,
        )
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [f"{m:,}\n(all)" if m == ticks[-1] else f"{m:,}" for m in ticks]
        )
        ax.minorticks_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, 1.04)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if idx == 0:
            ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])

    base = unseen[0]
    fig.suptitle(
        f"Seen against unseen for {site_label(site, with_id=True)}",
        x=0.055,
        y=0.962,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    # Subtitle width budget: canvas 10.7in, text starts at 0.59in, ~9.96in
    # usable. At 10pt that is ~130 characters, at 9pt ~145. Stay under it.
    fig.text(
        0.055,
        0.897,
        "How much worse a model that never saw this site is than one that has.",
        ha="left",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.text(
        0.055,
        0.866,
        f"{FLOW[direction]}. Tree Ensemble, {len(u_seeds)} seeds.",
        ha="left",
        fontsize=10,
        color=SECONDARY,
    )
    seen_note = (
        f"Seen arm: A0's two building folds unioned, {len(s_seeds)} seeds."
        if s_seeds
        else "Seen arm not yet run."
    )
    fig.text(
        0.055,
        0.840,
        f"{seen_note} Both on this site's {base['n_rows']:,} rows "
        f"({base['anomaly_rate']:.2%} anomalies).",
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
        "Score on the whole held-out site",
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
            label="Site unseen (B1)",
        ),
        plt.Line2D(
            [],
            [],
            color=INK,
            marker="p",
            markersize=4.2,
            linewidth=1.35,
            linestyle=(0, (5, 3)),
            label="Site seen (A0, two folds unioned)",
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
    path = out_dir / f"m6_seen_vs_unseen_site_{site}_pr_auc_f1_recall.png"
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

    meters = [r["meters"] for rows in data["unseen_by_site"].values() for r in rows]
    xlim = (min(meters) * 0.85, max(meters) * 1.18)
    for site in sorted(int(s) for s in data["unseen_by_site"]):
        path = render_site(data, site, args.out_dir, xlim)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
