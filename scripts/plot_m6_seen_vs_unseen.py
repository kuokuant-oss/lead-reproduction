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
                  data/processed/m6_tabpfn_b1_{a1,a2}_*_context{10000,100000}.json
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

# Arm colors. Both arms are the same model (Tree Ensemble), so this is a
# deliberate departure from plot-style-rules v0.3 §6/§7.2 (which encode the arm
# by linestyle and keep colour for model identity): the seen/unseen arms are
# distinguished by colour here, both drawn solid, at the owner's request. The
# legend labels ("B1"/"A0") disambiguate so neither reads as a different model.
UNSEEN_COLOR = "#2a78d6"  # site unseen (B1)
SEEN_COLOR = "#e07a00"  # site seen (A0, two folds unioned)
TABPFN_COLOR = "#6f4aa8"
TABPFN_CONTEXT_STYLES = {
    "10000": {
        "label": "TabPFN trained on 10,000 rows; site excluded",
        "linestyle": "-",
        "marker": "o",
    },
    "100000": {
        "label": "TabPFN trained on 100,000 rows; site excluded",
        "linestyle": "-",
        "marker": "D",
    },
}

BUDGETS = ("50", "100", "200", "400", "all")
SEEDS = (42, 123, 999)
FOLDS = ("a0odd", "a0even")
OBSERVATION = PROC / "m6_seen_vs_unseen.json"

SITE_DIRECTION = {
    **{s: "a1" for s in (1, 3, 5, 7, 9, 11, 13, 15)},
    **{s: "a2" for s in (0, 2, 4, 6, 8, 10, 12, 14)},
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


def observe_tabpfn() -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Read only completed full-site TabPFN cells; partial chunks never plot."""
    by_context: dict[str, dict[str, list[dict[str, Any]]]] = {}
    expected = {
        "1": {"rows": 553_357, "anomalies": 77_779},
        "8": {"rows": 567_915, "anomalies": 43_504},
    }
    for context_rows in TABPFN_CONTEXT_STYLES:
        context_sites: dict[str, list[dict[str, Any]]] = {}
        for direction, site in (("a1", 1), ("a2", 8)):
            sid = str(site)
            for budget in BUDGETS:
                for seed in SEEDS:
                    stem = (
                        f"m6_tabpfn_b1_{direction}_meters{budget}_seed{seed}"
                        f"_context{context_rows}"
                    )
                    payload = _load(stem)
                    if not payload:
                        continue
                    if int(payload.get("target_site", -1)) != site:
                        raise SystemExit(f"{stem}: target site drift")
                    query = payload["query"]
                    want = expected[sid]
                    if (
                        int(query["rows"]) != want["rows"]
                        or int(query["anomalies"]) != want["anomalies"]
                    ):
                        raise SystemExit(
                            f"{stem}: TabPFN full-site support drifted from "
                            f"{want['rows']:,} / {want['anomalies']:,}"
                        )
                    metric = payload["metrics"]["tabpfn"]
                    threshold = metric["threshold_0_5"]
                    context_sites.setdefault(sid, []).append(
                        {
                            "pr_auc": float(metric["pr_auc"]),
                            "f1": float(threshold["f1"]),
                            "recall": float(threshold["recall"]),
                            "precision": float(threshold["precision"]),
                            "budget": budget,
                            "seed": seed,
                            "meters": (
                                int(payload["context"].get("selected_meters", budget))
                                if budget != "all"
                                else int(
                                    payload["context"].get(
                                        "available_meters",
                                        payload["context"].get("selected_meters", 0),
                                    )
                                )
                            ),
                            "n_rows": int(query["rows"]),
                            "n_anomalies": int(query["anomalies"]),
                            "anomaly_rate": float(query["anomalies"] / query["rows"]),
                            "context_rows": int(payload["context"]["context_rows"]),
                        }
                    )
        by_context[context_rows] = context_sites
    return by_context


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
        "tabpfn_unseen_by_context": observe_tabpfn(),
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


def _curve(rows: list[dict[str, Any]], key: str, seeds: tuple[int, ...], all_x: int):
    """x is the meter budget: real counts for 50-400 (identical across arms),
    and a shared position `all_x` for the 'all' budget, whose true pool size
    differs by arm (site split 1,347 vs building split 1,196) and so cannot
    share a real meter x. The two 'all' points are the same budget level, so
    they align here; the exact pool sizes live in the report."""
    xs, ys = [], []
    for b in BUDGETS:
        vals = [r[key] for r in rows if r["budget"] == b and r["seed"] in seeds]
        if vals:
            xs.append(all_x if b == "all" else int(b))
            ys.append(float(np.mean(vals)))
    return xs, ys


def _available_curve(
    rows: list[dict[str, Any]], key: str, all_x: int
) -> tuple[list[int], list[float], list[int]]:
    """Plot completed full-site budget points; partial query chunks stay excluded."""
    xs, ys, counts = [], [], []
    for budget in BUDGETS:
        values = [float(row[key]) for row in rows if row["budget"] == budget]
        if not values:
            continue
        xs.append(all_x if budget == "all" else int(budget))
        ys.append(float(np.mean(values)))
        counts.append(len(values))
    return xs, ys, counts


def _complete_tabpfn_seeds(rows: list[dict[str, Any]]) -> tuple[int, ...]:
    """A TabPFN seed is plottable only after all five budgets complete."""
    return tuple(
        seed
        for seed in SEEDS
        if all(
            any(row["seed"] == seed and row["budget"] == budget for row in rows)
            for budget in BUDGETS
        )
    )


def render_site(
    data: dict[str, Any], site: int, out_dir: Path, xlim, all_x: int
) -> Path:
    unseen = data["unseen_by_site"][str(site)]
    seen = data["seen_by_site"].get(str(site), [])
    source_parity = "even-numbered" if SITE_DIRECTION[site] == "a1" else "odd-numbered"
    tabpfn_by_context = {
        context: sites.get(str(site), [])
        for context, sites in data.get("tabpfn_unseen_by_context", {}).items()
    }
    u_seeds = _complete_seeds(unseen)
    s_seeds = _complete_seeds(seen) if seen else ()
    if not u_seeds:
        raise SystemExit(f"site {site}: unseen arm has no complete seed")

    fig, axes = plt.subplots(1, 3, figsize=(10.7, 6.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    present = [
        b
        for b in BUDGETS
        if any(r["budget"] == b and r["seed"] in u_seeds for r in unseen)
    ]
    ticks = [all_x if b == "all" else int(b) for b in present]
    for idx, (ax, metric) in enumerate(zip(axes, METRICS, strict=True)):
        _style_axis(ax)
        k = metric["key"]

        if s_seeds:
            xs, ys = _curve(seen, k, s_seeds, all_x)
            ax.plot(
                xs,
                ys,
                color=SEEN_COLOR,
                marker="p",
                markersize=4.2,
                linewidth=1.35,
                zorder=2,
            )
        xs, ys = _curve(unseen, k, u_seeds, all_x)
        ax.plot(
            xs,
            ys,
            color=UNSEEN_COLOR,
            marker="p",
            markersize=4.2,
            linewidth=1.35,
            zorder=3,
        )
        for context, rows in tabpfn_by_context.items():
            if not rows:
                continue
            style = TABPFN_CONTEXT_STYLES[context]
            tx, ty, _seed_counts = _available_curve(rows, k, all_x)
            ax.plot(
                tx,
                ty,
                color=TABPFN_COLOR,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.5,
                linewidth=1.45,
                zorder=4,
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
        ax.set_xticks(ticks)
        ax.set_xticklabels(["all" if m == all_x else f"{m:,}" for m in ticks])
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
    subtitle_lines = [
        f"All models are scored on every row at this site: {base['n_rows']:,} rows, "
        f"{base['anomaly_rate']:.2%} anomalies.",
        f"Tree models use every row from selected meters. Blue trains on "
        f"{source_parity} sites; orange includes this site but holds out buildings.",
        "Purple TabPFN samples 10,000 rows from the same selected meters, fits once, "
        "then predicts this site in 4,000-row batches.",
    ]
    fig.text(
        0.055,
        0.905,
        "\n".join(subtitle_lines),
        ha="left",
        va="top",
        fontsize=9.5,
        linespacing=1.5,
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
            color=UNSEEN_COLOR,
            marker="p",
            markersize=4.2,
            linewidth=1.35,
            label="Tree model; training excluded this site",
        ),
        plt.Line2D(
            [],
            [],
            color=SEEN_COLOR,
            marker="p",
            markersize=4.2,
            linewidth=1.35,
            label="Tree model; training included this site",
        ),
    ]
    for context, rows in tabpfn_by_context.items():
        if not rows:
            continue
        style = TABPFN_CONTEXT_STYLES[context]
        handles.append(
            plt.Line2D(
                [],
                [],
                color=TABPFN_COLOR,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.5,
                linewidth=1.45,
                label=style["label"],
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=9.5,
        labelcolor=SECONDARY,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.720, bottom=0.169, wspace=0.07)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"m6_seen_vs_unseen_site_{site}_pr_auc_f1_recall.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs" / "reports" / "assets" / "m6" / "seen-vs-unseen",
    )
    p.add_argument(
        "--reuse", action="store_true", help="plot from the existing observation JSON"
    )
    p.add_argument(
        "--sites",
        type=int,
        nargs="+",
        help="Render only these site IDs. Defaults to every observed site.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.reuse and OBSERVATION.exists():
        data = json.loads(OBSERVATION.read_text(encoding="utf-8"))
        data["tabpfn_unseen_by_context"] = observe_tabpfn()
        OBSERVATION.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        data = observe()
        OBSERVATION.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"wrote {OBSERVATION.relative_to(ROOT)}")

    all_x = max(
        r["meters"]
        for arm in ("unseen_by_site", "seen_by_site")
        for rows in data[arm].values()
        for r in rows
        if r["budget"] == "all"
    )
    reals = [int(b) for b in BUDGETS if b != "all"]
    xlim = (min(reals) * 0.85, all_x * 1.18)
    sites = (
        sorted(args.sites)
        if args.sites
        else sorted(int(s) for s in data["unseen_by_site"])
    )
    for site in sites:
        if str(site) not in data["unseen_by_site"]:
            raise SystemExit(f"site {site}: no unseen B1 observations")
        path = render_site(data, site, args.out_dir, xlim, all_x)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
