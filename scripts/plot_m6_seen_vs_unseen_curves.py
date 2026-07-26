"""Seen against unseen as threshold-free curves, pooled and per site.

Question answered: on exactly the same evaluation rows, how much discrimination
does the Tree Ensemble lose when the site was never in training?

These are the seen/unseen counterparts of the four M3 feature-engineering
figures. Those figures hold the split fixed (the 50/50 building holdout) and
vary the feature set (17 versus 137). These hold the feature set fixed at 137
and vary the split instead:

  seen   = A0's odd-building fold at the full meter budget. It trains on the
           even buildings and scores the odd ones, so every row carries a
           prediction from a model that never trained on that building but did
           train on that site. This is the M3 figures' own evaluation set: its
           pooled scores reproduce their 0.9303 PR-AUC and 0.9918 ROC-AUC
           exactly, and the run asserts that below.
  unseen = B1's two site folds at the full meter budget. a1 trains on the eight
           even sites and scores the odd ones; a2 does the reverse. Unioned they
           cover every building, so they are then cut down to the odd buildings
           the seen arm holds out.

Restricting to one building parity, rather than unioning A0's two folds into
whole-site coverage, is what keeps these four figures on the same rows as the
M3 pair they are read against. The cut is done by row id, not by recomputing
``building_id % 2``, and the two arms are then compared element for element on
``validation_raw_index``, site and label before any curve is drawn.

There is no 17-feature site-transfer artifact, so no 17-feature line exists for
the unseen arm and none is drawn for the seen arm either; mixing one would turn
a split contrast back into a feature contrast.

Style contract: docs/reference/plot-style-rules.md v0.3.
Source artifacts: data/processed/m6_site_transfer_b1_{a0odd,a1,a2}_metersall_seed42
                  and their _predictions.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lead import PROC, ROOT
from m6_site_names import site_names

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"

# Arm colors. Both arms are the same model (Tree Ensemble, 137 features), so
# this is the same deliberate departure from plot-style-rules v0.3 sections 6
# and 7.2 that scripts/plot_m6_seen_vs_unseen.py makes: those encode the arm by
# linestyle and reserve colour for model identity, and here colour carries the
# split instead. The legend names the split, so neither arm reads as a different
# model.
#
# The assignment is the reverse of that sibling script, at the owner's request:
# there blue is the unseen arm, here blue is the seen arm, so that seen keeps
# the blue it has in the M3 137-feature figures these four are read against.
SEEN_COLOR = "#2a78d6"  # site seen in training (A0, odd-building fold)
UNSEEN_COLOR = "#e07a00"  # site unseen in training (B1, two site folds)

SEEN_FOLDS = ("a0odd",)
UNSEEN_FOLDS = ("a1", "a2")
BUDGET = "metersall"
SEED = 42

OBSERVATION = PROC / "m6_seen_vs_unseen_curves.json"
DEFAULT_OUT_DIR = ROOT / "docs" / "reports" / "assets" / "m6" / "seen-vs-unseen"


def _stem(fold: str) -> str:
    return f"m6_site_transfer_b1_{fold}_{BUDGET}_seed{SEED}"


def _compressed(
    x: np.ndarray, y: np.ndarray, max_points: int = 1_200
) -> tuple[np.ndarray, np.ndarray]:
    """Thin a curve for the PNG only; every metric is computed on full scores."""
    if len(x) <= max_points:
        return x, y
    positions = np.unique(np.linspace(0, len(x) - 1, max_points).round().astype(int))
    return x[positions], y[positions]


def load_arm(folds: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Union the folds of one arm into whole-holdout arrays, sorted by row id."""
    index, site, label, score = [], [], [], []
    for fold in folds:
        payload = PROC / f"{_stem(fold)}.json"
        archive = PROC / f"{_stem(fold)}_predictions.npz"
        if not payload.exists() or not archive.exists():
            raise SystemExit(f"{_stem(fold)}: missing result JSON or predictions NPZ")
        status = json.loads(payload.read_text(encoding="utf-8")).get("status")
        if status != "completed":
            raise SystemExit(f"{_stem(fold)}: status is {status!r}, not completed")
        # One archive at a time: these are ~200-260 MB each.
        with np.load(archive) as d:
            index.append(d["validation_raw_index"].copy())
            site.append(d["site_id"].astype(np.int16))
            label.append(d["anomaly"].astype(np.int8))
            score.append(d["ensemble"].astype(np.float32))
        print(f"  loaded {_stem(fold)}")

    order = np.argsort(np.concatenate(index), kind="stable")
    return {
        "index": np.concatenate(index)[order],
        "site": np.concatenate(site)[order],
        "y": np.concatenate(label)[order],
        "score": np.concatenate(score)[order],
    }


def restrict_to(
    arm: dict[str, np.ndarray], keep_index: np.ndarray
) -> dict[str, np.ndarray]:
    """Cut an arm down to ``keep_index``, matching on row id alone.

    The seen arm holds out one building parity, so the unseen arm has to lose
    the other one. Selecting by ``validation_raw_index`` means the cut is the
    A0 fold's own definition of its test set; recomputing ``building_id % 2``
    here would be a second, independent definition that could silently drift.
    """
    both_sorted_ascending = np.all(np.diff(arm["index"]) > 0) and np.all(
        np.diff(keep_index) > 0
    )
    if not both_sorted_ascending:
        raise SystemExit("restrict_to needs both row-id arrays sorted and unique")

    position = np.searchsorted(arm["index"], keep_index)
    if position.size and position.max() >= arm["index"].size:
        raise SystemExit("rows to keep are not all present in the arm being cut")
    if not np.array_equal(arm["index"][position], keep_index):
        missing = int((arm["index"][position] != keep_index).sum())
        raise SystemExit(
            f"{missing:,} of {len(keep_index):,} rows to keep are absent from the "
            "arm being cut; the two cells do not share a row space"
        )
    return {key: value[position] for key, value in arm.items()}


def assert_same_rows(
    seen: dict[str, np.ndarray], unseen: dict[str, np.ndarray]
) -> None:
    """Refuse to share an axis unless both arms scored the very same rows."""
    if not np.array_equal(seen["index"], unseen["index"]):
        raise SystemExit(
            "seen and unseen arms do not cover the same evaluation rows "
            f"({len(seen['index']):,} vs {len(unseen['index']):,} after union)"
        )
    if not np.array_equal(seen["site"], unseen["site"]):
        raise SystemExit("seen and unseen arms disagree on site_id for shared rows")
    if not np.array_equal(seen["y"], unseen["y"]):
        raise SystemExit("seen and unseen arms disagree on labels for shared rows")
    observed = set(np.unique(seen["site"]).astype(int).tolist())
    if observed != set(range(16)):
        raise SystemExit(f"expected site_id 0..15, got {sorted(observed)}")


def curves(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    """ROC and PR curves plus their full-precision AUCs for one score vector."""
    fpr, tpr, _ = roc_curve(y, score)
    precision, recall, _ = precision_recall_curve(y, score)
    fpr, tpr = _compressed(fpr, tpr)
    recall, precision = _compressed(recall, precision)
    return {
        "roc": {"x": fpr, "y": tpr, "score": float(roc_auc_score(y, score))},
        "precision_recall": {
            "x": recall,
            "y": precision,
            "score": float(average_precision_score(y, score)),
        },
    }


def assert_matches_m3_figures(seen: dict[str, np.ndarray]) -> None:
    """The seen arm must be the M3 figures' holdout, not merely one like it."""
    stem = _stem(SEEN_FOLDS[0])
    recorded = json.loads((PROC / f"{stem}.json").read_text(encoding="utf-8"))
    ensemble = recorded["metrics"]["ensemble"]
    for name, expected, observed in (
        (
            "PR-AUC",
            ensemble["pr_auc"],
            average_precision_score(seen["y"], seen["score"]),
        ),
        ("ROC-AUC", ensemble["roc_auc"], roc_auc_score(seen["y"], seen["score"])),
    ):
        # The cell scored its rows in load order; this arm scores them sorted by
        # row id. Equal float32 scores therefore break ties in a different order,
        # which moves the AUC by ~1e-8. A different evaluation set moves it by
        # 1e-2 or more, so this tolerance still fails loudly on real drift.
        if abs(float(expected) - float(observed)) > 1e-6:
            raise SystemExit(
                f"seen arm {name} is {observed:.10f}, but {stem} recorded "
                f"{float(expected):.10f}; this is not the M3 figures' holdout"
            )
    print(
        f"  seen arm reproduces {stem}: "
        f"PR-AUC {float(ensemble['pr_auc']):.4f}, "
        f"ROC-AUC {float(ensemble['roc_auc']):.4f}"
    )


def observe() -> dict[str, Any]:
    print("seen arm (site in training, building held out):")
    seen = load_arm(SEEN_FOLDS)
    print("unseen arm (site held out):")
    unseen = load_arm(UNSEEN_FOLDS)

    # The unseen folds cover every building; keep only the ones the seen fold
    # actually scores, so both arms land on the M3 figures' evaluation rows.
    unseen = restrict_to(unseen, seen["index"])
    assert_same_rows(seen, unseen)
    print(f"  both arms cover {len(seen['index']):,} identical rows")
    assert_matches_m3_figures(seen)

    result: dict[str, Any] = {
        "pooled": {
            "rows": int(len(seen["y"])),
            "anomalies": int(seen["y"].sum()),
            "seen": curves(seen["y"], seen["score"]),
            "unseen": curves(unseen["y"], unseen["score"]),
        },
        "by_site": {},
    }
    for site in range(16):
        mask = seen["site"] == site
        y_site = seen["y"][mask]
        if len(np.unique(y_site)) != 2:
            raise SystemExit(f"site {site} does not contain both classes")
        result["by_site"][str(site)] = {
            "rows": int(mask.sum()),
            "anomalies": int(y_site.sum()),
            "seen": curves(y_site, seen["score"][mask]),
            "unseen": curves(y_site, unseen["score"][mask]),
        }
    return result


def _jsonable(data: dict[str, Any]) -> dict[str, Any]:
    """Store AUCs and support only; the curve vertices stay derivable."""

    def scores(block: dict[str, Any]) -> dict[str, Any]:
        return {
            arm: {
                "roc_auc": block[arm]["roc"]["score"],
                "pr_auc": block[arm]["precision_recall"]["score"],
            }
            for arm in ("seen", "unseen")
        }

    return {
        "schema_version": 1,
        "experiment": "m6_seen_vs_unseen_curves",
        "note": (
            "Tree Ensemble, 137 features, on the M3 report figures' own evaluation "
            "rows: the odd buildings A0 holds out. seen is the a0odd fold; unseen "
            "unions the a1 and a2 site folds and is then cut to a0odd's rows by "
            "validation_raw_index. Both are the metersall budget at seed 42, cover "
            "identical rows, and the seen arm reproduces the a0odd cell's recorded "
            "pooled metrics exactly."
        ),
        "source_cells": {
            "seen": [_stem(f) for f in SEEN_FOLDS],
            "unseen": [_stem(f) for f in UNSEEN_FOLDS],
        },
        "pooled": {
            "rows": data["pooled"]["rows"],
            "anomalies": data["pooled"]["anomalies"],
            **scores(data["pooled"]),
        },
        "by_site": {
            sid: {"rows": v["rows"], "anomalies": v["anomalies"], **scores(v)}
            for sid, v in data["by_site"].items()
        },
    }


def _style_pooled_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    ax.set_box_aspect(1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))


def _style_site_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=7, length=2.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_box_aspect(1)


def render_pooled(data: dict[str, Any], path: Path, *, curve_type: str) -> None:
    is_roc = curve_type == "roc"
    pooled = data["pooled"]
    score_name = "ROC-AUC" if is_roc else "PR-AUC"

    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor=SURFACE)
    for arm, color, label in (
        ("seen", SEEN_COLOR, "Site seen in training; buildings held out"),
        ("unseen", UNSEEN_COLOR, "Site unseen in training"),
    ):
        curve = pooled[arm][curve_type]
        ax.plot(
            curve["x"],
            curve["y"],
            color=color,
            linewidth=1.45,
            label=f"{label}  {score_name} = {curve['score']:.4f}",
        )
    _style_pooled_axis(ax)
    ax.set_xlabel(
        "False positive rate" if is_roc else "Recall", fontsize=10, color=SECONDARY
    )
    ax.set_ylabel(
        "True positive rate" if is_roc else "Precision", fontsize=10, color=SECONDARY
    )

    fig.suptitle(
        f"Site-Transfer Cost: {'ROC' if is_roc else 'Precision–Recall'}",
        x=0.06,
        y=0.965,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.895,
        "137-feature Tree Ensemble · same final 50/50 building holdout · "
        "site seen versus unseen",
        ha="left",
        fontsize=10.5,
        color=SECONDARY,
    )
    ax.legend(
        frameon=False,
        fontsize=8.8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        handlelength=2.4,
        labelcolor=SECONDARY,
    )
    fig.subplots_adjust(left=0.14, right=0.96, top=0.78, bottom=0.24)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)


def render_by_site(data: dict[str, Any], path: Path, *, curve_type: str) -> None:
    is_roc = curve_type == "roc"
    names = site_names()
    pooled = data["pooled"]
    score_name = "ROC-AUC" if is_roc else "PR-AUC"

    fig, axes = plt.subplots(4, 4, figsize=(9.6, 11.5), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for index, ax in enumerate(axes.ravel()):
        row, col = divmod(index, 4)
        site = data["by_site"][str(index)]
        _style_site_axis(ax)

        # Same "before -> after" shape the M3 by-site figures use, so the two
        # sets can be read side by side; here the arrow runs seen -> unseen, and
        # the subtitle says so because the direction is a loss, not a gain.
        if is_roc:
            detail = (
                f"{score_name} {site['seen']['roc']['score']:.3f} -> "
                f"{site['unseen']['roc']['score']:.3f}"
            )
        else:
            prevalence = site["anomalies"] / site["rows"]
            ax.axhline(prevalence, color=AXIS, linewidth=0.7, linestyle=(0, (2, 3)))
            detail = (
                f"{score_name} {site['seen']['precision_recall']['score']:.3f} -> "
                f"{site['unseen']['precision_recall']['score']:.3f}\n"
                f"Prevalence {prevalence:.1%}"
            )

        for arm, color in (("seen", SEEN_COLOR), ("unseen", UNSEEN_COLOR)):
            curve = site[arm][curve_type]
            ax.plot(curve["x"], curve["y"], color=color, linewidth=1.35)

        ax.set_title(
            f"{names[index]} (site {index})",
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
            pad=20 if is_roc else 29,
        )
        ax.text(
            0,
            1.03,
            detail,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            color=MUTED,
        )
        if row == 3:
            ax.set_xlabel(
                "False-positive rate" if is_roc else "Recall",
                fontsize=8.5,
                color=SECONDARY,
            )
        if col == 0:
            ax.set_ylabel(
                "True-positive rate" if is_roc else "Precision",
                fontsize=8.5,
                color=SECONDARY,
            )

    pooled_seen = pooled["seen"][curve_type]["score"]
    pooled_unseen = pooled["unseen"][curve_type]["score"]
    fig.suptitle(
        f"Site-transfer cost by site: {'ROC' if is_roc else 'Precision-Recall'}",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.944,
        f"Blue: site seen in training (pooled {score_name} {pooled_seen:.4f}); "
        f"orange: site unseen (pooled {score_name} {pooled_unseen:.4f}).",
        ha="left",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.text(
        0.06,
        0.925,
        "Panel numbers read seen -> unseen; both arms score the same final 50/50 "
        f"building holdout ({pooled['rows']:,} rows).",
        ha="left",
        fontsize=10.5,
        color=SECONDARY,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.86,
        bottom=0.095,
        wspace=0.28,
        hspace=0.58,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=SEEN_COLOR,
                linewidth=1.8,
                label="Site seen in training; buildings held out",
            ),
            Line2D(
                [0],
                [0],
                color=UNSEEN_COLOR,
                linewidth=1.8,
                label="Site unseen in training",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        labelcolor=SECONDARY,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = observe()
    OBSERVATION.write_text(json.dumps(_jsonable(data), indent=2), encoding="utf-8")
    print(f"wrote {OBSERVATION.relative_to(ROOT)}")

    outputs = [
        (
            render_pooled,
            "precision_recall",
            args.out_dir / "m6_seen_vs_unseen_pooled_precision_recall.png",
        ),
        (render_pooled, "roc", args.out_dir / "m6_seen_vs_unseen_pooled_roc.png"),
        (
            render_by_site,
            "precision_recall",
            args.out_dir / "m6_seen_vs_unseen_by_site_precision_recall.png",
        ),
        (render_by_site, "roc", args.out_dir / "m6_seen_vs_unseen_by_site_roc.png"),
    ]
    for renderer, curve_type, path in outputs:
        renderer(data, path, curve_type=curve_type)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
