"""Plot the canonical M3 Tree Ensemble ROC and PR curves by site.

M3's prediction artifact contains aligned labels and ensemble scores, but its
``validation_raw_index`` cannot recover site membership.  M6 A0 reran the
identical M3 50/50 building holdout and preserved the correctly aligned
``site_id``.  This renderer proves row-for-row prediction identity before
combining the artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

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
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BASELINE = "#898781"
ENGINEERED = "#2a78d6"
TABPFN = "#d1580f"

DEFAULT_M3_PREDICTIONS = PROC / "m3_figure_predictions_50_50.npz"
DEFAULT_SITE_PREDICTIONS = (
    PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"
)
DEFAULT_BASELINE_PREDICTIONS = PROC / "m3_17_feature_ensemble_predictions.npz"
DEFAULT_TABPFN_PREDICTIONS = (
    PROC / "m5_tabpfn_distributed_context100000_predictions.npz"
)
DEFAULT_ROC_OUTPUT = (
    ROOT / "docs" / "reports" / "assets" / "m3" / "m3_tree_ensemble_by_site_roc.png"
)
DEFAULT_PR_OUTPUT = (
    ROOT
    / "docs"
    / "reports"
    / "assets"
    / "m3"
    / "m3_tree_ensemble_by_site_precision_recall.png"
)
DEFAULT_TABPFN_ROC_OUTPUT = (
    ROOT
    / "docs"
    / "reports"
    / "assets"
    / "m3"
    / "m3_tree_ensemble_by_site_roc_with_tabpfn.png"
)
DEFAULT_TABPFN_PR_OUTPUT = (
    ROOT
    / "docs"
    / "reports"
    / "assets"
    / "m3"
    / "m3_tree_ensemble_by_site_precision_recall_with_tabpfn.png"
)


@dataclass(frozen=True)
class Curve:
    x: np.ndarray
    y: np.ndarray
    score: float


@dataclass(frozen=True)
class SiteResult:
    site_id: int
    rows: int
    anomalies: int
    baseline_roc: Curve
    engineered_roc: Curve
    baseline_precision_recall: Curve
    engineered_precision_recall: Curve
    tabpfn_roc: Curve | None = None
    tabpfn_precision_recall: Curve | None = None


def _compressed(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int = 1_200,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    positions = np.linspace(0, len(x) - 1, max_points).round().astype(int)
    positions = np.unique(positions)
    return x[positions], y[positions]


def load_aligned_arrays(
    m3_predictions: Path,
    site_predictions: Path,
    baseline_predictions: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the ensemble and prove A0 supplies its matching site order."""
    with np.load(m3_predictions) as m3, np.load(site_predictions) as a0:
        required_m3 = {"anomaly", "ensemble"}
        required_a0 = {"anomaly", "ensemble", "site_id"}
        missing_m3 = required_m3 - set(m3.files)
        missing_a0 = required_a0 - set(a0.files)
        if missing_m3 or missing_a0:
            raise ValueError(
                f"missing NPZ arrays: M3={sorted(missing_m3)}, A0={sorted(missing_a0)}"
            )

        y_true = np.asarray(m3["anomaly"])
        ensemble = np.asarray(m3["ensemble"])
        site_id = np.asarray(a0["site_id"])
        if not (len(y_true) == len(ensemble) == len(site_id)):
            raise ValueError("M3 and A0 prediction arrays have different lengths")
        if not np.array_equal(y_true, a0["anomaly"]):
            raise ValueError("M3 and A0 labels are not row-for-row aligned")
        if not np.array_equal(ensemble, a0["ensemble"]):
            raise ValueError("M3 and A0 ensemble scores are not row-for-row aligned")

    with np.load(baseline_predictions) as baseline:
        required_baseline = {"anomaly", "ensemble", "site_id"}
        missing_baseline = required_baseline - set(baseline.files)
        if missing_baseline:
            raise ValueError(f"missing baseline NPZ arrays: {sorted(missing_baseline)}")
        baseline_y = np.asarray(baseline["anomaly"])
        baseline_ensemble = np.asarray(baseline["ensemble"])
        baseline_site_id = np.asarray(baseline["site_id"])
    if not np.array_equal(y_true, baseline_y):
        raise ValueError("17- and 137-feature labels are not row-for-row aligned")
    if not np.array_equal(site_id, baseline_site_id):
        raise ValueError("17- and 137-feature site IDs are not row-for-row aligned")

    observed_sites = set(np.unique(site_id).astype(int).tolist())
    if observed_sites != set(range(16)):
        raise ValueError(f"expected site_id 0..15, got {sorted(observed_sites)}")
    return y_true, baseline_ensemble, ensemble, site_id


def load_tabpfn_scores(
    tabpfn_predictions: Path,
    y_true: np.ndarray,
    site_id: np.ndarray,
) -> np.ndarray:
    """Load merged TabPFN scores and prove they share the plotted row order.

    The distributed merge already checks every 20k span against the canonical
    artifact, but this renderer combines three separate NPZ files, so it repeats
    the identity proof rather than trusting the file name.
    """
    with np.load(tabpfn_predictions) as tabpfn:
        required = {"anomaly", "tabpfn", "site_id"}
        missing = required - set(tabpfn.files)
        if missing:
            raise ValueError(f"missing TabPFN NPZ arrays: {sorted(missing)}")
        score = np.asarray(tabpfn["tabpfn"])
        if len(score) != len(y_true):
            raise ValueError("TabPFN and M3 prediction arrays have different lengths")
        if not np.array_equal(y_true, np.asarray(tabpfn["anomaly"])):
            raise ValueError("TabPFN and M3 labels are not row-for-row aligned")
        if not np.array_equal(site_id, np.asarray(tabpfn["site_id"])):
            raise ValueError("TabPFN and M3 site IDs are not row-for-row aligned")
    if not np.isfinite(score).all():
        raise ValueError("TabPFN scores contain non-finite values")
    return score


def compute_site_results(
    y_true: np.ndarray,
    baseline_ensemble: np.ndarray,
    engineered_ensemble: np.ndarray,
    site_id: np.ndarray,
    tabpfn: np.ndarray | None = None,
) -> list[SiteResult]:
    results: list[SiteResult] = []
    for value in range(16):
        mask = site_id == value
        y_site = y_true[mask]
        baseline_site = baseline_ensemble[mask]
        engineered_site = engineered_ensemble[mask]
        if len(np.unique(y_site)) != 2:
            raise ValueError(f"site {value} does not contain both classes")

        tabpfn_roc: Curve | None = None
        tabpfn_precision_recall: Curve | None = None
        if tabpfn is not None:
            tabpfn_site = tabpfn[mask]
            tabpfn_fpr, tabpfn_tpr, _ = roc_curve(y_site, tabpfn_site)
            tabpfn_fpr, tabpfn_tpr = _compressed(tabpfn_fpr, tabpfn_tpr)
            tabpfn_prec, tabpfn_rec, _ = precision_recall_curve(y_site, tabpfn_site)
            tabpfn_rec, tabpfn_prec = _compressed(tabpfn_rec, tabpfn_prec)
            tabpfn_roc = Curve(
                tabpfn_fpr,
                tabpfn_tpr,
                float(roc_auc_score(y_site, tabpfn_site)),
            )
            tabpfn_precision_recall = Curve(
                tabpfn_rec,
                tabpfn_prec,
                float(average_precision_score(y_site, tabpfn_site)),
            )

        baseline_fpr, baseline_tpr, _ = roc_curve(y_site, baseline_site)
        baseline_fpr, baseline_tpr = _compressed(baseline_fpr, baseline_tpr)
        engineered_fpr, engineered_tpr, _ = roc_curve(y_site, engineered_site)
        engineered_fpr, engineered_tpr = _compressed(
            engineered_fpr,
            engineered_tpr,
        )
        baseline_precision, baseline_recall, _ = precision_recall_curve(
            y_site,
            baseline_site,
        )
        baseline_recall, baseline_precision = _compressed(
            baseline_recall,
            baseline_precision,
        )
        engineered_precision, engineered_recall, _ = precision_recall_curve(
            y_site,
            engineered_site,
        )
        engineered_recall, engineered_precision = _compressed(
            engineered_recall,
            engineered_precision,
        )
        results.append(
            SiteResult(
                site_id=value,
                rows=int(mask.sum()),
                anomalies=int(y_site.sum()),
                baseline_roc=Curve(
                    baseline_fpr,
                    baseline_tpr,
                    float(roc_auc_score(y_site, baseline_site)),
                ),
                engineered_roc=Curve(
                    engineered_fpr,
                    engineered_tpr,
                    float(roc_auc_score(y_site, engineered_site)),
                ),
                baseline_precision_recall=Curve(
                    baseline_recall,
                    baseline_precision,
                    float(average_precision_score(y_site, baseline_site)),
                ),
                engineered_precision_recall=Curve(
                    engineered_recall,
                    engineered_precision,
                    float(average_precision_score(y_site, engineered_site)),
                ),
                tabpfn_roc=tabpfn_roc,
                tabpfn_precision_recall=tabpfn_precision_recall,
            )
        )
    return results


def _style_axis(ax: plt.Axes) -> None:
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
    ax.grid(color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_box_aspect(1)


def render(
    results: list[SiteResult],
    output: Path,
    *,
    curve_type: str,
    include_tabpfn: bool = False,
) -> None:
    if curve_type not in {"roc", "precision_recall"}:
        raise ValueError(f"unsupported curve type: {curve_type}")

    names = site_names()
    fig, axes = plt.subplots(4, 4, figsize=(9.6, 11.5), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for index, (ax, result) in enumerate(zip(axes.ravel(), results, strict=True)):
        row, col = divmod(index, 4)
        _style_axis(ax)
        if curve_type == "roc":
            baseline_curve = result.baseline_roc
            engineered_curve = result.engineered_roc
            tabpfn_curve = result.tabpfn_roc
            ax.plot(
                [0, 1],
                [0, 1],
                color=AXIS,
                linewidth=0.7,
                linestyle=(0, (2, 3)),
            )
            metric_detail = (
                f"ROC-AUC {baseline_curve.score:.3f} -> {engineered_curve.score:.3f}"
            )
        else:
            baseline_curve = result.baseline_precision_recall
            engineered_curve = result.engineered_precision_recall
            tabpfn_curve = result.tabpfn_precision_recall
            prevalence = result.anomalies / result.rows
            ax.axhline(
                prevalence,
                color=AXIS,
                linewidth=0.7,
                linestyle=(0, (2, 3)),
            )
            metric_detail = (
                f"PR-AUC {baseline_curve.score:.3f} -> "
                f"{engineered_curve.score:.3f}\nPrevalence {prevalence:.1%}"
            )
        ax.plot(
            baseline_curve.x,
            baseline_curve.y,
            color=BASELINE,
            linewidth=1.15,
        )
        ax.plot(
            engineered_curve.x,
            engineered_curve.y,
            color=ENGINEERED,
            linewidth=1.45,
        )
        if include_tabpfn:
            if tabpfn_curve is None:
                raise ValueError(f"site {result.site_id} has no TabPFN curve")
            ax.plot(
                tabpfn_curve.x,
                tabpfn_curve.y,
                color=TABPFN,
                linewidth=1.45,
            )
        ax.set_title(
            f"{names[result.site_id]} (site {result.site_id})",
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
            pad=29 if curve_type == "precision_recall" else 20,
        )
        ax.text(
            0,
            1.03,
            metric_detail,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            color=MUTED,
        )
        if row == 3:
            ax.set_xlabel(
                "False-positive rate" if curve_type == "roc" else "Recall",
                fontsize=8.5,
                color=SECONDARY,
            )
        if col == 0:
            ax.set_ylabel(
                "True-positive rate" if curve_type == "roc" else "Precision",
                fontsize=8.5,
                color=SECONDARY,
            )

    title_metric = "ROC" if curve_type == "roc" else "Precision-Recall"
    subtitle = (
        "Gray: 17 features (pooled ROC-AUC 0.9663); "
        "blue: 137 features (pooled ROC-AUC 0.9918)."
        if curve_type == "roc"
        else "Gray: 17 features (pooled PR-AUC 0.8221); "
        "blue: 137 features (pooled PR-AUC 0.9303)."
    )
    fig.suptitle(
        f"Tree Ensemble by site: {title_metric}",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.94,
        subtitle,
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
    handles = [
        Line2D(
            [0],
            [0],
            color=BASELINE,
            linewidth=1.6,
            label="17-feature Tree Ensemble",
        ),
        Line2D(
            [0],
            [0],
            color=ENGINEERED,
            linewidth=1.8,
            label="137-feature Tree Ensemble",
        ),
    ]
    if include_tabpfn:
        handles.append(
            Line2D(
                [0],
                [0],
                color=TABPFN,
                linewidth=1.8,
                label="TabPFN (17 features, context 100k)",
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=len(handles),
        frameon=False,
        # Three entries need the extra width a slightly smaller face buys.
        fontsize=10.5 if len(handles) == 2 else 9.6,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-predictions", type=Path, default=DEFAULT_M3_PREDICTIONS)
    parser.add_argument(
        "--site-predictions", type=Path, default=DEFAULT_SITE_PREDICTIONS
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=DEFAULT_BASELINE_PREDICTIONS,
    )
    parser.add_argument("--roc-output", type=Path, default=DEFAULT_ROC_OUTPUT)
    parser.add_argument("--pr-output", type=Path, default=DEFAULT_PR_OUTPUT)
    parser.add_argument(
        "--tabpfn-predictions",
        type=Path,
        default=DEFAULT_TABPFN_PREDICTIONS,
    )
    parser.add_argument(
        "--tabpfn-roc-output",
        type=Path,
        default=DEFAULT_TABPFN_ROC_OUTPUT,
    )
    parser.add_argument(
        "--tabpfn-pr-output",
        type=Path,
        default=DEFAULT_TABPFN_PR_OUTPUT,
    )
    parser.add_argument(
        "--skip-tabpfn",
        action="store_true",
        help="render only the two-series figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    y_true, baseline_ensemble, engineered_ensemble, site_id = load_aligned_arrays(
        args.m3_predictions,
        args.site_predictions,
        args.baseline_predictions,
    )
    results = compute_site_results(
        y_true,
        baseline_ensemble,
        engineered_ensemble,
        site_id,
    )
    render(results, args.roc_output, curve_type="roc")
    render(results, args.pr_output, curve_type="precision_recall")
    print(f"Saved {args.roc_output}")
    print(f"Saved {args.pr_output}")

    if args.skip_tabpfn:
        return
    tabpfn = load_tabpfn_scores(args.tabpfn_predictions, y_true, site_id)
    tabpfn_results = compute_site_results(
        y_true,
        baseline_ensemble,
        engineered_ensemble,
        site_id,
        tabpfn=tabpfn,
    )
    render(
        tabpfn_results,
        args.tabpfn_roc_output,
        curve_type="roc",
        include_tabpfn=True,
    )
    render(
        tabpfn_results,
        args.tabpfn_pr_output,
        curve_type="precision_recall",
        include_tabpfn=True,
    )
    print(f"Saved {args.tabpfn_roc_output}")
    print(f"Saved {args.tabpfn_pr_output}")


if __name__ == "__main__":
    main()
