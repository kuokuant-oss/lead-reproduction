"""Render pooled and by-site ROC/PR curves from saved TabPFN scoring rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
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

from lead import PROC, ROOT
from m6_site_names import site_names


SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
DEFAULT_SUMMARY = PROC / "m5_tabpfn_single_context_scaling.json"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "reports" / "assets" / "m5" / "tabpfn"


@dataclass(frozen=True)
class Predictions:
    budget: int
    y: np.ndarray
    score: np.ndarray
    site_id: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(summary_path: Path) -> list[Predictions]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    loaded: list[Predictions] = []
    for budget_text, result in summary.get("budget_results", {}).items():
        if result.get("status") != "completed":
            continue
        metadata = result.get("prediction_artifact", {})
        path = Path(metadata.get("path", ""))
        if not path.is_file():
            raise FileNotFoundError(
                f"prediction artifact missing for {budget_text}: {path}"
            )
        if metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"prediction artifact hash mismatch for {budget_text}")
        with np.load(path) as artifact:
            required = {"test_y", "test_score", "test_site_id"}
            missing = required - set(artifact.files)
            if missing:
                raise ValueError(
                    f"prediction artifact {path} missing {sorted(missing)}"
                )
            y = np.asarray(artifact["test_y"], dtype="int8")
            score = np.asarray(artifact["test_score"], dtype="float32")
            site_id = np.asarray(artifact["test_site_id"], dtype="int8")
        if not (len(y) == len(score) == len(site_id)) or not len(y):
            raise ValueError(f"unaligned or empty prediction artifact: {path}")
        if not set(np.unique(y)).issubset({0, 1}) or not np.isfinite(score).all():
            raise ValueError(f"invalid labels or scores in {path}")
        loaded.append(Predictions(int(budget_text), y, score, site_id))
    if not loaded:
        raise ValueError("summary contains no completed budgets with predictions")
    return sorted(loaded, key=lambda item: item.budget)


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY)
    ax.grid(color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)


def _curve(y: np.ndarray, score: np.ndarray, curve_type: str) -> tuple[Any, ...]:
    if curve_type == "roc":
        x, values, _ = roc_curve(y, score)
        return x, values, roc_auc_score(y, score)
    precision, recall, _ = precision_recall_curve(y, score)
    return recall, precision, average_precision_score(y, score)


def render_pooled(
    predictions: list[Predictions], output: Path, *, curve_type: str
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor=SURFACE)
    colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(predictions)))
    for item, color in zip(predictions, colors, strict=True):
        if len(np.unique(item.y)) != 2:
            raise ValueError(f"budget {item.budget} test sample lacks both classes")
        x, y, score = _curve(item.y, item.score, curve_type)
        metric = "ROC-AUC" if curve_type == "roc" else "PR-AUC"
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.5,
            label=f"{item.budget:,}  {metric}={score:.4f}",
        )
    _style_axis(ax)
    ax.set_box_aspect(1)
    ax.set_xlabel("False-positive rate" if curve_type == "roc" else "Recall")
    ax.set_ylabel("True-positive rate" if curve_type == "roc" else "Precision")
    title = "ROC" if curve_type == "roc" else "Precision–Recall"
    fig.suptitle(
        f"TabPFN Single-Context Scaling: {title}",
        x=0.08,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.08,
        0.91,
        "Same fixed natural-prevalence test rows · raw 17 features",
        color=SECONDARY,
        fontsize=10,
    )
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.14, right=0.96, top=0.82, bottom=0.24)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)


def render_by_site(item: Predictions, output: Path, *, curve_type: str) -> list[int]:
    names = site_names()
    fig, axes = plt.subplots(4, 4, figsize=(9.6, 11.5), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    unavailable: list[int] = []
    for site, ax in enumerate(axes.ravel()):
        _style_axis(ax)
        mask = item.site_id == site
        y_site, score_site = item.y[mask], item.score[mask]
        ax.set_title(
            f"{names[site]} (site {site})",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
        )
        if len(y_site) and len(np.unique(y_site)) == 2:
            x, y, score = _curve(y_site, score_site, curve_type)
            ax.plot(x, y, color=BLUE, linewidth=1.45)
            metric = "ROC-AUC" if curve_type == "roc" else "PR-AUC"
            ax.text(
                0,
                1.03,
                f"{metric} {score:.3f} · n={len(y_site):,}",
                transform=ax.transAxes,
                fontsize=7.5,
                color=MUTED,
            )
            if curve_type != "roc":
                ax.axhline(
                    float(y_site.mean()),
                    color=AXIS,
                    linewidth=0.7,
                    linestyle=(0, (2, 3)),
                )
        else:
            unavailable.append(site)
            ax.text(
                0.5,
                0.5,
                "Not estimable\n(sample lacks both classes)",
                ha="center",
                va="center",
                fontsize=7.5,
                color=MUTED,
            )
    title = "ROC" if curve_type == "roc" else "Precision–Recall"
    fig.suptitle(
        f"TabPFN {item.budget:,}-row Context by Site: {title}",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.945,
        "Curves use only saved fixed test-scoring rows",
        color=SECONDARY,
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.88, bottom=0.06, wspace=0.25, hspace=0.35
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    return unavailable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.summary)
    outputs = {
        "pooled_roc": args.output_dir / "m5_tabpfn_context_scaling_roc.png",
        "pooled_pr": args.output_dir / "m5_tabpfn_context_scaling_precision_recall.png",
        "site_roc": args.output_dir / "m5_tabpfn_by_site_roc.png",
        "site_pr": args.output_dir / "m5_tabpfn_by_site_precision_recall.png",
    }
    render_pooled(predictions, outputs["pooled_roc"], curve_type="roc")
    render_pooled(predictions, outputs["pooled_pr"], curve_type="precision_recall")
    unavailable_roc = render_by_site(
        predictions[-1], outputs["site_roc"], curve_type="roc"
    )
    unavailable_pr = render_by_site(
        predictions[-1], outputs["site_pr"], curve_type="precision_recall"
    )
    manifest = {
        "summary": str(args.summary.resolve()),
        "largest_completed_budget": predictions[-1].budget,
        "outputs": {key: str(value.resolve()) for key, value in outputs.items()},
        "sites_not_estimable": sorted(set(unavailable_roc + unavailable_pr)),
    }
    manifest_path = args.output_dir / "m5_tabpfn_curve_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
