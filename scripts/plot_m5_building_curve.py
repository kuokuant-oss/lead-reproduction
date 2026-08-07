"""Plot M5 building-count scaling and per-group ROC/PR prediction curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lead import PROC, ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROC / "m5_building_curve" / "aggregate" / "metrics.csv",
    )
    parser.add_argument(
        "--curves",
        type=Path,
        default=PROC / "m5_building_curve" / "aggregate" / "curves.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "docs" / "reports" / "assets" / "m5"
    )
    parser.add_argument("--profile", default="site_stratified_random")
    parser.add_argument("--features", type=int, default=137)
    parser.add_argument(
        "--grouping", choices=("overall", "meter", "site"), default="overall"
    )
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--building-budget", type=int, default=100)
    return parser.parse_args()


def plot_scaling(metrics: pd.DataFrame, args: argparse.Namespace) -> Path:
    selected = metrics[
        (metrics["sampling_profile"] == args.profile)
        & (metrics["features"] == args.features)
        & (metrics["grouping"] == args.grouping)
        & (metrics["group"] == args.group)
    ]
    if selected.empty:
        raise ValueError("no metrics match the requested profile/features/group")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for model, rows in selected.groupby("model", sort=True):
        rows = rows.sort_values("building_budget")
        axes[0].plot(rows["building_budget"], rows["pr_auc"], marker="o", label=model)
        axes[1].plot(rows["building_budget"], rows["roc_auc"], marker="o", label=model)
    axes[0].set_ylabel("PR-AUC")
    axes[1].set_ylabel("ROC-AUC")
    for axis in axes:
        axis.set_xlabel("Available even-building sources (K)")
        axis.grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    label = selected.iloc[0]["group_label"]
    fig.suptitle(f"Building-count scaling — {label}, {args.features} features")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / (
        f"m5_building_scaling_{args.profile}_f{args.features}_"
        f"{args.grouping}{args.group}.png"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_prediction_curves(curves: pd.DataFrame, args: argparse.Namespace) -> Path:
    selected = curves[
        (curves["sampling_profile"] == args.profile)
        & (curves["features"] == args.features)
        & (curves["grouping"] == args.grouping)
        & (curves["group"] == args.group)
        & (curves["building_budget"] == args.building_budget)
    ]
    if selected.empty:
        raise ValueError("no curve points match the requested cell")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for model, model_rows in selected.groupby("model", sort=True):
        roc = model_rows[model_rows["curve"] == "roc"].sort_values("point")
        pr = model_rows[model_rows["curve"] == "precision_recall"].sort_values("point")
        axes[0].plot(roc["x"], roc["y"], label=model)
        axes[1].plot(pr["x"], pr["y"], label=model)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    label = selected.iloc[0]["group_label"]
    fig.suptitle(
        f"Prediction curves — K={args.building_budget}, {label}, {args.features} features"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / (
        f"m5_prediction_curves_{args.profile}_k{args.building_budget}_"
        f"f{args.features}_{args.grouping}{args.group}.png"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    scaling = plot_scaling(pd.read_csv(args.metrics), args)
    prediction = plot_prediction_curves(pd.read_csv(args.curves), args)
    print(f"Wrote {scaling}\nWrote {prediction}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
