"""Render the M3 17-versus-137 Tree Ensemble curves separately by meter."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from lead import PROC, write_json_with_provenance
from plot_m3_figures import render_discrimination_curve


METER_SPECS = (
    (0, "electricity", "Electricity"),
    (1, "chilledwater", "Chilled Water"),
    (2, "steam", "Steam"),
    (3, "hotwater", "Hot Water"),
)
BASELINE_KEY = "m3_1_ensemble"
ENGINEERED_KEY = "ensemble"


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    required = {"anomaly", "meter", "row_identity", "ensemble"}
    with np.load(path) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        return {name: np.asarray(payload[name]) for name in required}


def load_aligned_ensembles(
    baseline_path: Path,
    engineered_path: Path,
) -> dict[str, np.ndarray]:
    """Load both ensembles only when labels and per-row meter identity agree."""
    baseline = _load_arrays(baseline_path)
    engineered = _load_arrays(engineered_path)
    for key in ("anomaly", "meter", "row_identity"):
        if not np.array_equal(baseline[key], engineered[key]):
            raise ValueError(f"17- and 137-feature artifacts do not align: {key}")
    if not np.isfinite(baseline["ensemble"]).all():
        raise ValueError("17-feature ensemble contains non-finite scores")
    if not np.isfinite(engineered["ensemble"]).all():
        raise ValueError("137-feature ensemble contains non-finite scores")
    return {
        "anomaly": baseline["anomaly"].astype("int8", copy=False),
        "meter": baseline["meter"].astype("int8", copy=False),
        BASELINE_KEY: baseline["ensemble"],
        ENGINEERED_KEY: engineered["ensemble"],
    }


def meter_curve_data(
    arrays: dict[str, np.ndarray],
    meter: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute the existing renderer's ROC and PR inputs for one meter."""
    mask = arrays["meter"] == meter
    labels = arrays["anomaly"][mask]
    if len(labels) == 0 or np.unique(labels).size != 2:
        raise ValueError(f"meter {meter} requires both label classes")
    metrics: dict[str, dict[str, float]] = {}
    curves: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for key in (BASELINE_KEY, ENGINEERED_KEY):
        scores = arrays[key][mask]
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        metrics[key] = {
            "roc_auc": float(roc_auc_score(labels, scores)),
            "pr_auc": float(average_precision_score(labels, scores)),
        }
        curves[key] = {
            "roc": {"x": fpr, "y": tpr},
            "precision_recall": {"x": recall, "y": precision},
        }
    data = {"metrics": metrics, "curves": curves}
    summary = {
        "rows": int(mask.sum()),
        "anomalies": int(labels.sum()),
        "anomaly_rate": float(labels.mean()),
        "metrics": metrics,
    }
    return data, summary


def render_all(
    arrays: dict[str, np.ndarray],
    asset_dir: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}
    summaries: dict[str, Any] = {}
    for meter, slug, label in METER_SPECS:
        data, summary = meter_curve_data(arrays, meter)
        subtitle = (
            f"Tree Ensemble · {label} · final 50/50 building holdout · "
            "17 versus 137 features"
        )
        series = [
            (BASELINE_KEY, "17 baseline features", "#898781"),
            (ENGINEERED_KEY, "137 features", "#2a78d6"),
        ]
        for curve_type, suffix in (
            ("precision_recall", "precision_recall"),
            ("roc", "roc"),
        ):
            output = asset_dir / f"m3_feature_engineering_{slug}_{suffix}.png"
            render_discrimination_curve(
                data,
                output,
                comparison="feature_engineering",
                curve_type=curve_type,
                custom_series=series,
                custom_title_prefix="Feature-Engineering Contribution",
                custom_subtitle=subtitle,
            )
            figures[f"{slug}_{suffix}"] = output
        summaries[slug] = summary
    return figures, summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=PROC / "m3_17_feature_ensemble_predictions_50_50.npz",
    )
    parser.add_argument(
        "--engineered-predictions",
        type=Path,
        default=PROC / "m3_137_feature_ensemble_predictions_50_50.npz",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=ROOT / "docs" / "reports" / "assets" / "m3",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=ROOT / "docs" / "metrics" / "m3_tree_ensemble_by_meter.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = load_aligned_ensembles(
        args.baseline_predictions,
        args.engineered_predictions,
    )
    figures, summaries = render_all(arrays, args.asset_dir)
    write_json_with_provenance(
        args.metrics_out,
        {
            "experiment": "m3_tree_ensemble_feature_engineering_by_meter",
            "split": "50_50_mod2",
            "feature_comparison": "equal-weight four-model Tree Ensemble, 17 versus 137 features",
            "artifacts": {
                "baseline_predictions": str(
                    args.baseline_predictions.relative_to(ROOT)
                ),
                "engineered_predictions": str(
                    args.engineered_predictions.relative_to(ROOT)
                ),
                "figures": {
                    name: str(path.relative_to(ROOT)) for name, path in figures.items()
                },
            },
            "meters": summaries,
        },
        root=ROOT,
        provenance={"note": "Curves use the existing M3 discrimination renderer."},
    )
    print(f"Saved {len(figures)} figures and {args.metrics_out}")


if __name__ == "__main__":
    main()
