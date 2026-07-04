"""Run the canonical M3.2 LightGBM baseline with timestamp-merge value-change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lead import PROC, load_m3_frame
from run_m4_3_timestamp_value_change import fit_m3_2_regime


VALUE_CHANGE_REGIME = "timestamp_merge"
EXPECTED_TIMESTAMP_MERGE_AUC = 0.99248
NOISE_FLOOR_AUC = 0.0005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "m3_2_results.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_m3_frame()
    mask_val = (df["building_id"] % 5 == 4).to_numpy()
    metrics = fit_m3_2_regime(
        df,
        mask_val,
        VALUE_CHANGE_REGIME,
        include_feature_importance=True,
    )

    val_auc = float(metrics["val_auc"])
    delta_vs_expected = val_auc - EXPECTED_TIMESTAMP_MERGE_AUC
    if abs(delta_vs_expected) > NOISE_FLOOR_AUC:
        raise RuntimeError(
            "M3.2 timestamp_merge AUC deviates from run_m4_3 reference: "
            f"{val_auc:.6f} vs {EXPECTED_TIMESTAMP_MERGE_AUC:.6f} "
            f"(delta {delta_vs_expected:+.6f})"
        )

    val_auc_m31 = 0.9562
    results = {
        "val_auc_m31": val_auc_m31,
        "val_auc_m32": val_auc,
        "delta_auc": float(val_auc - val_auc_m31),
        "n_baseline_features": 17,
        "n_value_change_features": int(metrics["n_value_change_features"]),
        "n_total_features": int(metrics["n_features"]),
        "n_train_downsampled": int(metrics["n_train_downsampled"]),
        "top_feature": metrics["top_feature"],
        "top_value_change_feature": metrics["top_value_change_feature"],
        "value_change_importance_pct": metrics["value_change_importance_pct"],
        "provenance": {
            "value_change_regime": VALUE_CHANGE_REGIME,
            "source_pipeline": "scripts/run_m4_3_timestamp_value_change.py::fit_m3_2_regime",
            "gate_reference_auc": EXPECTED_TIMESTAMP_MERGE_AUC,
            "noise_floor_auc": NOISE_FLOOR_AUC,
            "delta_vs_gate_reference": delta_vs_expected,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
