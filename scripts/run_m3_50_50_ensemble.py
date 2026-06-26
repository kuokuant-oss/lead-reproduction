"""Run PI-spec 50/50 4-model ensemble in offline and causal regimes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from lead import (
    BASELINE_FEATURE_COLS,
    DOWNSAMPLE_SEEDS,
    PROC,
    RANDOM_STATE,
    SHIFTS,
    add_value_change_features,
    assert_no_building_overlap,
    downsample_indices,
    load_m3_frame,
)
from run_m3_4_ensemble import (
    fit_predict_models,
    log,
)


PAST_SHIFTS = [n for n in SHIFTS if n > 0]
FUTURE_SHIFTS = [n for n in SHIFTS if n < 0]
M3_4_80_20_OFFLINE_ENSEMBLE_AUC = 0.9928


def run_regime(
    train_split: pd.DataFrame,
    val_split: pd.DataFrame,
    value_cols: list[str],
    regime: str,
) -> dict[str, object]:
    if regime == "offline":
        feature_cols = BASELINE_FEATURE_COLS + value_cols
    elif regime == "causal":
        past_cols = [c for c in value_cols if "_-" not in c]
        feature_cols = BASELINE_FEATURE_COLS + past_cols
    else:
        raise ValueError(f"Unknown regime: {regime}")

    expected_features = 137 if regime == "offline" else 77
    if len(feature_cols) != expected_features:
        raise AssertionError(
            f"{regime} expected {expected_features} features, got {len(feature_cols)}"
        )

    y_train = train_split["anomaly"]
    y_val = val_split["anomaly"]
    ds_idx = downsample_indices(y_train)
    log(f"{regime}: downsampled train rows: {len(ds_idx):,}")

    # Preserved for M3 numeric parity with the original script path.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_split.loc[ds_idx, feature_cols])
    x_val = scaler.transform(val_split[feature_cols])
    log(f"{regime}: scaled train/val matrices: {x_train.shape} / {x_val.shape}")

    run = fit_predict_models(
        x_train,
        y_train.loc[ds_idx],
        x_val,
        y_val,
        RANDOM_STATE,
    )
    return {
        "n_features": int(len(feature_cols)),
        "n_value_change_features": int(len(feature_cols) - len(BASELINE_FEATURE_COLS)),
        "n_train_downsampled": int(len(ds_idx)),
        "run": run,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "m3_50_50_ensemble_results.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    if len(SHIFTS) != 60 or len(PAST_SHIFTS) != 30 or len(FUTURE_SHIFTS) != 30:
        raise AssertionError("Unexpected value-change shift set")

    df = load_m3_frame()
    mask_val = (df["building_id"] % 2 == 1).to_numpy()
    train_buildings = set(df.loc[~mask_val, "building_id"].unique())
    val_buildings = set(df.loc[mask_val, "building_id"].unique())
    overlap = assert_no_building_overlap(
        train_buildings, val_buildings, split_name="50_50_mod2"
    )

    log(
        "50/50 mod2 split: "
        f"{len(train_buildings)} train buildings, {len(val_buildings)} val buildings"
    )
    if len(train_buildings) != 725 or len(val_buildings) != 724:
        raise AssertionError(
            f"Expected 725/724 train/val buildings, got "
            f"{len(train_buildings)}/{len(val_buildings)}"
        )

    log("Adding full offline value-change features once for both regimes")
    train_full = add_value_change_features(df.loc[~mask_val], list(SHIFTS))
    val_full = add_value_change_features(df.loc[mask_val], list(SHIFTS))
    value_cols = [c for c in train_full.columns if c.startswith("lag_value_")]
    if len(value_cols) != 120:
        raise AssertionError(
            f"Expected 120 value-change features, got {len(value_cols)}"
        )

    regimes = {
        "offline": run_regime(train_full, val_full, value_cols, "offline"),
        "causal": run_regime(train_full, val_full, value_cols, "causal"),
    }
    causal_auc = regimes["causal"]["run"]["ensemble"]["val_auc"]
    if causal_auc > M3_4_80_20_OFFLINE_ENSEMBLE_AUC:
        raise RuntimeError(
            "50/50 causal ensemble AUC unexpectedly exceeds the 80/20 offline "
            f"headline ({causal_auc:.6f} > {M3_4_80_20_OFFLINE_ENSEMBLE_AUC:.6f}); "
            "stop for review."
        )

    results: dict[str, object] = {
        "experiment": "m3_50_50_4_model_ensemble",
        "purpose": "PI-spec 50/50 building split ensemble follow-up",
        "split": {
            "name": "50_50_mod2",
            "protocol": "validation buildings are building_id % 2 == 1",
            "n_train_buildings": int(len(train_buildings)),
            "n_val_buildings": int(len(val_buildings)),
            "n_train_rows": int((~mask_val).sum()),
            "n_val_rows": int(mask_val.sum()),
            "train_anomaly_rate": float(df.loc[~mask_val, "anomaly"].mean()),
            "val_anomaly_rate": float(df.loc[mask_val, "anomaly"].mean()),
            "building_overlap": int(len(overlap)),
        },
        "random_state": RANDOM_STATE,
        "model_seed": RANDOM_STATE,
        "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
        "feature_counts": {
            "baseline": int(len(BASELINE_FEATURE_COLS)),
            "offline_value_change": int(len(value_cols)),
            "causal_value_change": int(len(PAST_SHIFTS) * 2),
            "offline_total": int(len(BASELINE_FEATURE_COLS) + len(value_cols)),
            "causal_total": int(len(BASELINE_FEATURE_COLS) + len(PAST_SHIFTS) * 2),
        },
        "regimes": regimes,
        "references": {
            "m3_2a_lightgbm_50_50_offline_auc": 0.9914,
            "m3_2a_lightgbm_50_50_causal_auc": 0.9903,
            "m3_4_80_20_offline_ensemble_auc": M3_4_80_20_OFFLINE_ENSEMBLE_AUC,
        },
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"Saved {args.out}")


if __name__ == "__main__":
    main()
