"""INV-4: root-cause label-shuffle AUC via feature-group ablations."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lead import (
    BASELINE_FEATURE_COLS,
    DOWNSAMPLE_SEEDS,
    PROC,
    RANDOM_STATE,
    ROOT,
    SHIFTS,
    add_value_change_features,
    assert_no_building_overlap,
    classification_metrics,
    downsample_indices,
    load_m3_frame,
    write_json_with_provenance,
)


VALUE_CHANGE_REGIME = "timestamp_merge"
SHUFFLE_SEEDS = (42, 123, 999, 2025, 7, 31415, 2718, 8080)
SPLIT_NAME = "80_20_mod5"
BUILDING_METADATA_COLS = (
    "primary_use_enc",
    "log_square_feet",
    "year_built",
    "floor_count",
)
WEATHER_COLS = (
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "inv4_shuffle_ablation.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--shuffle-seeds",
        type=int,
        nargs="+",
        default=list(SHUFFLE_SEEDS),
        help="Label-shuffle seeds.",
    )
    return parser.parse_args()


def shuffle_labels(y_train: pd.Series, seed: int) -> pd.Series:
    shuffled = y_train.sample(frac=1, random_state=seed)
    shuffled.index = y_train.index
    return shuffled


def fit_eval_shuffle(
    *,
    train_full: pd.DataFrame,
    val_full: pd.DataFrame,
    feature_cols: list[str],
    shuffle_seed: int,
) -> dict[str, Any]:
    y_train = train_full["anomaly"]
    y_fit = shuffle_labels(y_train, shuffle_seed)
    ds_idx = downsample_indices(y_fit)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[ds_idx, feature_cols])
    x_val = scaler.transform(val_full[feature_cols])

    t0 = time.perf_counter()
    model = lgb.LGBMClassifier(
        n_estimators=100,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_fit.loc[ds_idx])
    pred = model.predict_proba(x_val)[:, 1]
    return {
        **classification_metrics(val_full["anomaly"], pred),
        "shuffle_seed": int(shuffle_seed),
        "n_features": int(len(feature_cols)),
        "n_train_downsampled": int(len(ds_idx)),
        "elapsed_seconds": float(time.perf_counter() - t0),
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    aucs = [run["val_auc"] for run in runs]
    return {
        "n_runs": int(len(runs)),
        "mean_auc": float(mean(aucs)),
        "std_auc": float(pstdev(aucs)) if len(aucs) > 1 else 0.0,
        "min_auc": float(min(aucs)),
        "max_auc": float(max(aucs)),
        "runs": runs,
    }


def ablation_sets(value_cols: list[str]) -> dict[str, dict[str, Any]]:
    full = list(BASELINE_FEATURE_COLS) + value_cols
    groups = {
        "meter_reading": ["meter_reading"],
        "value_change": value_cols,
        "building_metadata": [
            col for col in BUILDING_METADATA_COLS if col in BASELINE_FEATURE_COLS
        ],
        "weather": [col for col in WEATHER_COLS if col in BASELINE_FEATURE_COLS],
    }
    sets = {
        "full": {
            "removed_group": None,
            "removed_cols": [],
            "feature_cols": full,
        }
    }
    for name, cols in groups.items():
        remove = set(cols)
        sets[f"remove_{name}"] = {
            "removed_group": name,
            "removed_cols": cols,
            "feature_cols": [col for col in full if col not in remove],
        }
    return sets


def closest_to_half(results: dict[str, Any]) -> dict[str, Any]:
    ranked = sorted(
        (
            {
                "ablation": name,
                "mean_auc": result["summary"]["mean_auc"],
                "distance_from_0_5": abs(result["summary"]["mean_auc"] - 0.5),
            }
            for name, result in results.items()
        ),
        key=lambda item: item["distance_from_0_5"],
    )
    return ranked[0]


def main() -> None:
    args = parse_args()
    t0 = time.time()
    df = load_m3_frame(verbose=True)
    mask_val = (df["building_id"] % 5 == 4).to_numpy()
    train_buildings = set(df.loc[~mask_val, "building_id"].unique())
    val_buildings = set(df.loc[mask_val, "building_id"].unique())
    overlap = assert_no_building_overlap(
        train_buildings,
        val_buildings,
        split_name=SPLIT_NAME,
    )

    log(f"Building {VALUE_CHANGE_REGIME} feature table")
    train_full = add_value_change_features(
        df.loc[~mask_val],
        list(SHIFTS),
        value_change_regime=VALUE_CHANGE_REGIME,
    )
    val_full = add_value_change_features(
        df.loc[mask_val],
        list(SHIFTS),
        value_change_regime=VALUE_CHANGE_REGIME,
    )
    value_cols = [c for c in train_full.columns if c.startswith("lag_value_")]
    if len(BASELINE_FEATURE_COLS) + len(value_cols) != 137:
        raise AssertionError("Unexpected M3 feature count")

    results_by_ablation: dict[str, Any] = {}
    for name, config in ablation_sets(value_cols).items():
        feature_cols = config["feature_cols"]
        log(f"Ablation {name}: {len(feature_cols)} features")
        runs = []
        for seed in args.shuffle_seeds:
            metrics = fit_eval_shuffle(
                train_full=train_full,
                val_full=val_full,
                feature_cols=feature_cols,
                shuffle_seed=seed,
            )
            runs.append(metrics)
            log(f"  seed={seed}: AUC={metrics['val_auc']:.4f}")
        results_by_ablation[name] = {
            "removed_group": config["removed_group"],
            "removed_cols": config["removed_cols"],
            "n_features": int(len(feature_cols)),
            "summary": aggregate(runs),
        }

    closest = closest_to_half(results_by_ablation)
    full_mean = results_by_ablation["full"]["summary"]["mean_auc"]
    interpretation = {
        "closest_to_random": closest,
        "full_mean_auc": full_mean,
        "likely_source": closest["ablation"].replace("remove_", "")
        if closest["ablation"].startswith("remove_")
        else "not_isolated_by_single_group_removal",
        "red_flag": False,
        "rule": (
            "Feature group whose removal moves shuffled-label AUC closest to 0.5 "
            "is the leading residual-structure source. If no group moves toward "
            "0.5, source remains unresolved."
        ),
    }
    results = {
        "experiment": "inv4_shuffle_ablation",
        "issue": 53,
        "scope": (
            "Diagnose elevated label-shuffle AUC by removing feature groups under "
            "the opt-in meter-aware value-change regime."
        ),
        "split": {
            "name": SPLIT_NAME,
            "protocol": "validation buildings are building_id % 5 == 4",
            "n_train_buildings": int(len(train_buildings)),
            "n_val_buildings": int(len(val_buildings)),
            "n_train_rows": int((~mask_val).sum()),
            "n_val_rows": int(mask_val.sum()),
            "train_anomaly_rate": float(df.loc[~mask_val, "anomaly"].mean()),
            "val_anomaly_rate": float(df.loc[mask_val, "anomaly"].mean()),
            "building_overlap": int(len(overlap)),
        },
        "value_change_regime": VALUE_CHANGE_REGIME,
        "shuffle_seeds": list(args.shuffle_seeds),
        "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
        "model_random_state": RANDOM_STATE,
        "feature_groups": {
            "meter_reading": ["meter_reading"],
            "value_change_count": int(len(value_cols)),
            "building_metadata": list(BUILDING_METADATA_COLS),
            "weather": list(WEATHER_COLS),
        },
        "ablations": results_by_ablation,
        "interpretation": interpretation,
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }
    write_json_with_provenance(
        args.out,
        results,
        root=ROOT,
        provenance={
            "command": f"uv run python scripts/run_inv4_shuffle_ablation.py --out {args.out}",
        },
    )
    log(f"Saved {args.out}")
    log(f"Closest to 0.5: {closest['ablation']} mean_auc={closest['mean_auc']:.4f}")


if __name__ == "__main__":
    main()
