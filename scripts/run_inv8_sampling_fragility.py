"""INV-8: downsample seed and positive-duplication fragility checks."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import lightgbm as lgb
import numpy as np
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
    load_m3_frame,
    write_json_with_provenance,
)


VALUE_CHANGE_REGIME = "timestamp_merge"
SEED_PAIRS = (
    (10, 20),
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (42, 123),
    (999, 2025),
    (31415, 2718),
)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "inv8_sampling_fragility.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def m3_style_downsample(y, neg_seed_1: int, neg_seed_2: int) -> np.ndarray:
    neg_idx = y.index[y == 0].to_numpy()
    pos_idx = y.index[y == 1].to_numpy()
    n_pos = len(pos_idx)
    negs1 = np.random.RandomState(neg_seed_1).choice(neg_idx, n_pos, replace=False)
    negs2 = np.random.RandomState(neg_seed_2).choice(neg_idx, n_pos, replace=False)
    return np.concatenate([negs1, pos_idx, negs2, pos_idx])


def clean_50_50_downsample(y, neg_seed: int) -> np.ndarray:
    neg_idx = y.index[y == 0].to_numpy()
    pos_idx = y.index[y == 1].to_numpy()
    n_pos = len(pos_idx)
    negs = np.random.RandomState(neg_seed).choice(neg_idx, n_pos, replace=False)
    return np.concatenate([negs, pos_idx])


def fit_eval(train_full, val_full, feature_cols: list[str], fit_idx) -> dict[str, Any]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[fit_idx, feature_cols])
    x_val = scaler.transform(val_full[feature_cols])
    y_train = train_full.loc[fit_idx, "anomaly"]
    y_val = val_full["anomaly"]
    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=100,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    pred = model.predict_proba(x_val)[:, 1]
    return {
        **classification_metrics(y_val, pred),
        "n_train_downsampled": int(len(fit_idx)),
        "n_unique_train_rows": int(len(np.unique(fit_idx))),
        "fit_positive_rate": float(y_train.mean()),
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    aucs = [run["val_auc"] for run in runs]
    return {
        "n_runs": int(len(runs)),
        "mean_auc": float(mean(aucs)),
        "std_auc": float(pstdev(aucs)) if len(aucs) > 1 else 0.0,
        "min_auc": float(min(aucs)),
        "max_auc": float(max(aucs)),
        "range_auc": float(max(aucs) - min(aucs)),
        "runs": runs,
    }


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
        split_name="80_20_mod5",
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
    value_cols = [col for col in train_full.columns if col.startswith("lag_value_")]
    feature_cols = BASELINE_FEATURE_COLS + value_cols
    if len(feature_cols) != 137:
        raise AssertionError(f"Expected 137 features, got {len(feature_cols)}")
    y_train = train_full["anomaly"]

    m3_runs = []
    clean_runs = []
    for neg_seed_1, neg_seed_2 in SEED_PAIRS:
        log(f"M3-style downsample seeds=({neg_seed_1}, {neg_seed_2})")
        fit_idx = m3_style_downsample(y_train, neg_seed_1, neg_seed_2)
        run = fit_eval(train_full, val_full, feature_cols, fit_idx)
        run.update({"neg_seed_1": int(neg_seed_1), "neg_seed_2": int(neg_seed_2)})
        m3_runs.append(run)
        log(f"  AUC={run['val_auc']:.6f}")

        log(f"Clean 50:50 downsample seed={neg_seed_1}")
        clean_idx = clean_50_50_downsample(y_train, neg_seed_1)
        clean_run = fit_eval(train_full, val_full, feature_cols, clean_idx)
        clean_run.update({"neg_seed": int(neg_seed_1)})
        clean_runs.append(clean_run)
        log(f"  AUC={clean_run['val_auc']:.6f}")

    m3_summary = aggregate(m3_runs)
    clean_summary = aggregate(clean_runs)
    canonical_auc = next(
        run["val_auc"]
        for run in m3_runs
        if (run["neg_seed_1"], run["neg_seed_2"]) == tuple(DOWNSAMPLE_SEEDS)
    )
    results = {
        "experiment": "inv8_sampling_fragility",
        "issue": 53,
        "scope": (
            "Measure LightGBM validation AUC sensitivity to M3-style negative "
            "downsample seeds and to removing positive duplication in a clean "
            "50:50 fit set."
        ),
        "value_change_regime": VALUE_CHANGE_REGIME,
        "split": {
            "name": "80_20_mod5",
            "n_train_buildings": int(len(train_buildings)),
            "n_val_buildings": int(len(val_buildings)),
            "n_train_rows": int((~mask_val).sum()),
            "n_val_rows": int(mask_val.sum()),
            "train_anomaly_rate": float(df.loc[~mask_val, "anomaly"].mean()),
            "val_anomaly_rate": float(df.loc[mask_val, "anomaly"].mean()),
            "building_overlap": int(len(overlap)),
        },
        "feature_counts": {
            "baseline": int(len(BASELINE_FEATURE_COLS)),
            "value_change": int(len(value_cols)),
            "total": int(len(feature_cols)),
        },
        "canonical_downsampling_seeds": list(DOWNSAMPLE_SEEDS),
        "canonical_m3_style_auc": float(canonical_auc),
        "m3_style_seed_sweep": m3_summary,
        "clean_50_50_no_positive_duplication": clean_summary,
        "comparison": {
            "clean_mean_minus_m3_style_mean": float(
                clean_summary["mean_auc"] - m3_summary["mean_auc"]
            ),
            "m3_style_range_auc": m3_summary["range_auc"],
            "clean_range_auc": clean_summary["range_auc"],
            "needs_sampling_caveat": bool(
                m3_summary["range_auc"] > 0.0005
                or abs(clean_summary["mean_auc"] - m3_summary["mean_auc"]) > 0.0005
            ),
        },
        "random_state": RANDOM_STATE,
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }
    write_json_with_provenance(
        args.out,
        results,
        root=ROOT,
        provenance={
            "command": f"uv run python scripts/run_inv8_sampling_fragility.py --out {args.out}",
        },
    )
    log(f"Saved {args.out}")
    log(f"Sampling caveat: {results['comparison']['needs_sampling_caveat']}")


if __name__ == "__main__":
    main()
