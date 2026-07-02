"""INV-7: per-building validation AUC distribution and bootstrap CIs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
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
    downsample_indices,
    load_m3_frame,
    write_json_with_provenance,
)
from run_m3_4_ensemble import fit_predict_models


VALUE_CHANGE_REGIME = "row_offset_meter_aware"
SPLIT_NAME = "80_20_mod5"


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "inv7_per_building_distribution.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Building-level bootstrap samples for per-building AUC means.",
    )
    return parser.parse_args()


def per_building_auc(val_full, pred: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    work = val_full[["building_id", "primary_use_enc", "anomaly"]].copy()
    work["_pred"] = pred
    for building_id, group in work.groupby("building_id", sort=False):
        y = group["anomaly"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        auc = roc_auc_score(y, group["_pred"].to_numpy())
        rows.append(
            {
                "building_id": int(building_id),
                "primary_use_enc": int(group["primary_use_enc"].iloc[0]),
                "n_rows": int(len(group)),
                "n_positive_rows": int(y.sum()),
                "auc": float(auc),
            }
        )
    return rows


def distribution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aucs = np.asarray([row["auc"] for row in rows], dtype=float)
    if not len(aucs):
        return {}
    return {
        "effective_buildings": int(len(rows)),
        "median_auc": float(np.median(aucs)),
        "mean_auc": float(np.mean(aucs)),
        "p10_auc": float(np.percentile(aucs, 10)),
        "p90_auc": float(np.percentile(aucs, 90)),
        "min_auc": float(np.min(aucs)),
        "max_auc": float(np.max(aucs)),
    }


def bootstrap_ci(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    aucs = np.asarray([row["auc"] for row in rows], dtype=float)
    if not len(aucs):
        return {"n": 0, "mean": None, "ci95": None}
    rng = np.random.RandomState(seed)
    means = [
        float(np.mean(rng.choice(aucs, size=len(aucs), replace=True)))
        for _ in range(samples)
    ]
    return {
        "n": int(len(aucs)),
        "mean": float(np.mean(aucs)),
        "ci95": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
        "bootstrap_samples": int(samples),
        "unit": "validation building with both classes",
    }


def primary_use_slices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    by_primary: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_primary.setdefault(row["primary_use_enc"], []).append(row)
    for primary_use_enc, group_rows in sorted(by_primary.items()):
        summary = distribution_summary(group_rows)
        high_small_slice = (
            summary["effective_buildings"] <= 10 and summary["median_auc"] >= 0.999
        )
        out.append(
            {
                "primary_use_enc": int(primary_use_enc),
                **summary,
                "n_rows": int(sum(row["n_rows"] for row in group_rows)),
                "n_positive_rows": int(
                    sum(row["n_positive_rows"] for row in group_rows)
                ),
                "high_score_small_slice": bool(high_small_slice),
            }
        )
    return out


def summarize_model(name: str, val_full, pred: np.ndarray, *, bootstrap_samples: int):
    rows = per_building_auc(val_full, pred)
    return {
        "name": name,
        "row_aggregate_auc": float(roc_auc_score(val_full["anomaly"], pred)),
        "per_building": distribution_summary(rows),
        "building_bootstrap_ci": bootstrap_ci(rows, samples=bootstrap_samples),
        "primary_use_slices": primary_use_slices(rows),
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
    value_cols = [col for col in train_full.columns if col.startswith("lag_value_")]
    feature_cols = BASELINE_FEATURE_COLS + value_cols
    if len(feature_cols) != 137:
        raise AssertionError(f"Expected 137 features, got {len(feature_cols)}")

    ds_idx = downsample_indices(train_full["anomaly"])
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[ds_idx, feature_cols])
    x_val = scaler.transform(val_full[feature_cols])
    y_fit = train_full.loc[ds_idx, "anomaly"]
    y_val = val_full["anomaly"]

    run = fit_predict_models(
        x_train,
        y_fit,
        x_val,
        y_val,
        RANDOM_STATE,
        return_predictions=True,
    )
    lightgbm_pred = run["raw_predictions"]["lightgbm"]
    ensemble_pred = run["raw_ensemble_prediction"]
    del run["raw_predictions"]
    del run["raw_ensemble_prediction"]

    models = {
        "lightgbm": summarize_model(
            "lightgbm",
            val_full,
            lightgbm_pred,
            bootstrap_samples=args.bootstrap_samples,
        ),
        "ensemble": summarize_model(
            "ensemble",
            val_full,
            ensemble_pred,
            bootstrap_samples=args.bootstrap_samples,
        ),
    }
    light_ci = models["lightgbm"]["building_bootstrap_ci"]["ci95"]
    ens_ci = models["ensemble"]["building_bootstrap_ci"]["ci95"]
    comparison = {
        "row_aggregate_delta_ensemble_minus_lightgbm": float(
            models["ensemble"]["row_aggregate_auc"]
            - models["lightgbm"]["row_aggregate_auc"]
        ),
        "building_bootstrap_ci_overlap": bool(
            max(light_ci[0], ens_ci[0]) <= min(light_ci[1], ens_ci[1])
        ),
        "lightgbm_ci95": light_ci,
        "ensemble_ci95": ens_ci,
    }

    results = {
        "experiment": "inv7_per_building_distribution",
        "issue": 53,
        "scope": (
            "Compute per-building validation AUC distribution and building-level "
            "bootstrap CIs for LightGBM and the 4-model ensemble."
        ),
        "value_change_regime": VALUE_CHANGE_REGIME,
        "split": {
            "name": SPLIT_NAME,
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
        "run": run,
        "models": models,
        "comparison": comparison,
        "random_state": RANDOM_STATE,
        "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }
    write_json_with_provenance(
        args.out,
        results,
        root=ROOT,
        provenance={
            "command": f"uv run python scripts/run_inv7_per_building_distribution.py --out {args.out}",
        },
    )
    log(f"Saved {args.out}")
    log(f"CI overlap: {comparison['building_bootstrap_ci_overlap']}")


if __name__ == "__main__":
    main()
