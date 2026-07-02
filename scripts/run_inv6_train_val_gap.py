"""INV-6: score fit-set, full train-buildings, and validation AUC gaps."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

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


VALUE_CHANGE_REGIME = "row_offset_meter_aware"
SPLIT_NAME = "80_20_mod5"


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "inv6_train_val_gap.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--score-chunk-rows",
        type=int,
        default=500_000,
        help="Rows per chunk when scoring full train-buildings.",
    )
    return parser.parse_args()


def predict_chunks(model, scaler, frame, feature_cols: list[str], *, chunk_rows: int):
    preds = []
    for start in range(0, len(frame), chunk_rows):
        chunk = frame.iloc[start : start + chunk_rows]
        x_chunk = scaler.transform(chunk[feature_cols])
        if isinstance(model, HistGradientBoostingClassifier):
            x_chunk = np.nan_to_num(x_chunk, nan=0)
        preds.append(model.predict_proba(x_chunk)[:, 1])
    return np.concatenate(preds)


def score_model(
    *,
    model,
    scaler,
    fit_frame,
    full_train_frame,
    val_frame,
    feature_cols: list[str],
    chunk_rows: int,
) -> dict[str, Any]:
    fit_pred = predict_chunks(
        model,
        scaler,
        fit_frame,
        feature_cols,
        chunk_rows=chunk_rows,
    )
    train_pred = predict_chunks(
        model,
        scaler,
        full_train_frame,
        feature_cols,
        chunk_rows=chunk_rows,
    )
    val_pred = predict_chunks(
        model,
        scaler,
        val_frame,
        feature_cols,
        chunk_rows=chunk_rows,
    )
    return {
        "fit_set_auc": float(roc_auc_score(fit_frame["anomaly"], fit_pred)),
        "full_train_buildings_auc": float(
            roc_auc_score(full_train_frame["anomaly"], train_pred)
        ),
        "val_auc": float(roc_auc_score(val_frame["anomaly"], val_pred)),
        "fit_set_metrics": classification_metrics(fit_frame["anomaly"], fit_pred),
        "full_train_buildings_metrics": classification_metrics(
            full_train_frame["anomaly"], train_pred
        ),
        "val_metrics": classification_metrics(val_frame["anomaly"], val_pred),
    }


def model_specs():
    return {
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=100,
            verbose=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=100,
            eval_metric="logloss",
            verbosity=0,
            random_state=RANDOM_STATE,
        ),
        "catboost": CatBoostClassifier(
            iterations=1000,
            verbose=False,
            random_seed=RANDOM_STATE,
            allow_writing_files=False,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            random_state=RANDOM_STATE,
        ),
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

    y_train = train_full["anomaly"]
    ds_idx = downsample_indices(y_train)
    fit_frame = train_full.loc[ds_idx]
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(fit_frame[feature_cols])
    y_fit = fit_frame["anomaly"]

    per_model: dict[str, Any] = {}
    ensemble_preds = {"fit_set": [], "full_train_buildings": [], "val": []}
    for name, model in model_specs().items():
        log(f"Fitting {name}")
        x_model = (
            np.nan_to_num(x_fit, nan=0) if name == "hist_gradient_boosting" else x_fit
        )
        model.fit(x_model, y_fit)
        scored = score_model(
            model=model,
            scaler=scaler,
            fit_frame=fit_frame,
            full_train_frame=train_full,
            val_frame=val_full,
            feature_cols=feature_cols,
            chunk_rows=args.score_chunk_rows,
        )
        per_model[name] = scored
        log(
            f"  {name}: fit={scored['fit_set_auc']:.6f} "
            f"train={scored['full_train_buildings_auc']:.6f} "
            f"val={scored['val_auc']:.6f}"
        )
        ensemble_preds["fit_set"].append(
            predict_chunks(
                model,
                scaler,
                fit_frame,
                feature_cols,
                chunk_rows=args.score_chunk_rows,
            )
        )
        ensemble_preds["full_train_buildings"].append(
            predict_chunks(
                model,
                scaler,
                train_full,
                feature_cols,
                chunk_rows=args.score_chunk_rows,
            )
        )
        ensemble_preds["val"].append(
            predict_chunks(
                model,
                scaler,
                val_full,
                feature_cols,
                chunk_rows=args.score_chunk_rows,
            )
        )

    ensemble_fit = sum(ensemble_preds["fit_set"]) / len(ensemble_preds["fit_set"])
    ensemble_train = sum(ensemble_preds["full_train_buildings"]) / len(
        ensemble_preds["full_train_buildings"]
    )
    ensemble_val = sum(ensemble_preds["val"]) / len(ensemble_preds["val"])
    ensemble = {
        "fit_set_auc": float(roc_auc_score(fit_frame["anomaly"], ensemble_fit)),
        "full_train_buildings_auc": float(
            roc_auc_score(train_full["anomaly"], ensemble_train)
        ),
        "val_auc": float(roc_auc_score(val_full["anomaly"], ensemble_val)),
        "fit_set_metrics": classification_metrics(fit_frame["anomaly"], ensemble_fit),
        "full_train_buildings_metrics": classification_metrics(
            train_full["anomaly"], ensemble_train
        ),
        "val_metrics": classification_metrics(val_full["anomaly"], ensemble_val),
    }
    results = {
        "experiment": "inv6_train_val_gap",
        "issue": 53,
        "scope": (
            "Score the fit-set, full train-buildings rows, and validation rows "
            "after fitting on the downsampled M3-compatible fit-set."
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
        "n_train_downsampled": int(len(ds_idx)),
        "score_chunk_rows": int(args.score_chunk_rows),
        "models": per_model,
        "ensemble": ensemble,
        "interpretation": {
            "lightgbm_full_train_minus_val_auc": float(
                per_model["lightgbm"]["full_train_buildings_auc"]
                - per_model["lightgbm"]["val_auc"]
            ),
            "ensemble_full_train_minus_val_auc": float(
                ensemble["full_train_buildings_auc"] - ensemble["val_auc"]
            ),
        },
        "random_state": RANDOM_STATE,
        "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }
    write_json_with_provenance(
        args.out,
        results,
        root=ROOT,
        provenance={
            "command": f"uv run python scripts/run_inv6_train_val_gap.py --out {args.out}",
        },
    )
    log(f"Saved {args.out}")
    log(
        f"Ensemble gap: {results['interpretation']['ensemble_full_train_minus_val_auc']:+.6f}"
    )


if __name__ == "__main__":
    main()
