"""Run M3.4 4-model equal-weight ensemble on M3.2 features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
M3 = ROOT / "data" / "raw" / "m3"
PROC = ROOT / "data" / "processed"

RANDOM_STATE = 42
DOWNSAMPLE_SEEDS = (10, 20)
DEFAULT_MODEL_SEEDS = (42, 123, 999)

BASELINE_FEATURE_COLS = [
    "meter",
    "meter_reading",
    "hour",
    "weekday",
    "month",
    "dayofyear",
    "primary_use_enc",
    "log_square_feet",
    "year_built",
    "floor_count",
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
]

SHIFTS = (
    list(range(-24, 0))
    + list(range(1, 25))
    + list(range(-168, -24, 24))
    + list(range(48, 169, 24))
)

M2_MODEL_AUCS = {
    "lightgbm": 0.9818,
    "xgboost": 0.9749,
    "catboost": 0.9797,
    "hist_gradient_boosting": 0.9806,
    "ensemble": 0.9830,
}
M2_RANKING = ["lightgbm", "hist_gradient_boosting", "catboost", "xgboost"]


def log(message: str) -> None:
    print(message, flush=True)


def load_m3_frame() -> pd.DataFrame:
    t0 = time.time()
    train = pd.read_csv(
        M3 / "train.csv",
        dtype={"building_id": "int16", "meter": "int8", "meter_reading": "float32"},
    )
    bad = pd.read_csv(M3 / "bad_meter_readings.csv")
    if len(bad) != len(train):
        raise ValueError("bad_meter_readings.csv must align 1:1 with train.csv")
    train["anomaly"] = bad["is_bad_meter_reading"].values.astype("int8")

    train["timestamp"] = pd.to_datetime(train["timestamp"])
    train["hour"] = train["timestamp"].dt.hour.astype("int8")
    train["weekday"] = train["timestamp"].dt.weekday.astype("int8")
    train["month"] = train["timestamp"].dt.month.astype("int8")
    train["dayofyear"] = (
        train["timestamp"].dt.dayofyear + train["timestamp"].dt.hour / 24
    ).astype("float32")

    meta = pd.read_csv(M3 / "building_metadata.csv")
    le = LabelEncoder()
    meta["primary_use_enc"] = le.fit_transform(
        meta["primary_use"].fillna("Unknown")
    ).astype("int8")
    meta["log_square_feet"] = np.log1p(meta["square_feet"]).astype("float32")
    meta_cols = [
        "building_id",
        "site_id",
        "primary_use_enc",
        "log_square_feet",
        "year_built",
        "floor_count",
    ]
    train = train.merge(meta[meta_cols], on="building_id", how="left")

    weather = pd.read_csv(M3 / "weather_train.csv")
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather["cloud_coverage"] = (
        weather["cloud_coverage"].replace({255: 10}).astype("float32")
    )
    weather_cols = [
        "site_id",
        "timestamp",
        "air_temperature",
        "cloud_coverage",
        "dew_temperature",
        "precip_depth_1_hr",
        "sea_level_pressure",
        "wind_direction",
        "wind_speed",
    ]
    train = train.merge(weather[weather_cols], on=["site_id", "timestamp"], how="left")

    keep_cols = ["building_id", "timestamp", "anomaly", *BASELINE_FEATURE_COLS]
    keep_cols = list(dict.fromkeys(keep_cols))
    log(f"Loaded M3 frame {train.shape} in {(time.time() - t0) / 60:.1f} min")
    return train[keep_cols]


def add_value_change_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True).copy()
    mr = out["meter_reading"]
    grouped = out.groupby("building_id", sort=False)["meter_reading"]
    new_cols = {}
    for n in SHIFTS:
        shifted = grouped.shift(n)
        new_cols[f"lag_value_diff_{n}"] = (mr - shifted).astype("float32")
        new_cols[f"lag_value_ratio_{n}"] = ((mr + 1) / (shifted + 1)).astype("float32")
    return pd.concat([out, pd.DataFrame(new_cols)], axis=1)


def downsample_indices(y: pd.Series) -> np.ndarray:
    neg_idx = y.index[y == 0].to_numpy()
    pos_idx = y.index[y == 1].to_numpy()
    n_pos = len(pos_idx)
    negs1 = np.random.RandomState(DOWNSAMPLE_SEEDS[0]).choice(
        neg_idx, n_pos, replace=False
    )
    negs2 = np.random.RandomState(DOWNSAMPLE_SEEDS[1]).choice(
        neg_idx, n_pos, replace=False
    )
    return np.concatenate([negs1, pos_idx, negs2, pos_idx])


def classification_metrics(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    pred_label = (pred >= 0.5).astype("int8")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred_label, average="binary", pos_label=1, zero_division=0
    )
    return {
        "val_auc": float(roc_auc_score(y_true, pred)),
        "precision_05": float(precision),
        "recall_05": float(recall),
        "f1_05": float(f1),
    }


def model_ranking(model_metrics: dict[str, dict[str, float]]) -> list[str]:
    return sorted(
        model_metrics,
        key=lambda name: model_metrics[name]["val_auc"],
        reverse=True,
    )


def fit_predict_models(
    x_train: np.ndarray,
    y_train: pd.Series,
    x_val: np.ndarray,
    y_val: pd.Series,
    seed: int,
    *,
    return_predictions: bool = False,
) -> dict[str, object]:
    log(f"Training 4-model ensemble seed={seed}")
    t0 = time.time()
    preds: dict[str, np.ndarray] = {}
    model_metrics: dict[str, dict[str, float]] = {}
    train_times: dict[str, float] = {}

    models = {
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=100,
            verbose=-1,
            random_state=seed,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=100,
            eval_metric="logloss",
            verbosity=0,
            random_state=seed,
        ),
        "catboost": CatBoostClassifier(
            iterations=1000,
            verbose=False,
            random_seed=seed,
            allow_writing_files=False,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            random_state=seed,
        ),
    }

    for name, model in models.items():
        mt0 = time.time()
        log(f"  fitting {name}")
        if name == "hist_gradient_boosting":
            x_train_fit = np.nan_to_num(x_train, nan=0)
            x_val_fit = np.nan_to_num(x_val, nan=0)
        else:
            x_train_fit = x_train
            x_val_fit = x_val
        model.fit(x_train_fit, y_train)
        pred = model.predict_proba(x_val_fit)[:, 1]
        preds[name] = pred
        model_metrics[name] = classification_metrics(y_val, pred)
        train_times[name] = round((time.time() - mt0) / 60, 3)
        log(
            f"    {name}: AUC={model_metrics[name]['val_auc']:.4f} "
            f"P/R/F1={model_metrics[name]['precision_05']:.4f}/"
            f"{model_metrics[name]['recall_05']:.4f}/"
            f"{model_metrics[name]['f1_05']:.4f} "
            f"({train_times[name]:.1f} min)"
        )

    ensemble_pred = sum(preds.values()) / len(preds)
    ensemble_metrics = classification_metrics(y_val, ensemble_pred)
    ranking = model_ranking(model_metrics)
    log(
        f"  ensemble seed={seed}: AUC={ensemble_metrics['val_auc']:.4f} "
        f"P/R/F1={ensemble_metrics['precision_05']:.4f}/"
        f"{ensemble_metrics['recall_05']:.4f}/"
        f"{ensemble_metrics['f1_05']:.4f}"
    )

    catboost_tree_count = getattr(models["catboost"], "tree_count_", None)
    result = {
        "seed": int(seed),
        "models": model_metrics,
        "ensemble": ensemble_metrics,
        "ranking": ranking,
        "ranking_matches_m2": ranking == M2_RANKING,
        "ensemble_delta_vs_m3_2": float(ensemble_metrics["val_auc"] - 0.9920),
        "ensemble_delta_vs_best_model": float(
            ensemble_metrics["val_auc"]
            - max(m["val_auc"] for m in model_metrics.values())
        ),
        "train_minutes": train_times,
        "catboost_tree_count": int(catboost_tree_count)
        if catboost_tree_count is not None
        else None,
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }
    if return_predictions:
        result["raw_predictions"] = preds
        result["raw_ensemble_prediction"] = ensemble_pred
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_MODEL_SEEDS),
        help="Model random seeds for ensemble sanity runs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "m3_4_results.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    if len(SHIFTS) != 60:
        raise AssertionError("Unexpected value-change shift set")
    if RANDOM_STATE not in args.model_seeds:
        raise ValueError("model seeds must include 42 for the canonical run")

    df = load_m3_frame()
    mask_val = (df["building_id"] % 5 == 4).to_numpy()
    train_buildings = set(df.loc[~mask_val, "building_id"].unique())
    val_buildings = set(df.loc[mask_val, "building_id"].unique())
    overlap = train_buildings & val_buildings
    if overlap:
        raise AssertionError(f"building overlap: {sorted(overlap)[:5]}")

    log("Adding M3.2 offline value-change features")
    train_full = add_value_change_features(df.loc[~mask_val])
    val_full = add_value_change_features(df.loc[mask_val])
    value_cols = [c for c in train_full.columns if c.startswith("lag_value_")]
    feature_cols = BASELINE_FEATURE_COLS + value_cols
    if len(feature_cols) != 137:
        raise AssertionError(f"Expected 137 M3.2 features, got {len(feature_cols)}")

    y_train = train_full["anomaly"]
    y_val = val_full["anomaly"]
    ds_idx = downsample_indices(y_train)
    log(f"Downsampled train rows: {len(ds_idx):,}")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[ds_idx, feature_cols])
    x_val = scaler.transform(val_full[feature_cols])
    y_fit = y_train.loc[ds_idx]
    log(f"Scaled train/val matrices: {x_train.shape} / {x_val.shape}")

    runs = {}
    for seed in args.model_seeds:
        runs[str(seed)] = fit_predict_models(x_train, y_fit, x_val, y_val, seed)

    canonical = runs[str(RANDOM_STATE)]
    seed_aucs = [run["ensemble"]["val_auc"] for run in runs.values()]
    results: dict[str, object] = {
        "experiment": "m3_4_4_model_ensemble",
        "canonical_line": "80_20_mod5_offline",
        "feature_set": "m3_2_137_features",
        "random_state": RANDOM_STATE,
        "model_seeds": [int(seed) for seed in args.model_seeds],
        "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
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
        "n_train_downsampled": int(len(ds_idx)),
        "m3_2_reference": {
            "val_auc": 0.9920,
            "precision_05": 0.6409,
            "recall_05": 0.9665,
            "f1_05": 0.7707,
        },
        "m2_reference": {
            "model_aucs": M2_MODEL_AUCS,
            "ranking": M2_RANKING,
        },
        "main": canonical,
        "multi_seed": {
            "runs": runs,
            "mean_ensemble_auc": float(np.mean(seed_aucs)),
            "std_ensemble_auc": float(np.std(seed_aucs)),
            "min_ensemble_auc": float(np.min(seed_aucs)),
            "max_ensemble_auc": float(np.max(seed_aucs)),
        },
        "interpretation": {
            "noise_floor_auc": 0.0005,
            "ensemble_auc_threshold_for_non_negligible_lift": 0.9925,
            "canonical_lift_label": "pending",
        },
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }

    canonical_delta = canonical["ensemble_delta_vs_m3_2"]
    if canonical_delta > 0.0005:
        results["interpretation"]["canonical_lift_label"] = "positive_lift"
    elif canonical_delta < -0.0005:
        results["interpretation"]["canonical_lift_label"] = "negative_lift"
    else:
        results["interpretation"]["canonical_lift_label"] = "negligible"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"Saved {args.out}")


if __name__ == "__main__":
    main()
