"""Fail-closed audit of the M5 tree runner against the M3 50/50 headline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lead import (
    BASELINE_FEATURE_COLS,
    DOWNSAMPLE_SEEDS,
    PROC,
    ROOT,
    SHIFTS,
    add_value_change_features,
    downsample_indices,
    load_m3_frame,
)
from m5_building_curve_protocol import resolve_cell_indices
from m5_tree_early_stopping import early_stopping_contract
from run_m3_figure_observations import frozen_model_contract
from run_m5_building_curve_tree_cell import (
    MATRIX_DTYPE,
    M3_SORT_KEYS,
    PREDICTION_DTYPE,
    _m3_downsampled_rows,
    _matrix_columns,
    _scale_matrix,
)
from run_m5_tree_ensemble_matched_context import (
    CANONICAL_ORDER,
    build_features_keeping_index,
)

RAW_INDEX = "__audit_raw_index"
DEFAULT_MANIFEST = (
    PROC
    / "m5_building_curve"
    / "protocol_full"
    / "representative"
    / "seed42"
    / "building_ladder.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample-buildings", type=int, default=10)
    return parser.parse_args()


def sha256_int64(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="int64").astype("<i8", copy=False).tobytes()
    ).hexdigest()


def ordered_raw_rows(frame: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    ordered = frame.loc[rows, [*M3_SORT_KEYS, "anomaly"]].copy()
    ordered[RAW_INDEX] = ordered.index.to_numpy(dtype="int64")
    return ordered.sort_values(list(M3_SORT_KEYS)).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    frame = load_m3_frame(verbose=True)
    even = frame["building_id"].mod(2).eq(0)
    odd = ~even
    even_buildings = np.sort(frame.loc[even, "building_id"].unique())
    odd_buildings = np.sort(frame.loc[odd, "building_id"].unique())
    if (len(even_buildings), len(odd_buildings)) != (725, 724):
        raise AssertionError("M3 building_id % 2 split is not 725/724")

    manifest = json.loads(args.building_manifest.read_text(encoding="utf-8"))
    full_budget = int(manifest["candidate_buildings"])
    if full_budget != 725:
        raise AssertionError("full tree cell does not contain all 725 even buildings")
    full_cell = manifest["cells"][str(full_budget)]
    if set(full_cell["available_buildings"]) != set(even_buildings.tolist()):
        raise AssertionError(
            "full tree building identities differ from M3 training half"
        )
    resolved = resolve_cell_indices(frame.loc[even], manifest, full_budget)
    available = np.asarray(resolved["available_rows"], dtype="int64")
    if set(available.tolist()) != set(frame.index[even].tolist()):
        raise AssertionError("full tree source rows differ from M3 training half")

    ordered_train = ordered_raw_rows(frame, available)
    sampled_positions = np.asarray(
        downsample_indices(ordered_train["anomaly"]), dtype="int64"
    )
    expected_rows = ordered_train.loc[sampled_positions, RAW_INDEX].to_numpy(
        dtype="int64"
    )
    actual_rows = _m3_downsampled_rows(frame, available)
    np.testing.assert_array_equal(actual_rows, expected_rows)

    canonical_path = CANONICAL_ORDER
    if not canonical_path.is_file():
        canonical_path = (
            ROOT.parent
            / "lead-reproduction"
            / "data"
            / "processed"
            / CANONICAL_ORDER.name
        )
    with np.load(canonical_path) as canonical:
        canonical_rows = np.asarray(canonical["validation_raw_index"], dtype="int64")
        canonical_y = np.asarray(canonical["anomaly"], dtype="int8")
    ordered_holdout = ordered_raw_rows(frame, frame.index[odd].to_numpy(dtype="int64"))
    expected_holdout = ordered_holdout[RAW_INDEX].to_numpy(dtype="int64")
    np.testing.assert_array_equal(canonical_rows, expected_holdout)
    np.testing.assert_array_equal(
        canonical_y, frame.loc[canonical_rows, "anomaly"].to_numpy(dtype="int8")
    )

    sample_ids = even_buildings[: args.sample_buildings]
    raw_sample = frame.loc[frame["building_id"].isin(sample_ids)].copy()
    sample_order = ordered_raw_rows(frame, raw_sample.index.to_numpy(dtype="int64"))
    expected_features = add_value_change_features(
        raw_sample.copy(), list(SHIFTS), value_change_regime="timestamp_merge"
    )
    actual_features = build_features_keeping_index(raw_sample.copy())
    actual_ordered = actual_features.loc[
        sample_order[RAW_INDEX].to_numpy(dtype="int64")
    ].reset_index(drop=True)
    value_columns = [
        name for name in expected_features.columns if name.startswith("lag_value_")
    ]
    columns = [*BASELINE_FEATURE_COLS, *value_columns]
    if len(value_columns) != 120 or len(columns) != 137:
        raise AssertionError("M3 feature list is not 17 + 120 = 137")
    if _matrix_columns(137, list(frame.columns)) != columns:
        raise AssertionError("M5 feature order differs from M3")
    pd.testing.assert_frame_equal(
        actual_ordered[columns], expected_features[columns], check_exact=True
    )

    sample_positions = np.asarray(
        downsample_indices(expected_features["anomaly"]), dtype="int64"
    )
    m3_scaler = StandardScaler()
    expected_scaled = m3_scaler.fit_transform(
        expected_features.loc[sample_positions, columns]
    )
    sample_raw_rows = sample_order.loc[sample_positions, RAW_INDEX].to_numpy(
        dtype="int64"
    )
    current_values = actual_features.loc[sample_raw_rows, columns].to_numpy(
        dtype=MATRIX_DTYPE
    )
    current_scaler = StandardScaler().fit(current_values)
    actual_scaled = _scale_matrix(current_scaler, current_values)
    np.testing.assert_array_equal(actual_scaled, expected_scaled)
    np.testing.assert_array_equal(current_scaler.mean_, m3_scaler.mean_)
    np.testing.assert_array_equal(current_scaler.scale_, m3_scaler.scale_)

    m3_contract = frozen_model_contract(seed=42)
    es_contract = early_stopping_contract(seed=42)
    for model in m3_contract:
        if es_contract[model]["class"] != m3_contract[model]["class"]:
            raise AssertionError(f"{model} class differs from M3")
        if es_contract[model]["selection_metric"] != "roc_auc":
            raise AssertionError(f"{model} does not select iterations by ROC-AUC")
    if es_contract["xgboost"]["params"]["eval_metric"] != "auc":
        raise AssertionError("XGBoost ROC-AUC adapter drifted")
    if es_contract["catboost"]["params"]["eval_metric"] != "AUC":
        raise AssertionError("CatBoost ROC-AUC adapter drifted")
    if es_contract["hist_gradient_boosting"]["params"]["scoring"] != "roc_auc":
        raise AssertionError("HistGradientBoosting ROC-AUC adapter drifted")
    if tuple(DOWNSAMPLE_SEEDS) != (10, 20):
        raise AssertionError("M3 downsampling seeds drifted")
    if MATRIX_DTYPE != np.dtype("float64") or PREDICTION_DTYPE != np.dtype("float64"):
        raise AssertionError("M5 numeric path differs from M3 float64")

    print(
        json.dumps(
            {
                "status": "passed",
                "split_buildings": [725, 724],
                "feature_count": len(columns),
                "downsampled_rows": len(actual_rows),
                "downsampled_row_sha256": sha256_int64(actual_rows),
                "holdout_rows": len(canonical_rows),
                "holdout_row_sha256": sha256_int64(canonical_rows),
                "matrix_dtype": MATRIX_DTYPE.name,
                "prediction_dtype": PREDICTION_DTYPE.name,
                "early_stopping_metric": "roc_auc",
                "ensemble": "equal_weight_mean_of_four",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
