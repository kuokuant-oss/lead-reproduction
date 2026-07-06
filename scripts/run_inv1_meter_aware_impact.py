"""INV-1: quantify meter-aware value-change impact on M3 model comparisons."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
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
from run_m3_4_ensemble import fit_predict_models


NOISE_FLOOR_AUC = 0.0005
REGIMES = ("row_offset", "row_offset_meter_aware", "timestamp_merge")
M3_2_80_20_GOLDEN_AUC = 0.9920119520500562
M3_4_80_20_GOLDEN_AUC = 0.9927886432126508
M3_2A_50_50_LIGHTGBM_REFERENCE_AUC = 0.9914
M3_50_50_ENSEMBLE_GOLDEN_AUC = 0.9921214897674221

SPLITS = {
    "80_20_mod5": {
        "protocol": "validation buildings are building_id % 5 == 4",
        "golden": {
            "single_lightgbm": M3_2_80_20_GOLDEN_AUC,
            "ensemble": M3_4_80_20_GOLDEN_AUC,
        },
        "golden_source": {
            "single_lightgbm": "prompt exact M3.2 rerun value",
            "ensemble": "tests/golden_metrics.json m3_4_ensemble_80_20_offline_auc",
        },
    },
    "50_50_mod2": {
        "protocol": "validation buildings are building_id % 2 == 1",
        "golden": {
            "single_lightgbm": M3_2A_50_50_LIGHTGBM_REFERENCE_AUC,
            "ensemble": M3_50_50_ENSEMBLE_GOLDEN_AUC,
        },
        "golden_source": {
            "single_lightgbm": "scripts/run_m3_50_50_ensemble.py reference",
            "ensemble": "tests/golden_metrics.json m3_50_50_offline_ensemble_auc",
        },
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "inv1_meter_aware_impact.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLITS),
        default=list(SPLITS),
        help="Splits to run.",
    )
    return parser.parse_args()


def split_mask(df: pd.DataFrame, name: str) -> np.ndarray:
    if name == "80_20_mod5":
        return (df["building_id"] % 5 == 4).to_numpy()
    if name == "50_50_mod2":
        return (df["building_id"] % 2 == 1).to_numpy()
    raise ValueError(f"Unknown split: {name}")


def split_summary(df: pd.DataFrame, mask_val: np.ndarray, name: str) -> dict[str, Any]:
    train_buildings = set(df.loc[~mask_val, "building_id"].unique())
    val_buildings = set(df.loc[mask_val, "building_id"].unique())
    overlap = assert_no_building_overlap(
        train_buildings, val_buildings, split_name=name
    )
    return {
        "name": name,
        "protocol": SPLITS[name]["protocol"],
        "n_train_buildings": int(len(train_buildings)),
        "n_val_buildings": int(len(val_buildings)),
        "n_train_rows": int((~mask_val).sum()),
        "n_val_rows": int(mask_val.sum()),
        "train_anomaly_rate": float(df.loc[~mask_val, "anomaly"].mean()),
        "val_anomaly_rate": float(df.loc[mask_val, "anomaly"].mean()),
        "building_overlap": int(len(overlap)),
    }


def fit_single_lightgbm(
    train_full: pd.DataFrame,
    val_full: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    y_train = train_full["anomaly"]
    y_val = val_full["anomaly"]
    ds_idx = downsample_indices(y_train)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[ds_idx, feature_cols])
    x_val = scaler.transform(val_full[feature_cols])

    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=100,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train.loc[ds_idx])
    pred = model.predict_proba(x_val)[:, 1]
    return {
        **classification_metrics(y_val, pred),
        "n_train_downsampled": int(len(ds_idx)),
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }


def fit_ensemble(
    train_full: pd.DataFrame,
    val_full: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    y_train = train_full["anomaly"]
    y_val = val_full["anomaly"]
    ds_idx = downsample_indices(y_train)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[ds_idx, feature_cols])
    x_val = scaler.transform(val_full[feature_cols])
    run = fit_predict_models(
        x_train,
        y_train.loc[ds_idx],
        x_val,
        y_val,
        RANDOM_STATE,
    )
    run["n_train_downsampled"] = int(len(ds_idx))
    return run


def run_regime(
    df: pd.DataFrame,
    mask_val: np.ndarray,
    regime: str,
) -> dict[str, Any]:
    log(f"  building feature table: {regime}")
    t0 = time.time()
    train_full = add_value_change_features(
        df.loc[~mask_val],
        list(SHIFTS),
        value_change_regime=regime,
    )
    val_full = add_value_change_features(
        df.loc[mask_val],
        list(SHIFTS),
        value_change_regime=regime,
    )
    value_cols = [c for c in train_full.columns if c.startswith("lag_value_")]
    feature_cols = BASELINE_FEATURE_COLS + value_cols
    if len(feature_cols) != 137:
        raise AssertionError(f"Expected 137 M3.2 features, got {len(feature_cols)}")

    log(f"  fitting single LightGBM: {regime}")
    single = fit_single_lightgbm(train_full, val_full, feature_cols)
    log(
        f"    single AUC={single['val_auc']:.6f} "
        f"P/R/F1={single['precision_05']:.4f}/"
        f"{single['recall_05']:.4f}/{single['f1_05']:.4f}"
    )

    log(f"  fitting 4-model ensemble: {regime}")
    ensemble = fit_ensemble(train_full, val_full, feature_cols)
    log(f"    ensemble AUC={ensemble['ensemble']['val_auc']:.6f}")

    return {
        "value_change_regime": regime,
        "single_lightgbm": single,
        "ensemble": ensemble,
        "feature_counts": {
            "baseline": int(len(BASELINE_FEATURE_COLS)),
            "value_change": int(len(value_cols)),
            "total": int(len(feature_cols)),
        },
        "elapsed_minutes": round((time.time() - t0) / 60, 3),
    }


def unequal_or_nan_mismatch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_nan = np.isnan(left)
    right_nan = np.isnan(right)
    return (left_nan != right_nan) | (~left_nan & ~right_nan & (left != right))


def finite_abs_delta(
    left: np.ndarray, right: np.ndarray, changed: np.ndarray
) -> np.ndarray:
    finite = changed & np.isfinite(left) & np.isfinite(right)
    return np.abs(left[finite] - right[finite])


def summarize_delta(values: list[np.ndarray]) -> dict[str, Any]:
    if not values:
        return {
            "n_finite_changed_cells": 0,
            "median_abs_delta": None,
            "p95_abs_delta": None,
        }
    arr = np.concatenate([v for v in values if len(v)])
    if not len(arr):
        return {
            "n_finite_changed_cells": 0,
            "median_abs_delta": None,
            "p95_abs_delta": None,
        }
    return {
        "n_finite_changed_cells": int(len(arr)),
        "median_abs_delta": float(np.median(arr)),
        "p95_abs_delta": float(np.percentile(arr, 95)),
    }


def shifted_reading_aligned(
    part: pd.DataFrame, shift_hours: int, group_keys: list[str]
) -> np.ndarray:
    work = part[
        ["row_id", "building_id", "meter", "timestamp", "meter_reading"]
    ].sort_values(
        group_keys + ["timestamp"],
        kind="mergesort",
    )
    shifted = work.groupby(group_keys, sort=False)["meter_reading"].shift(shift_hours)
    out = np.empty(len(part), dtype="float32")
    out.fill(np.nan)
    out[work["row_id"].to_numpy()] = shifted.to_numpy(dtype="float32", na_value=np.nan)
    return out


def contamination_for_part(part: pd.DataFrame) -> dict[str, Any]:
    if len(part) == 0:
        return {}
    part = part.reset_index(drop=True).copy()
    part["row_id"] = np.arange(len(part), dtype=np.int64)
    meter_counts = part.groupby("building_id", sort=False)["meter"].nunique()
    multi_meter_buildings = meter_counts[meter_counts > 1].index
    multi_meter_rows = part["building_id"].isin(multi_meter_buildings)

    total_diff_cells = 0
    changed_diff_cells = 0
    nan_mismatch_diff_cells = 0
    total_ratio_cells = 0
    changed_ratio_cells = 0
    nan_mismatch_ratio_cells = 0
    diff_deltas: list[np.ndarray] = []
    ratio_deltas: list[np.ndarray] = []
    current = part["meter_reading"].to_numpy(dtype="float32")

    for shift in SHIFTS:
        row_shifted = shifted_reading_aligned(part, shift, ["building_id"])
        meter_shifted = shifted_reading_aligned(part, shift, ["building_id", "meter"])

        row_diff = current - row_shifted
        meter_diff = current - meter_shifted
        diff_changed = unequal_or_nan_mismatch(row_diff, meter_diff)
        diff_nan_mismatch = np.isnan(row_diff) != np.isnan(meter_diff)
        total_diff_cells += int(len(diff_changed))
        changed_diff_cells += int(diff_changed.sum())
        nan_mismatch_diff_cells += int(diff_nan_mismatch.sum())
        diff_deltas.append(finite_abs_delta(row_diff, meter_diff, diff_changed))

        row_ratio = (current + 1) / (row_shifted + 1)
        meter_ratio = (current + 1) / (meter_shifted + 1)
        ratio_changed = unequal_or_nan_mismatch(row_ratio, meter_ratio)
        ratio_nan_mismatch = np.isnan(row_ratio) != np.isnan(meter_ratio)
        total_ratio_cells += int(len(ratio_changed))
        changed_ratio_cells += int(ratio_changed.sum())
        nan_mismatch_ratio_cells += int(ratio_nan_mismatch.sum())
        ratio_deltas.append(finite_abs_delta(row_ratio, meter_ratio, ratio_changed))

    total_cells = total_diff_cells + total_ratio_cells
    changed_cells = changed_diff_cells + changed_ratio_cells
    return {
        "n_rows": int(len(part)),
        "n_buildings": int(part["building_id"].nunique()),
        "n_multi_meter_buildings": int(len(multi_meter_buildings)),
        "multi_meter_building_rate": float(
            len(multi_meter_buildings) / meter_counts.size
        ),
        "n_multi_meter_rows": int(multi_meter_rows.sum()),
        "multi_meter_row_rate": float(multi_meter_rows.mean()),
        "value_change_cells": {
            "total_cells": int(total_cells),
            "changed_cells": int(changed_cells),
            "changed_cell_rate": float(changed_cells / total_cells),
            "nan_mismatch_cells": int(
                nan_mismatch_diff_cells + nan_mismatch_ratio_cells
            ),
            "nan_mismatch_cell_rate": float(
                (nan_mismatch_diff_cells + nan_mismatch_ratio_cells) / total_cells
            ),
        },
        "lag_value_diff_cells": {
            "total_cells": int(total_diff_cells),
            "changed_cells": int(changed_diff_cells),
            "changed_cell_rate": float(changed_diff_cells / total_diff_cells),
            "nan_mismatch_cells": int(nan_mismatch_diff_cells),
            "nan_mismatch_cell_rate": float(nan_mismatch_diff_cells / total_diff_cells),
            **summarize_delta(diff_deltas),
        },
        "lag_value_ratio_cells": {
            "total_cells": int(total_ratio_cells),
            "changed_cells": int(changed_ratio_cells),
            "changed_cell_rate": float(changed_ratio_cells / total_ratio_cells),
            "nan_mismatch_cells": int(nan_mismatch_ratio_cells),
            "nan_mismatch_cell_rate": float(
                nan_mismatch_ratio_cells / total_ratio_cells
            ),
            **summarize_delta(ratio_deltas),
        },
    }


def contamination_summary(df: pd.DataFrame, mask_val: np.ndarray) -> dict[str, Any]:
    log("  measuring feature-layer row_offset vs row_offset_meter_aware deltas")
    train = contamination_for_part(df.loc[~mask_val])
    val = contamination_for_part(df.loc[mask_val])
    return {
        "scope": (
            "Compares row_offset and row_offset_meter_aware value-change cells "
            "after aligning by (building_id, meter, timestamp)."
        ),
        "train": train,
        "val": val,
    }


def auc_summary(split_name: str, regimes: dict[str, Any]) -> dict[str, Any]:
    row = regimes["row_offset"]
    golden = SPLITS[split_name]["golden"]
    out: dict[str, Any] = {
        "noise_floor_auc": NOISE_FLOOR_AUC,
        "golden_auc": golden,
        "golden_source": SPLITS[split_name]["golden_source"],
        "by_regime": {},
    }
    for regime, run in regimes.items():
        single_auc = run["single_lightgbm"]["val_auc"]
        ensemble_auc = run["ensemble"]["ensemble"]["val_auc"]
        member_ranking = run["ensemble"]["ranking"]
        detector_ranking = sorted(
            {
                "single_lightgbm": single_auc,
                "ensemble": ensemble_auc,
            },
            key={"single_lightgbm": single_auc, "ensemble": ensemble_auc}.get,
            reverse=True,
        )
        out["by_regime"][regime] = {
            "single_lightgbm_auc": single_auc,
            "single_delta_vs_row_offset": single_auc
            - row["single_lightgbm"]["val_auc"],
            "single_delta_vs_golden": single_auc - golden["single_lightgbm"],
            "single_gate_vs_row_offset": "within_noise_floor"
            if abs(single_auc - row["single_lightgbm"]["val_auc"]) <= NOISE_FLOOR_AUC
            else "outside_noise_floor",
            "ensemble_auc": ensemble_auc,
            "ensemble_delta_vs_row_offset": ensemble_auc
            - row["ensemble"]["ensemble"]["val_auc"],
            "ensemble_delta_vs_golden": ensemble_auc - golden["ensemble"],
            "ensemble_gate_vs_row_offset": "within_noise_floor"
            if abs(ensemble_auc - row["ensemble"]["ensemble"]["val_auc"])
            <= NOISE_FLOOR_AUC
            else "outside_noise_floor",
            "detector_ranking": detector_ranking,
            "detector_ranking_changed_vs_row_offset": detector_ranking
            != sorted(
                {
                    "single_lightgbm": row["single_lightgbm"]["val_auc"],
                    "ensemble": row["ensemble"]["ensemble"]["val_auc"],
                },
                key={
                    "single_lightgbm": row["single_lightgbm"]["val_auc"],
                    "ensemble": row["ensemble"]["ensemble"]["val_auc"],
                }.get,
                reverse=True,
            ),
            "ensemble_member_ranking": member_ranking,
            "ensemble_member_ranking_changed_vs_row_offset": member_ranking
            != row["ensemble"]["ranking"],
        }
    return out


def split_verdict(summary: dict[str, Any]) -> dict[str, Any]:
    changed = []
    for regime, metrics in summary["by_regime"].items():
        if regime == "row_offset":
            continue
        if metrics["single_gate_vs_row_offset"] != "within_noise_floor":
            changed.append(f"{regime}:single_auc")
        if metrics["ensemble_gate_vs_row_offset"] != "within_noise_floor":
            changed.append(f"{regime}:ensemble_auc")
        if metrics["detector_ranking_changed_vs_row_offset"]:
            changed.append(f"{regime}:detector_ranking")
        if metrics["ensemble_member_ranking_changed_vs_row_offset"]:
            changed.append(f"{regime}:ensemble_member_ranking")
    return {
        "status": "material" if changed else "immaterial",
        "reasons": changed,
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    if len(SHIFTS) != 60:
        raise AssertionError("Unexpected value-change shift set")

    df = load_m3_frame()
    results: dict[str, Any] = {
        "experiment": "inv1_meter_aware_impact",
        "issue": 51,
        "scope": (
            "Quantify whether row_offset meter-crossing affects M3 headline AUC, "
            "model ranking, or value-change feature quality before M6.3 "
            "GBDT-vs-TabPFN comparison."
        ),
        "noise_floor_auc": NOISE_FLOOR_AUC,
        "random_state": RANDOM_STATE,
        "model_seed": RANDOM_STATE,
        "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
        "value_change_regimes": list(REGIMES),
        "splits": {},
    }

    for split_name in args.splits:
        log(f"Split: {split_name}")
        split_t0 = time.time()
        mask_val = split_mask(df, split_name)
        split_out: dict[str, Any] = {
            "split": split_summary(df, mask_val, split_name),
            "regimes": {},
        }
        for regime in REGIMES:
            split_out["regimes"][regime] = run_regime(df, mask_val, regime)
        split_out["auc_layer"] = auc_summary(split_name, split_out["regimes"])
        split_out["feature_layer"] = contamination_summary(df, mask_val)
        split_out["verdict"] = split_verdict(split_out["auc_layer"])
        split_out["elapsed_minutes"] = round((time.time() - split_t0) / 60, 3)
        results["splits"][split_name] = split_out
        log(
            f"Split {split_name} verdict={split_out['verdict']['status']} "
            f"elapsed={split_out['elapsed_minutes']:.1f} min"
        )

    material_splits = [
        name
        for name, split in results["splits"].items()
        if split["verdict"]["status"] == "material"
    ]
    results["overall_verdict"] = {
        "status": "material" if material_splits else "immaterial",
        "material_splits": material_splits,
        "decision_rule": (
            "Material if any split has |Delta AUC vs row_offset| > 0.0005 "
            "or any model ranking changes."
        ),
        "default_change": (
            "none in this ablation; timestamp_merge is the canonical M3 default, "
            "while row_offset remains the historical comparison arm"
        ),
        "follow_up_required": bool(material_splits),
    }
    results["elapsed_minutes"] = round((time.time() - t0) / 60, 3)

    write_json_with_provenance(
        args.out,
        results,
        root=ROOT,
        provenance={
            "command": (
                f"uv run python scripts/run_inv1_meter_aware_impact.py --out {args.out}"
            ),
        },
    )
    log(f"Saved {args.out}")
    log(f"Overall verdict: {results['overall_verdict']['status']}")


if __name__ == "__main__":
    main()
