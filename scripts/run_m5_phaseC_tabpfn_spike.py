"""Run the M5 Phase C TabPFN local feasibility spike on the M3 feature table."""

from __future__ import annotations

import argparse
import importlib.util
from importlib import metadata
import os
import subprocess
import time
import traceback
from pathlib import Path
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
    downsample_indices,
    load_m3_frame,
    write_json_with_provenance,
)

VALUE_CHANGE_REGIME = "row_offset"
SPLIT_NAME = "80_20_mod5"
TABPFN3_LIMITS = [
    {"max_rows": 1_000_000, "max_features": 200},
    {"max_rows": 100_000, "max_features": 2_000},
    {"max_rows": 1_000, "max_features": 20_000},
]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_model_path = (
        Path(os.environ["TABPFN_MODEL_CACHE_DIR"])
        / "tabpfn-v3-classifier-v3_default.ckpt"
        if os.environ.get("TABPFN_MODEL_CACHE_DIR")
        else None
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC / "m5_phaseC_tabpfn_spike.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--max-fit-rows",
        type=int,
        default=1000,
        help=(
            "Maximum downsampled training rows used for the paired local model "
            "comparison. The full M3 downsample shape is still measured."
        ),
    )
    parser.add_argument(
        "--max-val-rows",
        type=int,
        default=20000,
        help=(
            "Maximum validation rows scored in the local spike. Use 0 to score "
            "the full validation split."
        ),
    )
    parser.add_argument(
        "--tabpfn-batch-size",
        type=int,
        default=256,
        help="Prediction batch size recorded for the TabPFN latency signal.",
    )
    parser.add_argument(
        "--subsample-seed",
        type=int,
        default=RANDOM_STATE,
        help="Seed used when reducing train/validation rows for local feasibility.",
    )
    parser.add_argument(
        "--skip-tabpfn",
        action="store_true",
        help="Only measure the feature/split/sample path and GBDT anchor.",
    )
    parser.add_argument(
        "--allow-tabpfn-failure",
        action="store_true",
        help="Archive TabPFN failure evidence instead of exiting non-zero.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model_path,
        help=(
            "Local TabPFN checkpoint path. Defaults to "
            "$TABPFN_MODEL_CACHE_DIR/tabpfn-v3-classifier-v3_default.ckpt."
        ),
    )
    return parser.parse_args()


def tabpfn_limit_fit(n_rows: int, n_features: int) -> dict[str, Any]:
    fits = [
        limit
        for limit in TABPFN3_LIMITS
        if n_rows <= limit["max_rows"] and n_features <= limit["max_features"]
    ]
    return {
        "fits_documented_tabpfn3_limit": bool(fits),
        "matching_limits": fits,
        "documented_limits": TABPFN3_LIMITS,
    }


def sample_indices(indices: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or len(indices) <= max_rows:
        return indices
    rng = np.random.RandomState(seed)
    selected = rng.choice(indices, size=max_rows, replace=False)
    return np.asarray(selected)


def balanced_subsample_indices(
    ds_idx: np.ndarray, y, max_rows: int, seed: int
) -> np.ndarray:
    if max_rows <= 0 or len(ds_idx) <= max_rows:
        return ds_idx
    if max_rows < 2:
        raise ValueError("--max-fit-rows must be at least 2")
    rng = np.random.RandomState(seed)
    y_ds = y.loc[ds_idx].to_numpy()
    pos_positions = np.flatnonzero(y_ds == 1)
    neg_positions = np.flatnonzero(y_ds == 0)
    per_class = max_rows // 2
    n_pos = min(per_class, len(pos_positions))
    n_neg = min(max_rows - n_pos, len(neg_positions))
    chosen_pos = rng.choice(pos_positions, size=n_pos, replace=False)
    chosen_neg = rng.choice(neg_positions, size=n_neg, replace=False)
    chosen = np.concatenate([chosen_neg, chosen_pos])
    rng.shuffle(chosen)
    return ds_idx[chosen]


def torch_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "torch_installed": False,
        "tabpfn_installed": importlib.util.find_spec("tabpfn") is not None,
        "tabpfn_token_present": bool(os.environ.get("TABPFN_TOKEN")),
        "execution_path": "local",
        "cloud_client_used": False,
        "tabpfn_no_browser": bool(os.environ.get("TABPFN_NO_BROWSER")),
        "tabpfn_disable_telemetry": bool(os.environ.get("TABPFN_DISABLE_TELEMETRY")),
    }
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        gpu_rows = [line.strip() for line in smi.stdout.splitlines() if line.strip()]
        env["nvidia_smi_gpus"] = gpu_rows
    except (OSError, subprocess.CalledProcessError):
        env["nvidia_smi_gpus"] = []
    if importlib.util.find_spec("torch") is None:
        return env
    import torch

    env.update(
        {
            "torch_installed": True,
            "tabpfn_version": metadata.version("tabpfn")
            if env["tabpfn_installed"]
            else None,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
    )
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        env.update(
            {
                "device": "cuda",
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_vram_mib": int(props.total_memory // (1024 * 1024)),
            }
        )
    else:
        env["device"] = "cpu"
    return env


def threshold_05_counts(y_true, pred: np.ndarray) -> dict[str, int]:
    y = np.asarray(y_true)
    pred_label = (pred >= 0.5).astype("int8")
    return {
        "threshold_05_true_positive": int(((pred_label == 1) & (y == 1)).sum()),
        "threshold_05_false_positive": int(((pred_label == 1) & (y == 0)).sum()),
        "threshold_05_true_negative": int(((pred_label == 0) & (y == 0)).sum()),
        "threshold_05_false_negative": int(((pred_label == 0) & (y == 1)).sum()),
        "threshold_05_predicted_positive": int((pred_label == 1).sum()),
        "threshold_05_predicted_negative": int((pred_label == 0).sum()),
    }


def fit_gbdt(x_train, y_train, x_val, y_val) -> dict[str, Any]:
    t0 = time.perf_counter()
    model = lgb.LGBMClassifier(
        n_estimators=100,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    pred = model.predict_proba(x_val)[:, 1]
    elapsed = time.perf_counter() - t0
    return {
        **classification_metrics(y_val, pred),
        **threshold_05_counts(y_val, pred),
        "fit_predict_seconds": float(elapsed),
    }


def tabpfn_classifier(device: str, model_path: Path | None):
    from tabpfn import TabPFNClassifier

    kwargs: dict[str, Any] = {"device": device}
    if model_path is not None:
        kwargs["model_path"] = model_path
    try:
        return TabPFNClassifier(**kwargs)
    except TypeError:
        if model_path is not None:
            raise
        return TabPFNClassifier()


def fit_tabpfn(
    x_train, y_train, x_val, y_val, *, device: str, model_path: Path | None
) -> dict[str, Any]:
    t0 = time.perf_counter()
    t_init0 = time.perf_counter()
    model = tabpfn_classifier(device, model_path)
    init_elapsed = time.perf_counter() - t_init0
    t_fit0 = time.perf_counter()
    model.fit(x_train, y_train)
    fit_elapsed = time.perf_counter() - t_fit0
    t_predict0 = time.perf_counter()
    pred = model.predict_proba(x_val)[:, 1]
    predict_elapsed = time.perf_counter() - t_predict0
    elapsed = time.perf_counter() - t0
    return {
        **classification_metrics(y_val, pred),
        **threshold_05_counts(y_val, pred),
        "cold_start": True,
        "timing_note": (
            "fit_predict_seconds includes local checkpoint model initialization, "
            "fit, and predict_proba in this process."
        ),
        "model_init_seconds": float(init_elapsed),
        "fit_seconds": float(fit_elapsed),
        "predict_proba_seconds": float(predict_elapsed),
        "fit_predict_seconds": float(elapsed),
    }


def build_feature_table() -> dict[str, Any]:
    df = load_m3_frame(verbose=True)
    mask_val = (df["building_id"] % 5 == 4).to_numpy()
    train_buildings = set(df.loc[~mask_val, "building_id"].unique())
    val_buildings = set(df.loc[mask_val, "building_id"].unique())
    overlap = assert_no_building_overlap(
        train_buildings, val_buildings, split_name=SPLIT_NAME
    )

    log(f"Adding M3.2 value-change features: {VALUE_CHANGE_REGIME}")
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
    feature_cols = BASELINE_FEATURE_COLS + value_cols
    if len(feature_cols) != 137:
        raise AssertionError(f"Expected 137 M3.2 features, got {len(feature_cols)}")

    return {
        "df": df,
        "mask_val": mask_val,
        "train_full": train_full,
        "val_full": val_full,
        "feature_cols": feature_cols,
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
    }


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    model_path = args.model_path.resolve() if args.model_path is not None else None
    local_checkpoint_available = bool(model_path is not None and model_path.is_file())
    table = build_feature_table()
    train_full = table["train_full"]
    val_full = table["val_full"]
    feature_cols = table["feature_cols"]

    y_train_full = train_full["anomaly"]
    y_val_full = val_full["anomaly"]
    ds_idx_full = downsample_indices(y_train_full)
    full_shape = {
        "downsampled_train_rows": int(len(ds_idx_full)),
        "feature_count": int(len(feature_cols)),
        **tabpfn_limit_fit(len(ds_idx_full), len(feature_cols)),
    }

    fit_idx = balanced_subsample_indices(
        ds_idx_full,
        y_train_full,
        args.max_fit_rows,
        args.subsample_seed,
    )
    val_idx = sample_indices(
        val_full.index.to_numpy(),
        args.max_val_rows,
        args.subsample_seed,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_full.loc[fit_idx, feature_cols])
    x_val = scaler.transform(val_full.loc[val_idx, feature_cols])
    y_train = y_train_full.loc[fit_idx]
    y_val = y_val_full.loc[val_idx]

    local_shape = {
        "train_rows": int(len(fit_idx)),
        "validation_rows": int(len(val_idx)),
        "feature_count": int(len(feature_cols)),
        "subsample_seed": int(args.subsample_seed),
        "max_fit_rows": int(args.max_fit_rows),
        "max_val_rows": int(args.max_val_rows),
        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_val.mean()),
        **tabpfn_limit_fit(len(fit_idx), len(feature_cols)),
    }

    log(
        "Local paired table: "
        f"{local_shape['train_rows']} train x {local_shape['feature_count']} features; "
        f"{local_shape['validation_rows']} validation rows"
    )
    gbdt = fit_gbdt(x_train, y_train, x_val, y_val)
    log(
        "GBDT: "
        f"AUC={gbdt['val_auc']:.4f} "
        f"P/R/F1={gbdt['precision_05']:.4f}/"
        f"{gbdt['recall_05']:.4f}/{gbdt['f1_05']:.4f}"
    )

    env = torch_environment()
    env.update(
        {
            "tabpfn_model_path": str(model_path) if model_path is not None else None,
            "tabpfn_model_path_exists": local_checkpoint_available,
        }
    )
    tabpfn: dict[str, Any]
    if args.skip_tabpfn:
        tabpfn = {"status": "skipped_by_flag"}
    elif not env["tabpfn_installed"] or not env["torch_installed"]:
        tabpfn = {
            "status": "not_run_missing_optional_dependency",
            "missing": [
                name for name in ("torch", "tabpfn") if not env[f"{name}_installed"]
            ],
        }
    elif not env["tabpfn_token_present"] and not local_checkpoint_available:
        tabpfn = {
            "status": "not_run_missing_weights_and_token",
            "reason": (
                "TABPFN_TOKEN is not set and no readable local checkpoint was "
                "provided. The noninteractive runner does not launch the "
                "browser-based license flow."
            ),
        }
    else:
        device = str(env.get("device", "cpu"))
        try:
            tabpfn = {
                "status": "completed",
                "batch_size": int(args.tabpfn_batch_size),
                "device": device,
                "model_path": str(model_path) if model_path is not None else None,
                **fit_tabpfn(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    device=device,
                    model_path=model_path if local_checkpoint_available else None,
                ),
            }
            log(
                "TabPFN: "
                f"AUC={tabpfn['val_auc']:.4f} "
                f"P/R/F1={tabpfn['precision_05']:.4f}/"
                f"{tabpfn['recall_05']:.4f}/{tabpfn['f1_05']:.4f}"
            )
        except Exception as exc:  # pragma: no cover - depends on local license/GPU.
            tabpfn = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
                "batch_size": int(args.tabpfn_batch_size),
                "device": device,
                "model_path": str(model_path) if model_path is not None else None,
            }

    results = {
        "experiment": "m5_phaseC_tabpfn_spike",
        "issue": 30,
        "scope": "LEAD/M3-only local feasibility spike; no BDG2 data used",
        "canonical_line": f"{SPLIT_NAME}_offline_{VALUE_CHANGE_REGIME}",
        "random_state": RANDOM_STATE,
        "downsampling_seeds": list(DOWNSAMPLE_SEEDS),
        "feature_table": {
            "source": "load_m3_frame plus add_value_change_features",
            "value_change_regime": VALUE_CHANGE_REGIME,
            "baseline_features": int(len(BASELINE_FEATURE_COLS)),
            "value_change_features": int(len(SHIFTS) * 2),
            "total_features": int(len(feature_cols)),
            "upstream_feature_construction_changed": False,
        },
        "split": table["split"],
        "tabpfn3_fit": {
            "full_downsampled_m3": full_shape,
            "local_paired_comparison": local_shape,
            "interpretation": (
                "The full M3 downsample shape answers documented TabPFN-3 fit; "
                "the local paired shape is reduced for laptop feasibility."
            ),
        },
        "environment": env,
        "models": {
            "gbdt_lightgbm_anchor": {
                "status": "completed",
                "model": "LightGBM LGBMClassifier(n_estimators=100)",
                "same_local_table_as_tabpfn": True,
                **gbdt,
            },
            "tabpfn": tabpfn,
        },
        "latency_signal": {
            "tabpfn_fit_predict_seconds": tabpfn.get("fit_predict_seconds"),
            "tabpfn_cold_start": tabpfn.get("cold_start"),
            "tabpfn_model_init_seconds": tabpfn.get("model_init_seconds"),
            "tabpfn_fit_seconds": tabpfn.get("fit_seconds"),
            "tabpfn_predict_proba_seconds": tabpfn.get("predict_proba_seconds"),
            "train_rows": int(len(fit_idx)),
            "feature_count": int(len(feature_cols)),
            "batch_size": int(args.tabpfn_batch_size),
            "device": tabpfn.get("device", env.get("device")),
        },
        "one_shot_discipline": {
            "leaderboard_probing": False,
            "cloud_client_used": False,
            "bdg2_used": False,
        },
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    write_json_with_provenance(
        args.out,
        results,
        root=ROOT,
        provenance={
            "command": (
                "python scripts/run_m5_phaseC_tabpfn_spike.py "
                f"--max-fit-rows {args.max_fit_rows} "
                f"--max-val-rows {args.max_val_rows}"
                f" --tabpfn-batch-size {args.tabpfn_batch_size}"
                + (f" --model-path {model_path}" if model_path is not None else "")
            ),
        },
    )
    log(f"Saved {args.out}")

    if (
        tabpfn.get("status") != "completed"
        and not args.skip_tabpfn
        and not args.allow_tabpfn_failure
    ):
        raise RuntimeError(f"TabPFN did not complete: {tabpfn}")


if __name__ == "__main__":
    main()
