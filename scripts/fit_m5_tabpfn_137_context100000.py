"""Fit TabPFN once on the 137-feature offline set for the distributed full test.

This mirrors the canonical 17-feature fit exactly except for the feature matrix:
same 100k context rows (canonical_contract, seed 42), same foundation checkpoint,
same StandardScaler + save layout. The features are the 17 baseline columns plus
the 120 offline value-change columns, i.e. the same 137 the M3 tree ensemble used.

Output work dir mirrors the 17-feature one so the portable exporters can consume
it: model.tabpfn_fit, scaler.joblib, fit_manifest.json.

This is a one-time fit on 100k rows, not the forbidden local full-test inference.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.preprocessing import StandardScaler

from lead import (
    BASELINE_FEATURE_COLS,
    PROC,
    RANDOM_STATE,
    ROOT,
    SHIFTS,
    add_value_change_features,
    load_m3_frame,
)

CANONICAL = ROOT / "scripts" / "run_m5_tabpfn_canonical_full_test.py"
EXPERIMENT = "tabpfn_v3_canonical_m3_full_test_137feature"
VALUE_CHANGE_REGIME = "timestamp_merge"


def load_canonical_module():
    spec = importlib.util.spec_from_file_location("m5_canonical_137", CANONICAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=PROC / "m5_tabpfn_137_full_test_context100000.work",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / ".tabpfn-cache" / "tabpfn-v3-classifier-v3_default.ckpt",
    )
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    mod = load_canonical_module()

    # Canonical args the contract reads, with the same defaults the 17-feature
    # run used, so the 100k context is the identical row set.
    contract_args = SimpleNamespace(
        context_rows=args.context_rows,
        validation_rows=4_000,
        seed=args.seed,
        smoke=False,
        canonical_m3_predictions=PROC / "m3_figure_predictions_50_50.npz",
        canonical_site_predictions=(
            PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"
        ),
        canonical_baseline_predictions=(
            PROC / "m3_17_feature_ensemble_predictions.npz"
        ),
    )

    print("loading M3 frame", flush=True)
    frame = load_m3_frame(verbose=True)
    print("freezing canonical contract / context index", flush=True)
    contract = mod.canonical_contract(frame, contract_args)
    context_index = np.asarray(contract["context_index"], dtype="int64")

    # Value-change features are computed on the train split (building_id % 2 == 0),
    # which is where every context row lives. add_value_change_features sorts and
    # resets the index, so carry the original frame index as a column and restore
    # it afterward -- otherwise .loc[context_index] would silently pull the wrong
    # rows. This matches the feature set run_m3_50_50_ensemble trains the trees on.
    train_mask = (frame["building_id"] % 2 == 0).to_numpy()
    train_df = frame.loc[train_mask].copy()
    train_df["_orig_index"] = train_df.index.to_numpy()
    print("adding 120 offline value-change features on the train split", flush=True)
    train_full = add_value_change_features(
        train_df, list(SHIFTS), value_change_regime=VALUE_CHANGE_REGIME
    )
    train_full = train_full.set_index("_orig_index")
    value_cols = [c for c in train_full.columns if c.startswith("lag_value_")]
    if len(value_cols) != 120:
        raise AssertionError(
            f"expected 120 value-change features, got {len(value_cols)}"
        )
    feature_cols = list(BASELINE_FEATURE_COLS) + value_cols
    if len(feature_cols) != 137:
        raise AssertionError(f"expected 137 features, got {len(feature_cols)}")

    missing = set(context_index.tolist()) - set(train_full.index.tolist())
    if missing:
        raise AssertionError(f"{len(missing)} context rows are outside the train split")

    print(f"building {args.context_rows}x137 context matrix", flush=True)
    x_context = train_full.loc[context_index, feature_cols].to_numpy(
        dtype="float32", copy=True
    )
    y_context = frame.loc[context_index, "anomaly"].to_numpy(dtype="int64", copy=True)
    # Value-change merge misses are legitimately NaN (first hours / gaps). Trees
    # and TabPFN both handle missing values natively, and StandardScaler ignores
    # NaN when computing statistics, so it is not treated as an error here.
    nan_fraction = float(np.isnan(x_context).mean())
    print(f"context NaN fraction = {nan_fraction:.4%}", flush=True)

    scaler = StandardScaler(copy=False)
    x_context = scaler.fit_transform(x_context).astype("float32", copy=False)

    print("creating foundation model and fitting", flush=True)
    model = mod.create_real_model(args.model_path, args.seed)
    model.fit(x_context, y_context)
    context = mod.verify_fitted_context(model, args.context_rows)
    if context["status"] != "verified":
        raise RuntimeError("new 137-feature fitted state failed context verification")

    mod.atomic_joblib_dump(scaler, args.work_dir / "scaler.joblib")
    mod.atomic_save_fitted_model(model, args.work_dir / "model.tabpfn_fit")

    manifest = {
        "experiment": EXPERIMENT,
        "context_rows": args.context_rows,
        "context_sha256": contract["metadata"]["context"]["sha256"],
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
        "model_path": str(args.model_path.resolve()),
        "model_sha256": sha256_file(args.model_path),
        "seed": args.seed,
        "value_change_regime": VALUE_CHANGE_REGIME,
    }
    manifest_path = args.work_dir / "fit_manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)

    del x_context, y_context
    gc.collect()
    print(f"fit complete -> {args.work_dir}", flush=True)
    print(json.dumps({"context_verification": context}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
