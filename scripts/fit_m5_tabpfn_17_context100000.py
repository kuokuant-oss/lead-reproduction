"""Fit the canonical 17-feature TabPFN state for one n_estimators value.

The estimator sweep changes exactly one thing relative to the official
17-feature run: the TabPFN ensemble size. Ensemble members are built during
``fit``, so n=4 and n=8 each need their own fitted state. Everything else --
the 100k canonical context rows, the seed, the StandardScaler and the
foundation checkpoint -- is reproduced from the same code path the official run
used, and the resulting scaler is proved identical to the canonical one.

This is a one-time fit on 100k rows, not the forbidden local full-test
inference.
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

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from lead import BASELINE_FEATURE_COLS, PROC, RANDOM_STATE, ROOT, load_m3_frame

CANONICAL = ROOT / "scripts" / "run_m5_tabpfn_canonical_full_test.py"
CANONICAL_WORK = PROC / "m5_tabpfn_canonical_full_test_context100000.work"
EXPERIMENT = "tabpfn_v3_canonical_m3_full_test_17feature_estimator_sweep"


def load_canonical_module():
    spec = importlib.util.spec_from_file_location("m5_canonical_17", CANONICAL)
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


def prove_scaler_matches_canonical(scaler: StandardScaler) -> bool:
    """The sweep is only interpretable if queries are scaled identically."""
    reference_path = CANONICAL_WORK / "scaler.joblib"
    if not reference_path.is_file():
        return False
    reference = joblib.load(reference_path)
    for attribute in ("mean_", "scale_"):
        if not np.allclose(
            getattr(reference, attribute), getattr(scaler, attribute), rtol=0, atol=0
        ):
            raise AssertionError(
                f"refit scaler {attribute} differs from the canonical 17-feature run"
            )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-estimators", type=int, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="default: m5_tabpfn_canonical_full_test_context100000_n<k>.work",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / ".tabpfn-cache" / "tabpfn-v3-classifier-v3_default.ckpt",
    )
    args = parser.parse_args()
    if args.n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1, got {args.n_estimators}")
    if args.n_estimators == 1:
        raise ValueError(
            f"n=1 is the official run; reuse {CANONICAL_WORK} instead of refitting it"
        )
    if args.work_dir is None:
        args.work_dir = PROC / (
            f"m5_tabpfn_canonical_full_test_context100000_n{args.n_estimators}.work"
        )
    args.work_dir.mkdir(parents=True, exist_ok=True)

    mod = load_canonical_module()
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

    feature_cols = list(BASELINE_FEATURE_COLS)
    if len(feature_cols) != 17:
        raise AssertionError(f"expected 17 features, got {len(feature_cols)}")

    print(f"building {args.context_rows}x17 context matrix", flush=True)
    x_context = frame.loc[context_index, feature_cols].to_numpy(
        dtype="float32", copy=True
    )
    y_context = frame.loc[context_index, "anomaly"].to_numpy(dtype="int64", copy=True)
    del frame

    scaler = StandardScaler(copy=False)
    x_context = scaler.fit_transform(x_context).astype("float32", copy=False)
    scaler_proved = prove_scaler_matches_canonical(scaler)
    print(f"scaler matches canonical 17-feature run: {scaler_proved}", flush=True)

    print(f"creating foundation model (n_estimators={args.n_estimators})", flush=True)
    model = mod.create_real_model(args.model_path, args.seed, args.n_estimators)
    model.fit(x_context, y_context)
    context = mod.verify_fitted_context(model, args.context_rows, args.n_estimators)
    if context["status"] != "verified":
        raise RuntimeError("new fitted state failed context verification")
    if context["effective_estimators"] != args.n_estimators:
        raise RuntimeError(
            f"fitted state carries {context['effective_estimators']} estimators, "
            f"expected {args.n_estimators}"
        )

    mod.atomic_joblib_dump(scaler, args.work_dir / "scaler.joblib")
    mod.atomic_save_fitted_model(model, args.work_dir / "model.tabpfn_fit")

    manifest = {
        "experiment": EXPERIMENT,
        "context_rows": args.context_rows,
        "context_sha256": contract["metadata"]["context"]["sha256"],
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
        "n_estimators": args.n_estimators,
        "model_path": str(args.model_path.resolve()),
        "model_sha256": sha256_file(args.model_path),
        "seed": args.seed,
        "scaler_matches_canonical_run": scaler_proved,
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
