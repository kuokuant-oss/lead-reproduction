"""Build the small, portable inputs for the TabPFN n_estimators probe.

Produces one npz holding the scaled 100k context (x, y) and, per probe site, a
scaled random sample of that site's test rows (x, y). Everything is 17-feature,
built with the canonical 100k context (seed 42) so it matches the full-test
pipeline. The npz is tiny (~20 MB) and self-contained, so the actual TabPFN
sweep can run on Colab without the M3 frame.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.preprocessing import StandardScaler

from lead import BASELINE_FEATURE_COLS, PROC, RANDOM_STATE, ROOT, load_m3_frame

CANONICAL = ROOT / "scripts" / "run_m5_tabpfn_canonical_full_test.py"
SITE_PREDICTIONS = PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"


def load_canonical():
    spec = importlib.util.spec_from_file_location("m5_canon_probe_build", CANONICAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, nargs="+", default=[1, 2, 6, 8])
    parser.add_argument("--per-site-rows", type=int, default=50_000)
    parser.add_argument("--context-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--out", type=Path, default=PROC / "tabpfn_estimator_probe_inputs.npz"
    )
    args = parser.parse_args()

    mod = load_canonical()
    contract_args = SimpleNamespace(
        context_rows=args.context_rows,
        validation_rows=4_000,
        seed=args.seed,
        smoke=False,
        canonical_m3_predictions=PROC / "m3_figure_predictions_50_50.npz",
        canonical_site_predictions=SITE_PREDICTIONS,
        canonical_baseline_predictions=PROC / "m3_17_feature_ensemble_predictions.npz",
    )

    print("loading frame", flush=True)
    frame = load_m3_frame(verbose=True)
    contract = mod.canonical_contract(frame, contract_args)
    context_index = np.asarray(contract["context_index"], dtype="int64")

    x_context = frame.loc[context_index, BASELINE_FEATURE_COLS].to_numpy(
        dtype="float32", copy=True
    )
    y_context = frame.loc[context_index, "anomaly"].to_numpy(dtype="int64", copy=True)
    scaler = StandardScaler(copy=False)
    x_context = scaler.fit_transform(x_context).astype("float32", copy=False)

    with np.load(SITE_PREDICTIONS) as site:
        raw_index = np.asarray(site["validation_raw_index"], dtype="int64")
        y_all = np.asarray(site["anomaly"], dtype="int8")
        site_all = np.asarray(site["site_id"], dtype="int8")

    rng = np.random.default_rng(args.seed)
    arrays: dict[str, np.ndarray] = {
        "x_context": x_context,
        "y_context": y_context.astype("int8"),
        "sites": np.asarray(args.sites, dtype="int16"),
    }
    for s in args.sites:
        idx = np.flatnonzero(site_all == s)
        if len(idx) > args.per_site_rows:
            idx = rng.choice(idx, size=args.per_site_rows, replace=False)
            idx.sort()
        ri = raw_index[idx]
        y = y_all[idx].astype("int8")
        xb = frame.loc[ri, BASELINE_FEATURE_COLS].to_numpy(dtype="float32", copy=True)
        xb = scaler.transform(xb).astype("float32", copy=False)
        arrays[f"x_site_{s}"] = xb
        arrays[f"y_site_{s}"] = y
        print(f"site {s}: {len(y):,} rows, {int(y.sum()):,} positives", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)
    total_mb = args.out.stat().st_size / 1e6
    print(f"saved {args.out} ({total_mb:.1f} MB)", flush=True)
    print(
        f"context: {x_context.shape}, features={len(BASELINE_FEATURE_COLS)}", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
