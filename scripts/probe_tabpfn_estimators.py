"""Probe how TabPFN n_estimators affects per-site ROC-AUC / PR-AUC.

17-feature baseline, 100k context (canonical_contract, seed 42), matched to the
full-test pipeline. For each site and each n_estimators in {1,4,8}, refit on the
same context and score a fixed random sample of that site's test rows. The goal
is the *relative* gain from more estimators, measured cheaply before deciding on
any full re-run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from lead import BASELINE_FEATURE_COLS, PROC, RANDOM_STATE, ROOT, load_m3_frame

CANONICAL = ROOT / "scripts" / "run_m5_tabpfn_canonical_full_test.py"
SITE_PREDICTIONS = PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"
MODEL_PATH = ROOT / ".tabpfn-cache" / "tabpfn-v3-classifier-v3_default.ckpt"


def load_canonical():
    spec = importlib.util.spec_from_file_location("m5_canon_probe", CANONICAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_model(n_estimators: int, seed: int):
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        n_estimators=n_estimators,
        auto_scale_n_estimators=False,
        model_path=str(MODEL_PATH.resolve()),
        device="cuda",
        ignore_pretraining_limits=True,
        fit_mode="low_memory",
        memory_saving_mode=True,
        keep_cache_on_device=False,
        random_state=seed,
        n_preprocessing_jobs=1,
        inference_config={"SUBSAMPLE_SAMPLES": None},
        show_progress_bar=False,
    )


def predict_scores(model, x: np.ndarray, microbatch: int) -> np.ndarray:
    out = np.empty(len(x), dtype="float64")
    for s in range(0, len(x), microbatch):
        e = min(len(x), s + microbatch)
        out[s:e] = model.predict_proba(x[s:e])[:, 1]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, nargs="+", default=[1, 2, 6, 8])
    parser.add_argument("--estimators", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--per-site-rows", type=int, default=50_000)
    parser.add_argument("--context-rows", type=int, default=100_000)
    parser.add_argument("--microbatch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--out", type=Path, default=PROC / "tabpfn_estimator_probe.json"
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
    site_rows: dict[int, dict] = {}
    for s in args.sites:
        idx = np.flatnonzero(site_all == s)
        if len(idx) > args.per_site_rows:
            idx = rng.choice(idx, size=args.per_site_rows, replace=False)
            idx.sort()
        ri = raw_index[idx]
        y = y_all[idx].astype("int64")
        xb = frame.loc[ri, BASELINE_FEATURE_COLS].to_numpy(dtype="float32", copy=True)
        xb = scaler.transform(xb).astype("float32", copy=False)
        site_rows[s] = {"x": xb, "y": y, "n": len(y), "pos": int(y.sum())}
        print(
            f"site {s}: sampled {len(y):,} rows, {int(y.sum()):,} positives", flush=True
        )
    del frame

    results: dict[str, dict] = {}
    for n in args.estimators:
        print(
            f"=== n_estimators={n}: fitting on {args.context_rows} context ===",
            flush=True,
        )
        t0 = time.time()
        model = build_model(n, args.seed)
        model.fit(x_context, y_context)
        fit_s = time.time() - t0
        for s in args.sites:
            d = site_rows[s]
            t1 = time.time()
            scores = predict_scores(model, d["x"], args.microbatch)
            roc = float(roc_auc_score(d["y"], scores))
            pr = float(average_precision_score(d["y"], scores))
            results.setdefault(str(s), {})[str(n)] = {
                "roc_auc": roc,
                "pr_auc": pr,
                "rows": d["n"],
                "positives": d["pos"],
                "predict_seconds": round(time.time() - t1, 1),
            }
            print(
                f"  site {s}  n={n}  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  "
                f"({time.time() - t1:.0f}s)",
                flush=True,
            )
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"  n={n} fit took {fit_s:.0f}s", flush=True)

    # Deltas vs n=1 baseline.
    base = str(args.estimators[0])
    summary = {}
    for s in map(str, args.sites):
        row = {}
        b = results[s][base]
        for n in map(str, args.estimators):
            r = results[s][n]
            row[n] = {
                "roc_auc": round(r["roc_auc"], 4),
                "pr_auc": round(r["pr_auc"], 4),
                "d_roc_vs_n1": round(r["roc_auc"] - b["roc_auc"], 4),
                "d_pr_vs_n1": round(r["pr_auc"] - b["pr_auc"], 4),
            }
        summary[s] = row

    payload = {
        "config": {
            "sites": args.sites,
            "estimators": args.estimators,
            "per_site_rows": args.per_site_rows,
            "context_rows": args.context_rows,
            "features": len(BASELINE_FEATURE_COLS),
            "seed": args.seed,
        },
        "results": results,
        "summary_vs_n1": summary,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=== SUMMARY (delta vs n=1) ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
