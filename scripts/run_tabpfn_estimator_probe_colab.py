"""Self-contained Colab worker for the TabPFN n_estimators probe.

Reads the portable probe inputs npz and the foundation checkpoint from its own
directory, runs the {estimators} sweep for the requested sites, and writes a
results json. Imports torch/tabpfn only inside main and requires CUDA.

Run on the VM as, e.g.:
    python run_tabpfn_estimator_probe_colab.py \
        --inputs /content/probe/tabpfn_estimator_probe_inputs.npz \
        --model /content/probe/tabpfn-v3-classifier-v3_default.ckpt \
        --sites 1 2 --estimators 1 4 8 \
        --out /content/probe/probe_results_a.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def build_model(model_path: str, n_estimators: int, seed: int):
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        n_estimators=n_estimators,
        auto_scale_n_estimators=False,
        model_path=model_path,
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
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sites", type=int, nargs="+", required=True)
    parser.add_argument("--estimators", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--microbatch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; probe requires a GPU runtime")

    data = np.load(args.inputs)
    x_context = data["x_context"].astype("float32")
    y_context = data["y_context"].astype("int64")
    sites = {
        s: (data[f"x_site_{s}"].astype("float32"), data[f"y_site_{s}"].astype("int64"))
        for s in args.sites
    }
    for s, (x, y) in sites.items():
        print(f"site {s}: {len(y):,} rows, {int(y.sum()):,} positives", flush=True)

    results: dict[str, dict] = {}
    for n in args.estimators:
        print(
            f"=== n_estimators={n}: fit on {len(x_context):,} context ===", flush=True
        )
        t0 = time.time()
        model = build_model(args.model, n, args.seed)
        model.fit(x_context, y_context)
        fit_s = time.time() - t0
        for s, (x, y) in sites.items():
            t1 = time.time()
            scores = predict_scores(model, x, args.microbatch)
            roc = float(roc_auc_score(y, scores))
            pr = float(average_precision_score(y, scores))
            results.setdefault(str(s), {})[str(n)] = {
                "roc_auc": roc,
                "pr_auc": pr,
                "rows": int(len(y)),
                "positives": int(y.sum()),
                "predict_seconds": round(time.time() - t1, 1),
            }
            print(
                f"  site {s}  n={n}  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  "
                f"({time.time() - t1:.0f}s)",
                flush=True,
            )
            # Write partial results after every (site, n) so a dropped connection
            # or a recycled runtime still leaves the completed cells downloadable.
            Path(args.out).write_text(
                json.dumps({"results": results, "partial": True}, indent=2),
                encoding="utf-8",
            )
        del model
        torch.cuda.empty_cache()
        print(f"  n={n} fit {fit_s:.0f}s", flush=True)

    base = str(args.estimators[0])
    summary = {}
    for s in map(str, args.sites):
        b = results[s][base]
        summary[s] = {
            n: {
                "roc_auc": round(results[s][n]["roc_auc"], 4),
                "pr_auc": round(results[s][n]["pr_auc"], 4),
                "d_roc_vs_n1": round(results[s][n]["roc_auc"] - b["roc_auc"], 4),
                "d_pr_vs_n1": round(results[s][n]["pr_auc"] - b["pr_auc"], 4),
            }
            for n in map(str, args.estimators)
        }

    payload = {
        "config": {
            "sites": args.sites,
            "estimators": args.estimators,
            "context_rows": int(len(x_context)),
            "features": int(x_context.shape[1]),
            "seed": args.seed,
        },
        "results": results,
        "summary_vs_n1": summary,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("=== SUMMARY (delta vs n=1) ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
