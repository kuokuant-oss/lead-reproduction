"""Verify saved Path-A states by reloading and rescoring the screening query.

No model is fitted, no context is changed, and existing screening predictions
are preserved as the comparison target.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from lead import ROOT
from lead.m5_context import query_paths
from run_m5_story_ae_probe import (
    build_feature_matrix,
    load_tree_runner,
    validate_feature_matrix,
)


OUT = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"


def main() -> int:
    recovery = OUT / "recovery"
    query_manifest_path, query_path = query_paths(
        ROOT / "data" / "processed" / "m5_context_stories", "screening"
    )
    query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
    with np.load(query_path) as payload:
        raw, y = (
            np.asarray(payload["raw_index"], dtype="int64"),
            np.asarray(payload["anomaly"], dtype="int8"),
        )
    from lead import load_m3_frame

    frame = load_m3_frame(verbose=True)
    holdout = frame.loc[frame["building_id"] % 2 == 1]
    x = build_feature_matrix(holdout, raw, "F4", full_frame=holdout)
    validate_feature_matrix(x, matrix_name="recovery screening reload query")
    rows = []
    for model in ("tabpfn", "trees"):
        for cell in sorted((recovery / "states" / model).glob("seed*/*/*")):
            if not cell.is_dir():
                continue
            scaler = joblib.load(cell / "scaler.joblib")
            transformed = scaler.transform(x).astype("float32", copy=False)
            if model == "tabpfn":
                from tabpfn.model_loading import load_fitted_tabpfn_model

                estimator = load_fitted_tabpfn_model(
                    cell / "model.tabpfn_fit", device="cuda"
                )
                score = np.asarray(
                    estimator.predict_proba(transformed)[:, 1], dtype="float32"
                )
            else:
                saved = joblib.load(cell / "tree_ensemble.joblib")
                runner = load_tree_runner()
                score = np.mean(
                    [
                        runner.predict_probability(
                            name, saved["models"][name], transformed
                        )
                        for name in saved["model_order"]
                    ],
                    axis=0,
                ).astype("float32")
            target = cell / "reloaded_screening_predictions.npz"
            with target.with_name(target.name + ".tmp").open("wb") as stream:
                np.savez_compressed(
                    stream,
                    raw_index=raw,
                    anomaly=y,
                    score=score,
                    query_raw_index_sha256=np.asarray(
                        query_manifest["raw_index_sha256"]
                    ),
                )
            target.with_name(target.name + ".tmp").replace(target)
            with np.load(cell / "screening_predictions.npz") as previous:
                reference = np.asarray(previous["score"], dtype="float32")
            rows.append(
                {
                    "model": model,
                    "state": str(cell.relative_to(recovery)),
                    "rows": len(score),
                    "score_mae_vs_fit_time": float(np.mean(np.abs(score - reference))),
                    "score_max_abs_vs_fit_time": float(
                        np.max(np.abs(score - reference))
                    ),
                    "spearman_vs_fit_time": float(
                        pd.Series(score).corr(pd.Series(reference), method="spearman")
                    ),
                    "finite": bool(np.isfinite(score).all()),
                }
            )
            print(f"reloaded {model} {cell.relative_to(recovery)}", flush=True)
    report = pd.DataFrame(rows)
    report.to_csv(recovery / "reports" / "state_reload_verification.csv", index=False)
    print(f"wrote {len(report)} state reload checks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
