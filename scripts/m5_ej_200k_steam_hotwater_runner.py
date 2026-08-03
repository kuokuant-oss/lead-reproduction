"""Fit 200k balanced Steam+Hotwater Tree and score canonical odd Steam."""

# ruff: noqa: E701, E702, E225, E231, E261, E265, E401, E501
from __future__ import annotations
import argparse, json, numpy as np
from pathlib import Path
from m5_ei_all_even_steam_hotwater_runner import (
    ROOT,
    file_digest,
    fit_models,
    heartbeat,
    atomic_json,
    repo_commit,
    load_m3_frame,
    build_features_keeping_index,
    feature_columns,
    feature_names,
    frozen_model_contract,
    score,
)

NAME = "balanced_200k_steam_hotwater"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--m3-root", type=Path, default=ROOT / "data" / "raw" / "m3")
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("dry-run", "formal"), default="dry-run")
    p.add_argument("--confirm", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--predict-batch-rows", type=int, default=100000)
    a = p.parse_args()
    v = json.loads(a.preflight.read_text())
    raw = np.asarray(v.get("raw_index", ()), dtype="int64")
    if (
        v.get("schema") != "m5_ej_200k_steam_hotwater_preflight_v1"
        or len(raw) != 200000
        or len(np.unique(raw)) != 200000
    ):
        raise ValueError("invalid EJ preflight")
    if {n: file_digest(a.m3_root / n) for n in v["source_sha256"]} != v[
        "source_sha256"
    ]:
        raise ValueError("source digest mismatch")
    with np.load(a.canonical, allow_pickle=False) as z:
        test = z["raw_index"].astype("int64", copy=True)
    if a.mode == "dry-run":
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "fit": 0,
                    "predict": 0,
                    "training_rows": len(raw),
                    "expected_component_fits": 4,
                }
            )
        )
        return
    if a.confirm != "開始":
        raise SystemExit("formal requires --confirm 開始")
    if a.out.exists() and any(a.out.iterdir()) and not a.resume:
        raise SystemExit("formal root nonempty")
    a.out.mkdir(parents=True, exist_ok=True)
    heartbeat(a.out, phase="initialising", completed_models=0, expected_models=4)
    frame = load_m3_frame(verbose=True)
    rows = frame.loc[raw]
    if (
        (rows.building_id.to_numpy() % 2).any()
        or not np.isin(rows.meter, (2, 3)).all()
        or not np.array_equal(
            rows.anomaly.to_numpy(dtype="int8"),
            np.tile(np.array([1, 0], dtype="int8"), 100000),
        )
    ):
        raise AssertionError("training identity gate")
    train = frame.building_id.to_numpy() % 2 == 0
    if not np.array_equal(np.sort(test), frame.index[~train].to_numpy(dtype="int64")):
        raise AssertionError("A002 odd gate")
    steam = test[frame.loc[test, "meter"].to_numpy() == 2]
    train_full = build_features_keeping_index(frame.loc[train])
    cols = feature_columns(137, list(train_full.columns))
    if cols != feature_names("F4"):
        raise AssertionError("F4 gate")
    base = {
        "preflight_sha256": file_digest(a.preflight),
        "source_sha256": v["source_sha256"],
        "model_contract": frozen_model_contract(42),
        "repository_commit": repo_commit(),
        "training_rule": v["training_rule"],
    }
    scaler, models, prov = fit_models(
        NAME, raw, train_full, frame, cols, a.out, base, a.resume, expected_models=4
    )
    holdout = build_features_keeping_index(frame.loc[~train])
    score(
        NAME,
        models,
        scaler,
        holdout,
        steam,
        cols,
        a.out,
        a.predict_batch_rows,
        prov,
        expected_models=4,
    )
    atomic_json(
        a.out / "FORMAL_COMPLETE.json",
        {
            "expected_models": 4,
            "training_rows": len(raw),
            "score_rows": len(steam),
            "repository_commit": repo_commit(),
        },
    )


if __name__ == "__main__":
    main()
