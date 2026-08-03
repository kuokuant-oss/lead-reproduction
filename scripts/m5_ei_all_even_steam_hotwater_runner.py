"""Fit the frozen all-even Steam+Hotwater pool and score canonical odd Steam."""
# ruff: noqa: E701, E702, E402

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m5_eh_50k_steam_hotwater_runner import (  # noqa: E402
    ROOT,
    atomic_json,
    file_digest,
    fit_models,
    heartbeat,
    repo_commit,
    score,
)
from lead import load_m3_frame  # noqa: E402
from lead.m5_context import feature_names  # noqa: E402
from run_m3_figure_observations import frozen_model_contract  # noqa: E402
from run_m5_tree_ensemble_matched_context import (  # noqa: E402
    build_features_keeping_index,
    feature_columns,
)


NAME = "all_even_steam_hotwater_natural"


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    raw = np.asarray(value.get("raw_index", ()), dtype="int64")
    if (
        value.get("schema") != "m5_ei_all_even_steam_hotwater_preflight_v1"
        or len(raw) != value.get("counts", {}).get("rows")
        or len(raw) != len(np.unique(raw))
    ):
        raise ValueError("invalid EI preflight")
    return value


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--m3-root", type=Path, default=ROOT / "data" / "raw" / "m3")
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("dry-run", "formal"), default="dry-run")
    p.add_argument("--confirm", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--predict-batch-rows", type=int, default=100_000)
    a = p.parse_args()
    value = read(a.preflight)
    observed = {n: file_digest(a.m3_root / n) for n in value["source_sha256"]}
    if observed != value["source_sha256"]:
        raise ValueError("M3 source digest mismatch")
    with np.load(a.canonical, allow_pickle=False) as z:
        raw_test = z["raw_index"].astype("int64", copy=True)
    if a.mode == "dry-run":
        if (
            a.predict_batch_rows <= 0
            or a.predict_batch_rows > 100_000
            or len(raw_test) != 10_137_155
        ):
            raise ValueError("dry-run gate failed")
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "fit": 0,
                    "predict": 0,
                    "training_rows": value["counts"]["rows"],
                    "natural_prevalence": value["counts"]["prevalence"],
                    "expected_component_fits": 4,
                },
                sort_keys=True,
            )
        )
        return 0
    if a.confirm != "開始":
        raise SystemExit("formal mode requires --confirm 開始")
    if a.out.exists() and any(a.out.iterdir()) and not a.resume:
        raise SystemExit("formal root non-empty; use --resume")
    a.out.mkdir(parents=True, exist_ok=True)
    heartbeat(a.out, phase="initialising", completed_models=0, expected_models=4)
    frame = load_m3_frame(verbose=True)
    raw = np.asarray(value["raw_index"], dtype="int64")
    rows = frame.loc[raw]
    if (rows["building_id"].to_numpy() % 2).any() or not np.isin(
        rows["meter"], (2, 3)
    ).all():
        raise AssertionError("all-even Steam/Hotwater gate failed")
    train_mask = frame["building_id"].to_numpy() % 2 == 0
    if not np.array_equal(
        np.sort(raw_test), frame.index[~train_mask].to_numpy(dtype="int64")
    ):
        raise AssertionError("A002 odd identity gate failed")
    steam_test = raw_test[frame.loc[raw_test, "meter"].to_numpy() == 2]
    if len(steam_test) != 1_350_609:
        raise AssertionError("canonical Steam count gate failed")
    train_full = build_features_keeping_index(frame.loc[train_mask])
    columns = feature_columns(137, list(train_full.columns))
    if columns != feature_names("F4"):
        raise AssertionError("F4 order drift")
    base = {
        "preflight_sha256": file_digest(a.preflight),
        "source_sha256": value["source_sha256"],
        "feature_names_sha256": hashlib.sha256("\n".join(columns).encode()).hexdigest(),
        "model_contract": frozen_model_contract(42),
        "repository_commit": repo_commit(),
        "training_rule": value["training_rule"],
    }
    scaler, models, provenance = fit_models(
        NAME, raw, train_full, frame, columns, a.out, base, a.resume, expected_models=4
    )
    holdout = build_features_keeping_index(frame.loc[~train_mask])
    score(
        NAME,
        models,
        scaler,
        holdout,
        steam_test,
        columns,
        a.out,
        a.predict_batch_rows,
        provenance,
        expected_models=4,
    )
    atomic_json(
        a.out / "FORMAL_COMPLETE.json",
        {
            "expected_models": 4,
            "training_rows": len(raw),
            "score_rows": len(steam_test),
            "repository_commit": repo_commit(),
        },
    )
    return 0


if __name__ == "__main__":
    main()
