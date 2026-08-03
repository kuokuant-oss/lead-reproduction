"""Fit frozen balanced Steam contexts and score canonical odd Steam."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from m5_ei_all_even_steam_hotwater_runner import (  # noqa: E402
    ROOT,
    atomic_json,
    build_features_keeping_index,
    feature_columns,
    file_digest,
    fit_models,
    frozen_model_contract,
    heartbeat,
    load_m3_frame,
    repo_commit,
    score,
)


NAMES = ("steam_100k", "steam_hw_100k")
EXPECTED_COMPONENT_FITS = len(NAMES) * 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--m3-root", type=Path, default=ROOT / "data" / "raw" / "m3")
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("dry-run", "formal"), default="dry-run")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    items = preflight.get("items", {})
    if preflight.get("schema") != "m5_ek_steam_budget_preflight_v1" or set(
        items
    ) != set(NAMES):
        raise ValueError("invalid M5 EK preflight")
    observed = {
        name: file_digest(args.m3_root / name) for name in preflight["source_sha256"]
    }
    if observed != preflight["source_sha256"]:
        raise ValueError("M3 source digest mismatch")
    with np.load(args.canonical, allow_pickle=False) as artifact:
        canonical_raw_index = artifact["raw_index"].astype("int64", copy=True)

    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "fit": 0,
                    "predict": 0,
                    "expected_component_fits": EXPECTED_COMPONENT_FITS,
                    "rows": {name: items[name]["rows"] for name in NAMES},
                },
                sort_keys=True,
            )
        )
        return 0
    if args.confirm != "FORMAL_RUN":
        raise SystemExit("formal mode requires --confirm FORMAL_RUN")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit("formal output root is non-empty; use --resume")

    args.out.mkdir(parents=True, exist_ok=True)
    heartbeat(
        args.out,
        phase="initialising",
        completed_models=0,
        expected_models=EXPECTED_COMPONENT_FITS,
    )
    frame = load_m3_frame(verbose=True)
    even = frame.building_id.to_numpy() % 2 == 0
    if not np.array_equal(
        np.sort(canonical_raw_index), frame.index[~even].to_numpy(dtype="int64")
    ):
        raise AssertionError("A002 canonical holdout gate failed")
    steam_raw_index = canonical_raw_index[
        frame.loc[canonical_raw_index, "meter"].to_numpy() == 2
    ]
    full_even_features = build_features_keeping_index(frame.loc[even])
    columns = feature_columns(137, list(full_even_features.columns))
    odd_features = None
    for name in NAMES:
        raw_index = np.asarray(items[name]["raw_index"], dtype="int64")
        rows = frame.loc[raw_index]
        if (
            len(raw_index) != items[name]["rows"]
            or (rows.building_id.to_numpy() % 2).any()
            or not np.isin(rows.meter, items[name]["meters"]).all()
            or len(np.unique(raw_index)) != len(raw_index)
        ):
            raise AssertionError(f"{name}: frozen pool identity gate failed")
        base_provenance = {
            "preflight_sha256": file_digest(args.preflight),
            "source_sha256": preflight["source_sha256"],
            "model_contract": frozen_model_contract(42),
            "repository_commit": repo_commit(),
            "selection": preflight["selection"],
        }
        scaler, models, provenance = fit_models(
            name,
            raw_index,
            full_even_features,
            frame,
            columns,
            args.out,
            base_provenance,
            args.resume,
            expected_models=EXPECTED_COMPONENT_FITS,
        )
        if odd_features is None:
            odd_features = build_features_keeping_index(frame.loc[~even])
        score(
            name,
            models,
            scaler,
            odd_features,
            steam_raw_index,
            columns,
            args.out,
            100_000,
            provenance,
            expected_models=EXPECTED_COMPONENT_FITS,
        )
    atomic_json(
        args.out / "FORMAL_COMPLETE.json",
        {
            "expected_models": EXPECTED_COMPONENT_FITS,
            "names": NAMES,
            "score_rows": len(steam_raw_index),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
