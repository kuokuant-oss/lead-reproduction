"""Fit frozen balanced Steam contexts and score canonical odd Steam."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from m5_ei_all_even_steam_hotwater_runner import (  # noqa: E402
    ROOT,
    atomic_json,
    feature_columns,
    file_digest,
    fit_models,
    frozen_model_contract,
    heartbeat,
    load_m3_frame,
    repo_commit,
    score,
)
from lead import SHIFTS  # noqa: E402


NAMES = ("steam_100k", "steam_hw_100k")
EXPECTED_COMPONENT_FITS = len(NAMES) * 4


def build_selected_timestamp_features(
    source: pd.DataFrame, raw_index: np.ndarray
) -> pd.DataFrame:
    """Compute exact timestamp-merge F4 values only for requested rows.

    ``add_value_change_features`` materializes 120 columns for every source
    row.  For this 100K experiment that would create the same multi-gigabyte
    all-even temporary as an all-data fit.  Timestamp-merge values for a row
    depend only on its same-building/same-meter source reading at each frozen
    timestamp offset, so this indexed implementation has identical F4
    semantics while materializing the 137-column matrix only for the frozen
    context or the Steam holdout.
    """
    if not source.index.is_unique or len(raw_index) != len(np.unique(raw_index)):
        raise ValueError("source and requested raw_index must be unique")
    selected = source.loc[raw_index].copy()
    key_columns = ["building_id", "meter", "timestamp"]
    source_keys = pd.MultiIndex.from_frame(source[key_columns])
    if not source_keys.is_unique:
        raise ValueError("timestamp-merge source keys are not unique")
    readings = source["meter_reading"].to_numpy()
    selected_readings = selected["meter_reading"].to_numpy()
    value_columns: dict[str, np.ndarray] = {}
    for shift in SHIFTS:
        shifted_timestamps = selected["timestamp"] - pd.Timedelta(hours=shift)
        lookup = pd.MultiIndex.from_arrays(
            [
                selected["building_id"].to_numpy(),
                selected["meter"].to_numpy(),
                shifted_timestamps.to_numpy(),
            ],
            names=key_columns,
        )
        locations = source_keys.get_indexer(lookup)
        shifted = np.full(len(selected), np.nan, dtype="float64")
        present = locations >= 0
        shifted[present] = readings[locations[present]]
        value_columns[f"lag_value_diff_{shift}"] = (selected_readings - shifted).astype(
            "float32"
        )
        value_columns[f"lag_value_ratio_{shift}"] = (
            (selected_readings + 1) / (shifted + 1)
        ).astype("float32")
    return pd.concat(
        [selected, pd.DataFrame(value_columns, index=selected.index)], axis=1
    )


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
        source = frame.loc[even & frame.meter.isin(items[name]["meters"]).to_numpy()]
        selected_features = build_selected_timestamp_features(source, raw_index)
        columns = feature_columns(137, list(selected_features.columns))
        if not np.array_equal(
            selected_features.index.to_numpy(dtype="int64"), raw_index
        ):
            raise AssertionError(f"{name}: selected feature order drift")
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
            selected_features,
            frame,
            columns,
            args.out,
            base_provenance,
            args.resume,
            expected_models=EXPECTED_COMPONENT_FITS,
        )
        if odd_features is None:
            odd_steam = frame.loc[(~even) & (frame.meter.to_numpy() == 2)]
            odd_features = build_selected_timestamp_features(odd_steam, steam_raw_index)
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
