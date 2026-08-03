"""Freeze feasible no-replacement 100K balanced Steam contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPECS = {
    "steam_100k": (100_000, (2,)),
    "steam_hw_100k": (100_000, (2, 3)),
}


def array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(values, dtype="<i8")).tobytes()
    ).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"preflight output already exists: {args.out}")

    frame = pd.read_csv(
        args.m3_root / "train.csv",
        usecols=["building_id", "meter"],
        dtype={"building_id": "int16", "meter": "int8"},
    )
    labels = (
        pd.read_csv(
            args.m3_root / "bad_meter_readings.csv",
            usecols=["is_bad_meter_reading"],
            dtype={"is_bad_meter_reading": "int8"},
        )
        .iloc[:, 0]
        .to_numpy(dtype="int8")
    )
    raw_index = np.arange(len(frame), dtype="int64")
    even = frame.building_id.to_numpy() % 2 == 0
    validation = np.random.RandomState(20_042).choice(
        raw_index[even], size=4_000, replace=False
    )
    candidates = raw_index[even & ~np.isin(raw_index, validation)]
    meters = frame.meter.to_numpy(dtype="int8")
    items: dict[str, dict] = {}

    for name, (row_count, allowed_meters) in SPECS.items():
        pool = candidates[np.isin(meters[candidates], allowed_meters)]
        positives = pool[labels[pool] == 1].copy()
        negatives = pool[labels[pool] == 0].copy()
        needed = row_count // 2
        if len(positives) < needed or len(negatives) < needed:
            raise AssertionError(
                f"{name}: requires {needed} per class; "
                f"available positive={len(positives)}, negative={len(negatives)}"
            )
        rng = np.random.RandomState(42)
        rng.shuffle(positives)
        rng.shuffle(negatives)
        selected = np.empty(row_count, dtype="int64")
        selected[0::2] = positives[:needed]
        selected[1::2] = negatives[:needed]
        expected_labels = np.tile(np.array([1, 0], dtype="int8"), needed)
        if len(np.unique(selected)) != row_count or not np.array_equal(
            labels[selected], expected_labels
        ):
            raise AssertionError(f"{name}: balanced no-replacement gate failed")
        selected_meters = meters[selected]
        items[name] = {
            "raw_index": selected.tolist(),
            "raw_index_sha256": array_digest(selected),
            "rows": row_count,
            "anomaly": needed,
            "normal": needed,
            "meters": list(allowed_meters),
            "meter_counts": {
                str(meter): int((selected_meters == meter).sum())
                for meter in allowed_meters
            },
            "duplicate_raw_index_count": 0,
            "maximum_multiplicity": 1,
        }

    sources = {
        name: file_digest(args.m3_root / name)
        for name in (
            "train.csv",
            "bad_meter_readings.csv",
            "building_metadata.csv",
            "weather_train.csv",
        )
    }
    payload = {
        "schema": "m5_ek_steam_budget_preflight_v1",
        "source_sha256": sources,
        "validation_sha256": array_digest(np.sort(validation)),
        "items": items,
        "selection": (
            "E0/E1 seed-42 balanced class shuffle; fixed seed-20042 "
            "4,000-row validation exclusion; no replacement"
        ),
    }
    args.out.mkdir(parents=True)
    temporary = args.out / "preflight.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out / "preflight.json")
    print(
        json.dumps(
            {
                name: {
                    key: item[key]
                    for key in ("rows", "anomaly", "normal", "meter_counts")
                }
                for name, item in items.items()
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
