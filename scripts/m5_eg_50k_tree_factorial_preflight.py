"""Preflight the E0/E1-exact 50k Hotwater Tree factorial on this laptop.

This command is deliberately data-only: it does not import a learner, build
F4 features, fit, or predict.  It reconstructs E0/E1's ordered 100k context,
takes its exact 50k prefix, and materialises four validated row manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROWS = 50_000
MODEL_SEED = 42
VALIDATION_ROWS = 4_000
E0_100K_DIGEST = "e9ffe0cffd2e0cf304d213a02e68f2d7ef092172efc0343e680f982a2d688cbe"
CELLS = {
    "11": (True, True),
    "01": (False, True),
    "10": (True, False),
    "00": (False, False),
}


def digest(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(values, dtype="<i8")).tobytes()
    ).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def stable_row_priority(raw_index: np.ndarray, *, seed: int) -> np.ndarray:
    """Match the repository's deterministic no-replacement reserve ordering."""
    values = np.asarray(raw_index, dtype="uint64")
    z = values ^ (np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15))
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_m3_identity_frame(m3_root: Path) -> pd.DataFrame:
    """Load only the source fields needed for an exact row-identity preflight."""
    train = pd.read_csv(
        m3_root / "train.csv",
        usecols=["building_id", "meter"],
        dtype={"building_id": "int16", "meter": "int8"},
    )
    labels = pd.read_csv(
        m3_root / "bad_meter_readings.csv",
        usecols=["is_bad_meter_reading"],
        dtype={"is_bad_meter_reading": "int8"},
    )
    if len(train) != len(labels):
        raise AssertionError("M3 train/label positional lengths differ")
    values = labels["is_bad_meter_reading"].to_numpy(dtype="int8")
    if not np.isin(values, (0, 1)).all():
        raise AssertionError("M3 anomaly labels are not binary")
    train.insert(0, "raw_index", np.arange(len(train), dtype="int64"))
    train["anomaly"] = values
    if train["raw_index"].duplicated().any():
        raise AssertionError("M3 positional raw_index is not unique")
    return train


def normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"raw_index", "building_id", "meter", "anomaly"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing {sorted(missing)}")
    clean = frame.loc[:, ["raw_index", "building_id", "meter", "anomaly"]].copy()
    clean["raw_index"] = clean["raw_index"].astype("int64")
    clean["building_id"] = clean["building_id"].astype("int16")
    clean["meter"] = clean["meter"].astype("int8")
    clean["anomaly"] = clean["anomaly"].astype("int8")
    if (
        clean["raw_index"].duplicated().any()
        or not np.isin(clean["anomaly"], (0, 1)).all()
    ):
        raise ValueError("raw_index must be unique and anomaly must be binary")
    return clean.set_index("raw_index", drop=False).sort_index()


def nested_100k(candidate: np.ndarray, labels: np.ndarray) -> np.ndarray:
    positive, negative = candidate[labels == 1].copy(), candidate[labels == 0].copy()
    rng = np.random.RandomState(MODEL_SEED)
    rng.shuffle(positive)
    rng.shuffle(negative)
    if len(positive) < 50_000 or len(negative) < 50_000:
        raise ValueError("insufficient class support for the frozen 100k ordering")
    result = np.empty(100_000, dtype="int64")
    result[0::2], result[1::2] = positive[:50_000], negative[:50_000]
    return result


def largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    if total == 0:
        return np.zeros(len(weights), dtype="int64")
    desired = np.asarray(weights, dtype="float64") / np.sum(weights) * total
    result = np.floor(desired).astype("int64")
    result[
        np.argsort(-(desired - result), kind="stable")[: total - int(result.sum())]
    ] += 1
    return result


def replacement_indices(
    indexed: pd.DataFrame,
    *,
    base: np.ndarray,
    candidates: np.ndarray,
    label: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    base_rows = indexed.loc[base]
    slots = np.flatnonzero(
        (base_rows["meter"].to_numpy() == 3)
        & (base_rows["anomaly"].to_numpy() == label)
    )
    source = indexed.loc[candidates]
    source = source.loc[
        (source["anomaly"] == label) & (source["meter"] != 3) & ~source.index.isin(base)
    ]
    pooled_non_hotwater = base_rows.loc[
        (base_rows["anomaly"] == label) & (base_rows["meter"] != 3), "meter"
    ]
    meters = np.asarray(
        sorted(int(value) for value in pooled_non_hotwater.unique()), dtype="int8"
    )
    allocation = largest_remainder(
        np.asarray(
            [(pooled_non_hotwater == meter).sum() for meter in meters], dtype="int64"
        ),
        len(slots),
    )
    choices: list[np.ndarray] = []
    choice_meters: list[np.ndarray] = []
    for meter, count in zip(meters, allocation, strict=True):
        eligible = source.loc[source["meter"] == meter, "raw_index"].to_numpy(
            dtype="int64"
        )
        if len(eligible) < count:
            raise ValueError(
                f"insufficient label={label}, meter={meter} non-Hotwater replacements"
            )
        order = np.lexsort(
            (
                eligible,
                stable_row_priority(eligible, seed=seed + 10_000 * label + int(meter)),
            )
        )
        choices.append(eligible[order[:count]])
        choice_meters.append(np.full(count, meter, dtype="int8"))
    values = np.concatenate(choices) if choices else np.empty(0, dtype="int64")
    meters_out = (
        np.concatenate(choice_meters) if choice_meters else np.empty(0, dtype="int8")
    )
    order = np.lexsort(
        (values, stable_row_priority(values, seed=seed + 100_000 + label))
    )
    values, meters_out = values[order], meters_out[order]
    records = [
        {
            "slot": int(slot),
            "label": label,
            "old_raw_index": int(base[slot]),
            "new_raw_index": int(new),
            "replacement_meter": int(meter),
        }
        for slot, new, meter in zip(slots, values, meters_out, strict=True)
    ]
    return slots, values, records


def summarize(indexed: pd.DataFrame, raw: np.ndarray) -> dict[str, Any]:
    rows = indexed.loc[raw]
    meter, label = rows["meter"].to_numpy(), rows["anomaly"].to_numpy()
    return {
        "rows": int(len(raw)),
        "unique_rows": int(len(np.unique(raw))),
        "raw_index_sha256": digest(raw),
        "label_counts": {
            "normal": int((label == 0).sum()),
            "anomaly": int((label == 1).sum()),
        },
        "meter_label_counts": {
            str(m): {
                "normal": int(((meter == m) & (label == 0)).sum()),
                "anomaly": int(((meter == m) & (label == 1)).sum()),
            }
            for m in range(4)
        },
    }


def build_cells(
    indexed: pd.DataFrame,
    *,
    base: np.ndarray,
    candidates: np.ndarray,
    replacement_seed: int,
) -> dict[str, dict[str, Any]]:
    maps = {
        label: replacement_indices(
            indexed,
            base=base,
            candidates=candidates,
            label=label,
            seed=replacement_seed,
        )
        for label in (0, 1)
    }
    output: dict[str, dict[str, Any]] = {}
    for cell, (positive_present, negative_present) in CELLS.items():
        raw, changes = base.copy(), []
        for label, present in ((1, positive_present), (0, negative_present)):
            slots, values, records = maps[label]
            if not present:
                raw[slots] = values
                changes.extend(records)
            elif not np.array_equal(raw[slots], base[slots]):
                raise AssertionError(f"{cell}: present Hotwater slots changed")
        detail = summarize(indexed, raw)
        rows = indexed.loc[raw]
        if (
            detail["rows"] != ROWS
            or detail["unique_rows"] != ROWS
            or detail["label_counts"] != {"normal": 25_000, "anomaly": 25_000}
        ):
            raise AssertionError(f"{cell}: row/uniqueness/label gate failed")
        for label, present in ((1, positive_present), (0, negative_present)):
            has_hotwater = ((rows["meter"] == 3) & (rows["anomaly"] == label)).any()
            if has_hotwater != present:
                raise AssertionError(
                    f"{cell}: Hotwater label={label} intervention gate failed"
                )
        if not np.isin(raw, candidates).all():
            raise AssertionError(f"{cell}: contains validation or holdout row")
        output[cell] = {
            "raw_index": raw.tolist(),
            "summary": detail,
            "replacements": changes,
        }
    if output["11"]["raw_index"] != base.tolist():
        raise AssertionError("cell11 is not the original E0/E1 50k context")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--frame",
        type=Path,
        help="parquet/csv with raw_index, building_id, meter, anomaly",
    )
    source.add_argument(
        "--m3-root",
        type=Path,
        help="M3 raw directory; loads train.csv and bad_meter_readings.csv only",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replacement-seed", type=int, required=True)
    parser.add_argument("--expected-100k-digest", default=E0_100K_DIGEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = (
        load_m3_identity_frame(args.m3_root)
        if args.m3_root
        else (
            pd.read_parquet(args.frame)
            if args.frame.suffix == ".parquet"
            else pd.read_csv(args.frame)
        )
    )
    indexed = normalise_frame(source)
    train = indexed.loc[indexed["building_id"] % 2 == 0, "raw_index"].to_numpy(
        dtype="int64"
    )
    validation = (
        np.random.RandomState(MODEL_SEED + 20_000)
        .choice(train, VALIDATION_ROWS, replace=False)
        .astype("int64")
    )
    candidates = train[~np.isin(train, validation)]
    full = nested_100k(
        candidates, indexed.loc[candidates, "anomaly"].to_numpy(dtype="int8")
    )
    if digest(full) != args.expected_100k_digest:
        raise AssertionError("E0/E1 100k ordering digest mismatch")
    base = full[:ROWS]
    cells = build_cells(
        indexed,
        base=base,
        candidates=candidates,
        replacement_seed=args.replacement_seed,
    )
    source_digests = (
        {
            name: file_digest(args.m3_root / name)
            for name in (
                "train.csv",
                "bad_meter_readings.csv",
                "building_metadata.csv",
                "weather_train.csv",
            )
        }
        if args.m3_root
        else {args.frame.name: file_digest(args.frame)}
    )
    payload = {
        "schema": "m5_eg_50k_tree_factorial_preflight_v2",
        "mode": "preflight_only_no_fit_no_predict",
        "source": str(args.m3_root or args.frame),
        "source_sha256": source_digests,
        "context_rows": ROWS,
        "model_seed": MODEL_SEED,
        "validation_rows": VALIDATION_ROWS,
        "replacement_seed": args.replacement_seed,
        "e0_100k_digest": digest(full),
        "cell11_is_exact_e0_prefix": True,
        "validation_raw_index_sha256": digest(np.sort(validation)),
        "cells": cells,
    }
    atomic_json(args.out / "preflight.json", payload)
    print(
        json.dumps(
            {
                "preflight": str(args.out / "preflight.json"),
                "cells": list(cells),
                "e0_100k_digest": digest(full),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
