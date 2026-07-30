"""Reconstruct meter x label composition of the frozen nested M5 contexts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CONTEXTS = (5_000, 10_000, 20_000, 50_000, 100_000)
FROZEN_100K_SHA256 = "e9ffe0cffd2e0cf304d213a02e68f2d7ef092172efc0343e680f982a2d688cbe"


def digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def main() -> int:
    path = ROOT / "data" / "raw" / "m3" / "train.csv"
    frame = pd.read_csv(
        path,
        usecols=["building_id", "meter"],
        dtype={"building_id": "int16", "meter": "int8"},
    )
    labels = pd.read_csv(
        ROOT / "data" / "raw" / "m3" / "bad_meter_readings.csv",
        usecols=["is_bad_meter_reading"],
        dtype={"is_bad_meter_reading": "int8"},
    )["is_bad_meter_reading"].to_numpy()
    if len(labels) != len(frame):
        raise AssertionError("positional label file length mismatch")
    train_index = np.flatnonzero(frame["building_id"].to_numpy() % 2 == 0)
    validation_index = (
        np.random.RandomState(42 + 20_000)
        .choice(train_index, 4_000, replace=False)
        .astype("int64")
    )
    candidate_index = train_index[~np.isin(train_index, validation_index)]
    candidate_y = labels[candidate_index]
    positive = candidate_index[candidate_y == 1].copy()
    negative = candidate_index[candidate_y == 0].copy()
    rng = np.random.RandomState(42)
    rng.shuffle(positive)
    rng.shuffle(negative)
    maximum = np.empty(100_000, dtype="int64")
    maximum[0::2] = positive[:50_000]
    maximum[1::2] = negative[:50_000]
    if digest(maximum) != FROZEN_100K_SHA256:
        raise AssertionError("reconstructed 100k context digest mismatch")

    rows: list[dict[str, object]] = []
    meters = frame["meter"].to_numpy()
    for context in CONTEXTS:
        index = maximum[:context]
        for label in (0, 1):
            for meter in range(4):
                count = int(((labels[index] == label) & (meters[index] == meter)).sum())
                rows.append(
                    {
                        "context_rows": context,
                        "anomaly": label,
                        "meter": meter,
                        "rows": count,
                        "within_label_share": count / (context / 2),
                    }
                )
    output = PROC / "m5_context_stories" / "reports"
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(
        output / "m5_context_curve_context_meter_label_counts.csv", index=False
    )
    (output / "m5_context_curve_context_meter_label_counts.json").write_text(
        json.dumps(
            {
                "frozen_100k_sha256": FROZEN_100K_SHA256,
                "contexts": list(CONTEXTS),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
