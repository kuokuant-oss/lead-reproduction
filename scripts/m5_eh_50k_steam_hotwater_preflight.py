"""Freeze the four Steam/Hotwater-only 50k Tree contexts without fitting."""

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
CONDITIONS = {
    "steam_only": "meter == steam",
    "steam_hw_all": "meter == steam or hotwater",
    "steam_hw_anomaly": "meter == steam or (hotwater and anomaly)",
    "steam_hw_normal": "meter == steam or (hotwater and normal)",
}


def digest(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(values, dtype="<i8")).tobytes()
    ).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_identity(m3: Path) -> pd.DataFrame:
    train = pd.read_csv(
        m3 / "train.csv",
        usecols=["building_id", "meter"],
        dtype={"building_id": "int16", "meter": "int8"},
    )
    labels = pd.read_csv(
        m3 / "bad_meter_readings.csv",
        usecols=["is_bad_meter_reading"],
        dtype={"is_bad_meter_reading": "int8"},
    )
    if len(train) != len(labels):
        raise AssertionError("M3 train/label positional length mismatch")
    frame = train.copy()
    frame.insert(0, "raw_index", np.arange(len(frame), dtype="int64"))
    frame["anomaly"] = labels.iloc[:, 0].to_numpy(dtype="int8")
    if not np.isin(frame["anomaly"], (0, 1)).all():
        raise AssertionError("non-binary anomaly label")
    return frame.set_index("raw_index", drop=False)


def candidate_mask(frame: pd.DataFrame, condition: str) -> np.ndarray:
    meter, label = frame["meter"].to_numpy(), frame["anomaly"].to_numpy()
    if condition == "steam_only":
        return meter == 2
    if condition == "steam_hw_all":
        return np.isin(meter, (2, 3))
    if condition == "steam_hw_anomaly":
        return (meter == 2) | ((meter == 3) & (label == 1))
    if condition == "steam_hw_normal":
        return (meter == 2) | ((meter == 3) & (label == 0))
    raise ValueError(f"unknown condition {condition}")


def balanced_context(candidates: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pos, neg = candidates[labels == 1].copy(), candidates[labels == 0].copy()
    rng = np.random.RandomState(MODEL_SEED)
    rng.shuffle(pos)
    rng.shuffle(neg)
    if len(pos) < ROWS // 2 or len(neg) < ROWS // 2:
        raise ValueError("candidate pool cannot supply 25k rows of each label")
    out = np.empty(ROWS, dtype="int64")
    out[0::2], out[1::2] = pos[: ROWS // 2], neg[: ROWS // 2]
    return out


def summary(frame: pd.DataFrame, raw: np.ndarray) -> dict[str, Any]:
    rows = frame.loc[raw]
    m, y = rows["meter"].to_numpy(), rows["anomaly"].to_numpy()
    return {
        "rows": int(len(raw)),
        "unique_rows": int(len(np.unique(raw))),
        "raw_index_sha256": digest(raw),
        "label_counts": {"normal": int((y == 0).sum()), "anomaly": int((y == 1).sum())},
        "meter_label_counts": {
            str(k): {
                "normal": int(((m == k) & (y == 0)).sum()),
                "anomaly": int(((m == k) & (y == 1)).sum()),
            }
            for k in range(4)
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m3-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    frame = load_identity(args.m3_root)
    even = frame["building_id"].to_numpy() % 2 == 0
    train = frame.index[even].to_numpy(dtype="int64")
    validation = (
        np.random.RandomState(MODEL_SEED + 20_000)
        .choice(train, VALIDATION_ROWS, replace=False)
        .astype("int64")
    )
    candidate = train[~np.isin(train, validation)]
    manifests: dict[str, Any] = {}
    for condition in CONDITIONS:
        eligible = candidate[candidate_mask(frame, condition)[candidate]]
        raw = balanced_context(
            eligible, frame.loc[eligible, "anomaly"].to_numpy(dtype="int8")
        )
        item = summary(frame, raw)
        if (
            item["rows"] != ROWS
            or item["unique_rows"] != ROWS
            or item["label_counts"] != {"normal": 25_000, "anomaly": 25_000}
        ):
            raise AssertionError(f"{condition}: cardinality gate failed")
        rows = frame.loc[raw]
        if (
            not np.isin(rows["meter"], (2, 3)).all()
            or not np.isin(raw, candidate).all()
        ):
            raise AssertionError(f"{condition}: meter or split isolation gate failed")
        if condition == "steam_only" and (rows["meter"] == 3).any():
            raise AssertionError("steam-only pool contains hotwater")
        if (
            condition == "steam_hw_anomaly"
            and ((rows["meter"] == 3) & (rows["anomaly"] == 0)).any()
        ):
            raise AssertionError("HW-anomaly pool contains hotwater normal")
        if (
            condition == "steam_hw_normal"
            and ((rows["meter"] == 3) & (rows["anomaly"] == 1)).any()
        ):
            raise AssertionError("HW-normal pool contains hotwater anomaly")
        manifests[condition] = {
            "raw_index": raw.tolist(),
            "summary": item,
            "candidate_counts": {
                "rows": int(len(eligible)),
                "anomaly": int(frame.loc[eligible, "anomaly"].sum()),
                "normal": int(len(eligible) - frame.loc[eligible, "anomaly"].sum()),
            },
        }
    source = {
        name: file_digest(args.m3_root / name)
        for name in (
            "train.csv",
            "bad_meter_readings.csv",
            "building_metadata.csv",
            "weather_train.csv",
        )
    }
    atomic_json(
        args.out / "preflight.json",
        {
            "schema": "m5_eh_50k_steam_hotwater_preflight_v1",
            "mode": "preflight_only_no_fit_no_predict",
            "context_rows": ROWS,
            "model_seed": MODEL_SEED,
            "validation_rows": VALIDATION_ROWS,
            "validation_raw_index_sha256": digest(np.sort(validation)),
            "selection": "condition-local E0/E1 nested_balanced_indices semantics; seed 42; no replacement",
            "conditions": CONDITIONS,
            "source_sha256": source,
            "manifests": manifests,
        },
    )
    print(
        json.dumps(
            {
                "preflight": str(args.out / "preflight.json"),
                "conditions": list(manifests),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
