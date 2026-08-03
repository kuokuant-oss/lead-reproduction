"""Freeze the all-even, natural-prevalence Steam+Hotwater Tree training pool."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def sha(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(values, dtype="<i8")).tobytes()
    ).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m3-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    t = pd.read_csv(
        a.m3_root / "train.csv",
        usecols=["building_id", "meter"],
        dtype={"building_id": "int16", "meter": "int8"},
    )
    y = (
        pd.read_csv(
            a.m3_root / "bad_meter_readings.csv",
            usecols=["is_bad_meter_reading"],
            dtype={"is_bad_meter_reading": "int8"},
        )
        .iloc[:, 0]
        .to_numpy()
    )
    raw = np.arange(len(t), dtype="int64")
    pool = raw[
        (t["building_id"].to_numpy() % 2 == 0) & np.isin(t["meter"].to_numpy(), (2, 3))
    ]
    labels, meter = y[pool].astype("int8"), t["meter"].to_numpy()[pool]
    if len(pool) != len(np.unique(pool)) or not np.isin(meter, (2, 3)).all():
        raise AssertionError("unique Steam/Hotwater all-even pool gate failed")
    counts = {
        "rows": int(len(pool)),
        "unique_rows": int(len(np.unique(pool))),
        "anomaly": int(labels.sum()),
        "normal": int(len(pool) - labels.sum()),
        "prevalence": float(labels.mean()),
        "steam_rows": int((meter == 2).sum()),
        "hotwater_rows": int((meter == 3).sum()),
        "steam_anomaly": int(((meter == 2) & (labels == 1)).sum()),
        "steam_normal": int(((meter == 2) & (labels == 0)).sum()),
        "hotwater_anomaly": int(((meter == 3) & (labels == 1)).sum()),
        "hotwater_normal": int(((meter == 3) & (labels == 0)).sum()),
    }
    source = {
        n: file_sha(a.m3_root / n)
        for n in (
            "train.csv",
            "bad_meter_readings.csv",
            "building_metadata.csv",
            "weather_train.csv",
        )
    }
    write(
        a.out / "preflight.json",
        {
            "schema": "m5_ei_all_even_steam_hotwater_preflight_v1",
            "mode": "preflight_only_no_fit_no_predict",
            "training_rule": "all unique even-building Steam+Hotwater rows; natural anomaly prevalence; no sampling, weighting, or synthetic rows",
            "raw_index": pool.tolist(),
            "raw_index_sha256": sha(pool),
            "counts": counts,
            "source_sha256": source,
        },
    )
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
