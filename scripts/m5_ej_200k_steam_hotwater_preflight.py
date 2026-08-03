"""Freeze the 200k balanced even Steam+Hotwater context without fitting."""

# ruff: noqa: E701, E702, E225, E231, E261, E265, E401, E501, F401
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd

ROWS, SEED, VALIDATION = 200_000, 42, 4_000


def sha(a):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a, dtype="<i8")).tobytes()
    ).hexdigest()


def fsha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
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
        .to_numpy(dtype="int8")
    )
    raw = np.arange(len(t), dtype="int64")
    even = t.building_id.to_numpy() % 2 == 0
    validation = np.random.RandomState(SEED + 20_000).choice(
        raw[even], VALIDATION, replace=False
    )
    candidate = raw[
        even & ~np.isin(raw, validation) & np.isin(t.meter.to_numpy(), (2, 3))
    ]
    pos, neg = candidate[y[candidate] == 1].copy(), candidate[y[candidate] == 0].copy()
    rng = np.random.RandomState(SEED)
    rng.shuffle(pos)
    rng.shuffle(neg)
    if len(pos) < ROWS // 2 or len(neg) < ROWS // 2:
        raise AssertionError("insufficient Steam/Hotwater class support for 200k")
    chosen = np.empty(ROWS, dtype="int64")
    chosen[0::2], chosen[1::2] = pos[: ROWS // 2], neg[: ROWS // 2]
    m = t.meter.to_numpy()[chosen]
    if (
        len(np.unique(chosen)) != ROWS
        or not np.isin(m, (2, 3)).all()
        or not np.array_equal(
            y[chosen], np.tile(np.array([1, 0], dtype="int8"), ROWS // 2)
        )
    ):
        raise AssertionError("200k context gate failed")
    source = {
        n: fsha(a.m3_root / n)
        for n in (
            "train.csv",
            "bad_meter_readings.csv",
            "building_metadata.csv",
            "weather_train.csv",
        )
    }
    value = {
        "schema": "m5_ej_200k_steam_hotwater_preflight_v1",
        "mode": "preflight_only_no_fit_no_predict",
        "training_rule": "200k unique even Steam+Hotwater rows; 100k anomaly+100k normal; E0/E1 nested shuffle seed 42; no replacement",
        "raw_index": chosen.tolist(),
        "raw_index_sha256": sha(chosen),
        "validation_raw_index_sha256": sha(np.sort(validation)),
        "counts": {
            "rows": ROWS,
            "unique_rows": ROWS,
            "anomaly": 100000,
            "normal": 100000,
            "steam_rows": int((m == 2).sum()),
            "hotwater_rows": int((m == 3).sum()),
        },
        "source_sha256": source,
    }
    a.out.mkdir(parents=True, exist_ok=False)
    tmp = a.out / ".preflight.tmp"
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, a.out / "preflight.json")
    print(json.dumps(value["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
