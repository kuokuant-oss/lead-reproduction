"""Identity and composition gates for the strict 50k Tree factorial preflight."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "m5_eg_50k_tree_factorial_preflight.py"
SPEC = importlib.util.spec_from_file_location("m5_eg_preflight", SCRIPT)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def fixture() -> pd.DataFrame:
    rows = 220_000
    raw = np.arange(rows, dtype="int64")
    return pd.DataFrame(
        {
            "raw_index": raw,
            "building_id": np.full(rows, 2, dtype="int16"),
            "meter": (raw % 4).astype("int8"),
            "anomaly": ((raw // 4) % 2 == 0).astype("int8"),
        }
    ).set_index("raw_index", drop=False)


def test_nested_context_is_deterministic_and_label_alternating() -> None:
    frame = fixture()
    candidate = frame["raw_index"].to_numpy(dtype="int64")
    full = MOD.nested_100k(candidate, frame["anomaly"].to_numpy(dtype="int8"))
    assert len(full) == 100_000
    assert len(np.unique(full)) == 100_000
    labels = frame.loc[full, "anomaly"].to_numpy(dtype="int8")
    assert np.array_equal(labels[0::2], np.ones(50_000, dtype="int8"))
    assert np.array_equal(labels[1::2], np.zeros(50_000, dtype="int8"))
    assert np.array_equal(
        full, MOD.nested_100k(candidate, frame["anomaly"].to_numpy(dtype="int8"))
    )


def test_factorial_preserves_original_cell11_and_replaces_only_expected_slots() -> None:
    frame = fixture()
    candidate = frame["raw_index"].to_numpy(dtype="int64")
    base = MOD.nested_100k(candidate, frame["anomaly"].to_numpy(dtype="int8"))[
        : MOD.ROWS
    ]
    cells = MOD.build_cells(
        frame, base=base, candidates=candidate, replacement_seed=20260803
    )
    assert cells["11"]["raw_index"] == base.tolist()
    for cell, (positive_present, negative_present) in MOD.CELLS.items():
        raw = np.asarray(cells[cell]["raw_index"], dtype="int64")
        rows = frame.loc[raw]
        assert len(raw) == 50_000
        assert len(np.unique(raw)) == 50_000
        assert rows["anomaly"].sum() == 25_000
        for label, present in ((1, positive_present), (0, negative_present)):
            hotwater = ((rows["meter"] == 3) & (rows["anomaly"] == label)).any()
            assert hotwater == present
        changed = raw != base
        for label, present in ((1, positive_present), (0, negative_present)):
            slots = (frame.loc[base, "meter"].to_numpy() == 3) & (
                frame.loc[base, "anomaly"].to_numpy() == label
            )
            assert np.array_equal(changed[slots], np.full(slots.sum(), not present))
        assert np.array_equal(raw[~changed], base[~changed])
