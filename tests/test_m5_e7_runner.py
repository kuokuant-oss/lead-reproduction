from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import m5_e7_protocol as p  # noqa: E402
import m5_e7_runner as r  # noqa: E402


def test_unit_validation_rejects_missing_marker(tmp_path):
    with pytest.raises(ValueError, match="completion marker"):
        r.validate_unit(tmp_path / "none", {"query_rows": 2})


def test_quarantine_preserves_partial_material(tmp_path):
    unit = tmp_path / "unit"
    unit.mkdir()
    (unit / "partial").write_text("x")
    r.quarantine(unit, "test")
    dead = next(tmp_path.glob("unit.quarantine-*"))
    assert (dead / "partial").exists()
    assert json.loads((dead / "QUARANTINE.json").read_text())["reason"] == "test"


def test_support_cell_semantics_only_remove_declared_hotwater_strata():
    frame = __import__("pandas").DataFrame(
        {"meter": [3, 3, 2, 2, 0, 0], "anomaly": [1, 0, 1, 0, 1, 0]}, index=np.arange(6)
    )
    # n_pos stays non-zero for each cell and downsampling produces a historic 4-block vector.
    assert r.support_pool(frame, "00").size == 8
    assert r.support_pool(frame, "11").size == 12


def test_neutral_pool_exactly_matches_paired_class_counts():
    frame = __import__("pandas").DataFrame(
        {"meter": [2] * 12, "anomaly": [0] * 6 + [1] * 6}, index=np.arange(12)
    )
    support = r.support_pool(frame, "11")
    neutral = r.neutral_pool(frame, support, "n11")
    assert len(neutral) == len(support)
    assert int(frame.loc[neutral, "anomaly"].sum()) == int(
        frame.loc[support, "anomaly"].sum()
    )


def test_component_params_enforce_execution_correction_002_thread_policy():
    assert p.component_params("lightgbm")["num_threads"] == 8
    assert p.component_params("xgboost")["nthread"] == 8
    assert p.component_params("catboost")["thread_count"] == 8
    assert p.resource_environment()["OMP_NUM_THREADS"] == "8"


def test_cached_f4_selection_preserves_requested_order_and_duplicates():
    raw = np.array([20, 10, 30], dtype="int64")
    matrix = np.arange(3 * 137, dtype="float32").reshape(3, 137)
    requested = np.array([30, 20, 30], dtype="int64")
    selected = r.select_cached_f4_rows(raw, matrix, requested)
    assert selected.dtype == np.float32
    assert np.array_equal(selected, matrix[[2, 0, 2]])


def test_cached_f4_selection_rejects_unknown_raw_index():
    with pytest.raises(RuntimeError, match="absent"):
        r.select_cached_f4_rows(
            np.array([1], dtype="int64"),
            np.zeros((1, 137), dtype="float32"),
            np.array([2], dtype="int64"),
        )
