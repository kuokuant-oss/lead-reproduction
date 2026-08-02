"""Focused no-data tests for the M5 E7 frozen contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import m5_e7_protocol as p  # noqa: E402


def test_census_is_exactly_192_components():
    assert len(p.expected_units("oof")) == 160
    assert len(p.expected_units("final")) == 32
    assert p.EXPECTED_COMPONENTS == 192


def test_atomic_json_is_lf_and_digest_valid(tmp_path):
    target = tmp_path / "x.json"
    digest = p.atomic_json(target, {"a": [1, 2]})
    assert digest == p.sha256_file(target)
    assert b"\r\n" not in target.read_bytes()


def test_factorial_representation_has_declared_formula():
    scores = {
        f"s{k}": np.full(3, value, dtype="float32")
        for k, value in zip(p.SUPPORT_CELLS, (1, 3, 5, 11), strict=True)
    }
    result = p.factor_features(scores)
    assert np.allclose(result[0], [11, 5, 4, 6, 4])


def test_cpu_guard_is_fail_closed(monkeypatch):
    monkeypatch.setenv("NO_REMOTE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(SystemExit, match="CUDA"):
        p.require_local_cpu()


def test_unit_names_are_unique_and_descriptive():
    units = p.expected_units("oof") + p.expected_units("final")
    assert len(units) == len(set(units))
    assert all("__" in unit for unit in units)


def test_protocol_never_allows_remote_tabpfn_or_e6():
    frozen = p.protocol()["execution"]
    assert frozen["remote_commands_allowed"] is False
    assert frozen["tabpfn_allowed"] is False
    assert frozen["active_e6_allowed"] is False
    assert frozen["process_pool_allowed"] is False
