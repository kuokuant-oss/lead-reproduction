"""Focused contract tests for the Steam/Hotwater TabPFN runner."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).parents[1] / "scripts" / "m5_steam_hotwater_tabpfn_runner.py"
    spec = importlib.util.spec_from_file_location(
        "m5_steam_hotwater_tabpfn_runner", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_loader_preserves_order_and_derives_prefixes(
    tmp_path, runner, monkeypatch
):
    vectors = {
        name: np.arange(i * 50_000, (i + 1) * 50_000, dtype="int64")
        for i, name in enumerate(runner.CONDITIONS)
    }
    digests = {name: runner.sha256_i64(raw) for name, raw in vectors.items()}
    monkeypatch.setattr(runner, "FROZEN_50K", digests)
    payload = {
        "schema": "m5_eh_50k_steam_hotwater_preflight_v1",
        "manifests": {
            name: {"raw_index": raw.tolist()} for name, raw in vectors.items()
        },
    }
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    got = runner.load_contexts(path)
    for name, raw in vectors.items():
        assert np.array_equal(got[f"50k_{name}"], raw)
        assert np.array_equal(got[f"20k_{name}"], raw[:20_000])


def test_context_gate_rejects_wrong_hotwater_membership(runner):
    raw = np.arange(20_000, dtype="int64")
    frame = pd.DataFrame(
        {
            "building_id": np.zeros(20_000, dtype="int16"),
            "meter": np.where(raw % 2 == 0, 2, 3).astype("int8"),
            "anomaly": (raw % 2 == 0).astype("int8"),
        },
        index=raw,
    )
    with pytest.raises(ValueError, match="Hotwater anomaly"):
        runner.verify_context(
            "20k_steam_hw_normal", raw, frame, np.array([], dtype="int64")
        )


def test_checkpoint_requires_matching_atomic_metadata(tmp_path, runner):
    path = tmp_path / "part.npy"
    provenance = {"label": "20k_steam_only", "start": 0, "stop": 2}
    runner.save_checkpoint(path, np.array([0.1, 0.2], dtype="float32"), provenance)
    assert runner.checkpoint_ok(path, (2,), provenance)
    path.with_suffix(".npy.json").write_text("{}", encoding="utf-8")
    assert not runner.checkpoint_ok(path, (2,), provenance)
