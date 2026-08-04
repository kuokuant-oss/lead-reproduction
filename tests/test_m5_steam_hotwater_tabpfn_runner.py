"""Focused contract tests for the Steam/Hotwater TabPFN runner."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def load_runner():
    path = Path(__file__).parents[1] / "scripts" / "m5_steam_hotwater_tabpfn_runner.py"
    spec = importlib.util.spec_from_file_location(
        "m5_steam_hotwater_tabpfn_runner", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SteamHotwaterTabPFNRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_frozen_loader_preserves_order_and_derives_prefixes(self):
        runner = self.runner
        vectors = {
            name: np.arange(i * 50_000, (i + 1) * 50_000, dtype="int64")
            for i, name in enumerate(runner.CONDITIONS)
        }
        original = runner.FROZEN_50K
        runner.FROZEN_50K = {
            name: runner.sha256_i64(raw) for name, raw in vectors.items()
        }
        try:
            payload = {
                "schema": "m5_eh_50k_steam_hotwater_preflight_v1",
                "manifests": {
                    name: {"raw_index": raw.tolist()} for name, raw in vectors.items()
                },
            }
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "preflight.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                got = runner.load_contexts(path)
            for name, raw in vectors.items():
                np.testing.assert_array_equal(got[f"50k_{name}"], raw)
                np.testing.assert_array_equal(got[f"20k_{name}"], raw[:20_000])
        finally:
            runner.FROZEN_50K = original

    def test_context_gate_rejects_wrong_hotwater_membership(self):
        runner = self.runner
        raw = np.arange(20_000, dtype="int64")
        frame = pd.DataFrame(
            {
                "building_id": np.zeros(20_000, dtype="int16"),
                "meter": np.where(raw % 2 == 0, 3, 2).astype("int8"),
                "anomaly": (raw % 2 == 0).astype("int8"),
            },
            index=raw,
        )
        with self.assertRaisesRegex(ValueError, "Hotwater anomaly"):
            runner.verify_context(
                "20k_steam_hw_normal", raw, frame, np.array([], dtype="int64")
            )

    def test_checkpoint_requires_matching_atomic_metadata(self):
        runner = self.runner
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part.npy"
            provenance = {"label": "20k_steam_only", "start": 0, "stop": 2}
            runner.save_checkpoint(
                path, np.array([0.1, 0.2], dtype="float32"), provenance
            )
            self.assertTrue(runner.checkpoint_ok(path, (2,), provenance))
            path.with_suffix(".npy.json").write_text("{}", encoding="utf-8")
            self.assertFalse(runner.checkpoint_ok(path, (2,), provenance))


if __name__ == "__main__":
    unittest.main()
