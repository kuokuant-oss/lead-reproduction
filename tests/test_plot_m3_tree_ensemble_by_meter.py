from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def load_module():
    path = ROOT / "scripts" / "plot_m3_tree_ensemble_by_meter.py"
    spec = importlib.util.spec_from_file_location("m3_meter_curves", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestM3TreeEnsembleByMeter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_module()

    def test_loader_rejects_meter_order_mismatch(self) -> None:
        identity = np.rec.fromarrays(
            [np.array([1, 2]), np.array([0, 0]), np.array([10, 20])],
            names="building_id,meter,timestamp_ns",
        )
        arrays = {
            "anomaly": np.array([0, 1], dtype="int8"),
            "meter": np.array([0, 0], dtype="int8"),
            "row_identity": identity,
            "ensemble": np.array([0.1, 0.9], dtype="float32"),
        }
        with tempfile.TemporaryDirectory() as raw:
            first = Path(raw) / "first.npz"
            second = Path(raw) / "second.npz"
            np.savez_compressed(first, **arrays)
            changed = dict(arrays)
            changed["meter"] = np.array([1, 0], dtype="int8")
            np.savez_compressed(second, **changed)
            with self.assertRaisesRegex(ValueError, "meter"):
                self.m.load_aligned_ensembles(first, second)

    def test_meter_curve_metrics_use_only_the_selected_meter(self) -> None:
        arrays = {
            "anomaly": np.array([0, 1, 0, 1, 0, 1], dtype="int8"),
            "meter": np.array([0, 0, 0, 0, 1, 1], dtype="int8"),
            "m3_1_ensemble": np.array([0.1, 0.8, 0.2, 0.7, 0.2, 0.9]),
            "ensemble": np.array([0.05, 0.95, 0.1, 0.9, 0.1, 0.95]),
        }
        _data, summary = self.m.meter_curve_data(arrays, 0)
        self.assertEqual(summary["rows"], 4)
        self.assertEqual(summary["anomalies"], 2)
        self.assertEqual(summary["metrics"]["ensemble"]["roc_auc"], 1.0)

    def test_render_all_writes_the_eight_meter_figures_via_m3_renderer(self) -> None:
        meter = np.repeat(np.arange(4, dtype="int8"), 4)
        arrays = {
            "anomaly": np.tile(np.array([0, 1, 0, 1], dtype="int8"), 4),
            "meter": meter,
            "m3_1_ensemble": np.tile(np.array([0.1, 0.7, 0.2, 0.8]), 4),
            "ensemble": np.tile(np.array([0.05, 0.95, 0.1, 0.9]), 4),
        }
        with tempfile.TemporaryDirectory() as raw:
            figures, summaries = self.m.render_all(arrays, Path(raw))
            self.assertEqual(len(figures), 8)
            self.assertEqual(
                set(summaries), {"electricity", "chilledwater", "steam", "hotwater"}
            )
            self.assertTrue(
                all(
                    path.is_file() and path.stat().st_size > 0
                    for path in figures.values()
                )
            )
