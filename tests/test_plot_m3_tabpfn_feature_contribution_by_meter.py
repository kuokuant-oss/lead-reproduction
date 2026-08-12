from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_module():
    path = ROOT / "scripts" / "plot_m3_tabpfn_feature_contribution_by_meter.py"
    spec = importlib.util.spec_from_file_location("m3_tabpfn_meter_curves", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestM3TabPFNFeatureContributionByMeter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_module()

    def test_loader_rejects_m3_meter_order_mismatch(self) -> None:
        tabpfn = {
            "anomaly": np.array([0, 1], dtype="int8"),
            "site_id": np.array([0, 1], dtype="int8"),
            "tabpfn": np.array([0.1, 0.9], dtype="float32"),
        }
        metadata = {
            "anomaly": np.array([0, 1], dtype="int8"),
            "site_id": np.array([0, 1], dtype="int8"),
            "meter": np.array([0, 1], dtype="int8"),
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = [root / f"{name}.npz" for name in ("t17", "t137", "m17", "m137")]
            np.savez_compressed(paths[0], **tabpfn)
            np.savez_compressed(paths[1], **tabpfn)
            np.savez_compressed(paths[2], **metadata)
            changed = dict(metadata)
            changed["meter"] = np.array([1, 0], dtype="int8")
            np.savez_compressed(paths[3], **changed)
            with self.assertRaisesRegex(ValueError, "meter"):
                self.m.load_aligned_tabpfn(*paths)

    def test_meter_metrics_and_grid_are_scoped_to_each_meter(self) -> None:
        arrays = {
            "anomaly": np.tile(np.array([0, 1, 0, 1], dtype="int8"), 4),
            "meter": np.repeat(np.arange(4, dtype="int8"), 4),
            "tabpfn_17_features": np.tile(np.array([0.1, 0.7, 0.2, 0.8]), 4),
            "tabpfn_137_features": np.tile(np.array([0.05, 0.95, 0.1, 0.9]), 4),
        }
        results = self.m.compute_meter_results(arrays)
        self.assertEqual(len(results), 4)
        self.assertEqual(results[0].rows, 4)
        self.assertEqual(results[0].engineered_roc.score, 1.0)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            roc = root / "roc.png"
            pr = root / "pr.png"
            self.m.render_grid(results, roc, curve_type="roc")
            self.m.render_grid(results, pr, curve_type="precision_recall")
            self.assertTrue(roc.is_file() and roc.stat().st_size > 0)
            self.assertTrue(pr.is_file() and pr.stat().st_size > 0)
