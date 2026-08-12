from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_module():
    path = ROOT / "scripts" / "run_m3_feature_count_ensembles.py"
    spec = importlib.util.spec_from_file_location("m3_feature_count_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestM3FeatureCountEnsembles(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_module()

    def prepared(self) -> dict[str, object]:
        return {
            "labels": np.array([0, 1, 0], dtype="int8"),
            "meter": np.array([0, 1, 2], dtype="int8"),
            "site_id": np.array([1, 1, 2], dtype="int16"),
            "row_identity": np.rec.fromarrays(
                [
                    np.array([1, 2, 3], dtype="int32"),
                    np.array([0, 1, 2], dtype="int8"),
                    np.array([10, 20, 30], dtype="int64"),
                ],
                names="building_id,meter,timestamp_ns",
            ),
        }

    def test_model_contract_matches_frozen_m3_declarations(self) -> None:
        self.assertEqual(
            self.m.model_contract()["lightgbm"],
            {"n_estimators": 100, "verbose": -1, "random_state": 42},
        )
        self.assertEqual(self.m.model_contract()["catboost"]["iterations"], 1000)
        self.assertEqual(
            self.m.MODEL_ORDER,
            ("lightgbm", "xgboost", "catboost", "hist_gradient_boosting"),
        )

    def test_atomic_npz_round_trips_and_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "checkpoint.npz"
            expected = {
                "anomaly": np.array([0, 1], dtype="int8"),
                "lightgbm": np.array([0.2, 0.8], dtype="float32"),
            }
            self.m.atomic_npz(path, expected)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_name("checkpoint.npz.tmp").exists())
            with np.load(path) as actual:
                np.testing.assert_array_equal(actual["anomaly"], expected["anomaly"])
                np.testing.assert_array_equal(actual["lightgbm"], expected["lightgbm"])

    def test_resume_rejects_row_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "checkpoint.npz"
            prepared = self.prepared()
            arrays = self.m.prediction_arrays(
                prepared, {"lightgbm": np.array([0.1, 0.2, 0.3])}
            )
            arrays["meter"] = np.array([3, 1, 2], dtype="int8")
            self.m.atomic_npz(path, arrays)
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                self.m.load_prediction_checkpoint(path, prepared)

    def test_resume_reuses_a_valid_completed_model(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "checkpoint.npz"
            prepared = self.prepared()
            self.m.atomic_npz(
                path,
                self.m.prediction_arrays(
                    prepared, {"lightgbm": np.array([0.1, 0.2, 0.3])}
                ),
            )
            recovered = self.m.load_prediction_checkpoint(path, prepared)
            np.testing.assert_array_equal(
                recovered["lightgbm"], np.array([0.1, 0.2, 0.3], dtype="float32")
            )

    def test_corrupt_prediction_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "checkpoint.npz"
            path.write_bytes(b"not-an-npz")
            with self.assertRaises(Exception):
                self.m.load_prediction_checkpoint(path, self.prepared())

    def test_finalization_rejects_missing_model_unit(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.m.require_finalizable({"lightgbm": np.array([0.1])})

    def test_provenance_mismatch_refuses_resume(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "provenance differs"):
            self.m.assert_provenance({"input": "a"}, {"input": "b"})

    def test_validation_mode_requires_explicit_positive_caps(self) -> None:
        args = argparse.Namespace(
            mode="validation", validation_train_buildings=None, validation_buildings=1
        )
        with self.assertRaisesRegex(ValueError, "positive deterministic"):
            self.m.validate_mode_options(args)
        args = argparse.Namespace(
            mode="formal", validation_train_buildings=1, validation_buildings=None
        )
        with self.assertRaisesRegex(ValueError, "must not apply"):
            self.m.validate_mode_options(args)

    def test_status_is_atomic_and_reports_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "status.json"
            self.m.write_status(
                path,
                stage="completed_lightgbm",
                completed=1,
                total=4,
                started=0.0,
                unit_seconds=[12.0],
                mode="validation",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["completed_units"], 1)
            self.assertEqual(payload["estimated_remaining_seconds"], 36.0)
            self.assertIn("updated_at_utc", payload)
