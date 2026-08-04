from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from lead import BASELINE_FEATURE_COLS, BASELINE_FEATURE_COLS_WITH_BUILDING_ID


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_m3_50_50_ensemble as runner  # noqa: E402


class TestM3BuildingIdAblation(unittest.TestCase):
    def test_building_id_variant_preserves_original_baseline(self) -> None:
        self.assertEqual(
            BASELINE_FEATURE_COLS_WITH_BUILDING_ID,
            ["building_id", *BASELINE_FEATURE_COLS],
        )
        self.assertNotIn("building_id", BASELINE_FEATURE_COLS)

    def test_flag_defaults_to_false_and_can_be_enabled(self) -> None:
        with patch.object(sys, "argv", ["run_m3_50_50_ensemble.py"]):
            self.assertFalse(runner.parse_args().include_building_id)
        with patch.object(
            sys,
            "argv",
            ["run_m3_50_50_ensemble.py", "--include-building-id"],
        ):
            self.assertTrue(runner.parse_args().include_building_id)

    def test_selected_baseline_controls_offline_and_causal_feature_counts(self) -> None:
        value_cols = [
            f"lag_value_{kind}_{shift}"
            for shift in runner.SHIFTS
            for kind in ("diff", "ratio")
        ]
        columns = [
            "anomaly",
            *BASELINE_FEATURE_COLS_WITH_BUILDING_ID,
            *value_cols,
        ]
        train = pd.DataFrame(
            np.ones((2, len(columns)), dtype="float32"), columns=columns
        )
        val = train.copy()
        train["anomaly"] = [0, 1]
        val["anomaly"] = [0, 1]

        fake_run = {
            "ensemble": {
                "val_auc": 0.5,
                "precision_05": 0.5,
                "recall_05": 0.5,
                "f1_05": 0.5,
            }
        }
        with (
            patch.object(runner, "downsample_indices", return_value=np.array([0, 1])),
            patch.object(runner, "fit_predict_models", return_value=fake_run) as fit,
        ):
            offline = runner.run_regime(
                train,
                val,
                value_cols,
                "offline",
                BASELINE_FEATURE_COLS_WITH_BUILDING_ID,
            )
            causal = runner.run_regime(
                train,
                val,
                value_cols,
                "causal",
                BASELINE_FEATURE_COLS_WITH_BUILDING_ID,
            )

        self.assertEqual(offline["n_features"], 138)
        self.assertEqual(causal["n_features"], 78)
        self.assertEqual(fit.call_args_list[0].args[0].shape[1], 138)
        self.assertEqual(fit.call_args_list[1].args[0].shape[1], 78)

        with (
            patch.object(runner, "downsample_indices", return_value=np.array([0, 1])),
            patch.object(runner, "fit_predict_models", return_value=fake_run) as fit,
        ):
            offline = runner.run_regime(
                train,
                val,
                value_cols,
                "offline",
                BASELINE_FEATURE_COLS,
            )
            causal = runner.run_regime(
                train,
                val,
                value_cols,
                "causal",
                BASELINE_FEATURE_COLS,
            )

        self.assertEqual(offline["n_features"], 137)
        self.assertEqual(causal["n_features"], 77)
        self.assertEqual(fit.call_args_list[0].args[0].shape[1], 137)
        self.assertEqual(fit.call_args_list[1].args[0].shape[1], 77)


if __name__ == "__main__":
    unittest.main()
