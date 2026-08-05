from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import lead.features as feature_module
from lead import add_value_change_features, downsample_indices
from scripts import run_m5_building_curve_overnight as supervisor
from scripts.m5_tree_early_stopping import model_matrix
from scripts.run_m5_building_curve_tree_cell import (
    MATRIX_DTYPE,
    M3_SORT_KEYS,
    PREDICTION_DTYPE,
    _m3_downsampled_rows,
    _matrix_columns,
    _scale_matrix,
)


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "building_id": [2, 2, 2, 2, 4, 4, 4],
            "meter": [0, 1, 0, 1, 0, 0, 0],
            "timestamp": pd.to_datetime(
                [
                    "2016-01-01 00:00:00",
                    "2016-01-01 00:00:00",
                    "2016-01-01 01:00:00",
                    "2016-01-01 01:00:00",
                    "2016-01-01 00:00:00",
                    "2016-01-01 01:00:00",
                    "2016-01-01 03:00:00",
                ]
            ),
            "meter_reading": np.asarray(
                [10, 100, 12, 105, 20, 22, 30], dtype="float32"
            ),
        },
        index=[9, 4, 7, 2, 8, 3, 1],
    )


def _reference_features(
    frame: pd.DataFrame, shifts: list[int], regime: str
) -> pd.DataFrame:
    sort_keys = ["building_id", "timestamp"]
    if regime == "row_offset_meter_aware":
        sort_keys = ["building_id", "meter", "timestamp"]
    out = frame.sort_values(sort_keys).reset_index(drop=True).copy()
    meter_reading = out["meter_reading"]
    columns: dict[str, pd.Series] = {}
    for shift in shifts:
        if regime == "timestamp_merge":
            shifted = feature_module._timestamp_merge_shifted(out, shift)
        else:
            shifted = feature_module._row_offset_shifted(
                out,
                shift,
                meter_aware=regime == "row_offset_meter_aware",
            )
        columns[f"lag_value_diff_{shift}"] = (meter_reading - shifted).astype("float32")
        columns[f"lag_value_ratio_{shift}"] = (
            (meter_reading + 1) / (shifted + 1)
        ).astype("float32")
    return pd.concat([out, pd.DataFrame(columns)], axis=1)


class TestMemorySemantics(unittest.TestCase):
    def test_incremental_features_equal_previous_concat_implementation(self) -> None:
        frame = _feature_frame()
        for regime in (
            "timestamp_merge",
            "row_offset",
            "row_offset_meter_aware",
        ):
            with self.subTest(regime=regime):
                actual = add_value_change_features(
                    frame,
                    [-1, 1],
                    value_change_regime=regime,
                )
                expected = _reference_features(frame, [-1, 1], regime)
                pd.testing.assert_frame_equal(actual, expected)

    def test_in_place_standard_scaling_is_bitwise_equal(self) -> None:
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(128, 17)).astype("float32")
        expected_scaler = StandardScaler()
        expected = expected_scaler.fit_transform(raw.copy())
        actual = raw.copy()
        actual_scaler = StandardScaler()
        actual_scaler.fit(actual)
        returned = actual_scaler.transform(actual, copy=False)
        self.assertIs(returned, actual)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_scaler.mean_, expected_scaler.mean_)
        np.testing.assert_array_equal(actual_scaler.scale_, expected_scaler.scale_)

    def test_read_only_scaler_input_returns_writable_exact_matrix(self) -> None:
        raw = np.asarray([[1.0, np.nan], [2.0, 5.0], [4.0, 8.0]], dtype="float32")
        scaler = StandardScaler().fit(raw)
        expected = scaler.transform(raw.copy())
        read_only = raw.copy()
        read_only.flags.writeable = False

        returned = _scale_matrix(scaler, read_only)

        self.assertTrue(returned.flags.writeable)
        self.assertFalse(read_only.flags.writeable)
        np.testing.assert_array_equal(returned, expected)
        self.assertIs(model_matrix("hist_gradient_boosting", returned), returned)

    def test_hist_matrix_copies_read_only_input_with_same_values(self) -> None:
        values = np.asarray([[1.0, np.nan], [np.nan, -2.0]], dtype="float32")
        values.flags.writeable = False

        returned = model_matrix("hist_gradient_boosting", values)

        self.assertIsNot(returned, values)
        self.assertTrue(returned.flags.writeable)
        self.assertFalse(values.flags.writeable)
        np.testing.assert_array_equal(
            returned,
            np.asarray([[1.0, 0.0], [0.0, -2.0]], dtype="float32"),
        )

    def test_hist_matrix_reuses_storage_with_same_values(self) -> None:
        values = np.asarray([[1.0, np.nan], [np.nan, -2.0]], dtype="float32")
        returned = model_matrix("hist_gradient_boosting", values)
        self.assertIs(returned, values)
        np.testing.assert_array_equal(
            returned,
            np.asarray([[1.0, 0.0], [0.0, -2.0]], dtype="float32"),
        )

    def test_matrix_columns_do_not_require_materialized_features(self) -> None:
        columns = _matrix_columns(137, ["building_id", "meter_reading"])
        self.assertEqual(len(columns), 137)
        self.assertEqual(len(set(columns)), 137)

    def test_tree_training_restores_frozen_m3_downsampling(self) -> None:
        # Intentionally unsorted: M3 samples positions only after its feature
        # builder sorts by building_id/timestamp and resets the index.
        frame = pd.DataFrame(
            {
                "building_id": [4, 2, 4, 2, 2, 4, 2, 4] * 2,
                "timestamp": pd.to_datetime(
                    [f"2016-01-01 {hour:02d}:00:00" for hour in range(16)]
                ),
                "anomaly": np.asarray(
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                    dtype="int8",
                ),
            },
            index=np.arange(100, 116, dtype="int64"),
        )

        sampled = _m3_downsampled_rows(frame, frame.index.to_numpy())
        ordered = frame.loc[:, [*M3_SORT_KEYS, "anomaly"]].copy()
        ordered["raw_index"] = ordered.index.to_numpy(dtype="int64")
        ordered = ordered.sort_values(list(M3_SORT_KEYS)).reset_index(drop=True)
        expected_positions = downsample_indices(ordered["anomaly"])
        expected = ordered.loc[expected_positions, "raw_index"].to_numpy(dtype="int64")
        labels = frame.loc[sampled, "anomaly"].to_numpy()

        np.testing.assert_array_equal(sampled, expected)
        self.assertEqual(len(sampled), 16)
        self.assertEqual(int(np.count_nonzero(labels == 0)), 8)
        self.assertEqual(int(np.count_nonzero(labels == 1)), 8)
        positive_rows = frame.index[frame["anomaly"].eq(1)].to_numpy()
        for row in positive_rows:
            self.assertEqual(int(np.count_nonzero(sampled == row)), 2)

    def test_current_scaler_path_is_bitwise_equal_to_m3_dataframe_path(self) -> None:
        frame = pd.DataFrame(
            {
                "float32": np.asarray([1.25, 2.5, np.nan, -3.0], dtype="float32"),
                "float64": np.asarray([2.0, -5.5, 8.25, 3.0], dtype="float64"),
                "int8": np.asarray([1, 0, 1, 0], dtype="int8"),
            }
        )
        m3_scaler = StandardScaler()
        expected = m3_scaler.fit_transform(frame)

        values = frame.to_numpy(dtype=MATRIX_DTYPE)
        current_scaler = StandardScaler()
        current_scaler.fit(values)
        actual = _scale_matrix(current_scaler, values)

        self.assertEqual(actual.dtype, np.dtype("float64"))
        self.assertEqual(MATRIX_DTYPE, np.dtype("float64"))
        self.assertEqual(PREDICTION_DTYPE, np.dtype("float64"))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(current_scaler.mean_, m3_scaler.mean_)
        np.testing.assert_array_equal(current_scaler.scale_, m3_scaler.scale_)


class TestSupervisorReliability(unittest.TestCase):
    def test_attempt_count_survives_supervisor_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "status.json"
            state.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "stage": "tree_full_f137",
                        "attempt": 2,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                supervisor._prior_stage_attempts(state, "tree_full_f137"), 2
            )
            self.assertEqual(supervisor._prior_stage_attempts(state, "other"), 0)

    def test_failed_marker_and_status_are_atomic_equivalents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "status.json"
            failed = root / "FAILED.json"
            with mock.patch.object(supervisor, "FAILED_MARKER", failed):
                supervisor._mark_failed(
                    state,
                    stage="tree_full_f137",
                    attempts=3,
                    reason="stage_retry_limit_exhausted",
                )
            self.assertEqual(
                json.loads(state.read_text(encoding="utf-8")),
                json.loads(failed.read_text(encoding="utf-8")),
            )
            self.assertEqual(
                json.loads(failed.read_text(encoding="utf-8"))["status"], "failed"
            )

    def test_failed_stage_continues_queue_and_resume_skips_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stages = [
                {"name": "failed_stage", "publish": False, "command": ["fail"]},
                {"name": "next_stage", "publish": False, "command": ["next"]},
            ]
            args = SimpleNamespace(
                retry_delay=1,
                stage_retries=0,
                push_retries=1,
                git_push_timeout=1,
            )
            failed_command = subprocess.CompletedProcess(["fail"], 1)
            with (
                mock.patch.object(supervisor, "SUPERVISOR_ROOT", root),
                mock.patch.object(
                    supervisor, "STAGE_FAILED_ROOT", root / "failed_stages"
                ),
                mock.patch.object(supervisor, "FAILED_MARKER", root / "FAILED.json"),
                mock.patch.object(supervisor, "REPORT", root / "report.md"),
                mock.patch.object(supervisor, "parse_args", return_value=args),
                mock.patch.object(supervisor, "_clean_gate"),
                mock.patch.object(supervisor, "_ensure_protocols"),
                mock.patch.object(supervisor, "_stages", return_value=stages),
                mock.patch.object(
                    supervisor,
                    "_valid_complete",
                    side_effect=lambda stage: stage["name"] == "next_stage",
                ),
                mock.patch.object(
                    supervisor, "_command", return_value=failed_command
                ) as command,
                mock.patch.object(supervisor, "_event") as event,
            ):
                self.assertEqual(supervisor.main(), 0)
                marker = root / "failed_stages" / "failed_stage.json"
                failure = json.loads(marker.read_text(encoding="utf-8"))
                self.assertEqual(failure["status"], "stage_failed")
                self.assertTrue(failure["requires_review"])
                complete = json.loads(
                    (root / "COMPLETE.json").read_text(encoding="utf-8")
                )
                self.assertEqual(complete["status"], "completed_with_failures")
                self.assertEqual(complete["failed_stages"], ["failed_stage"])
                command.assert_called_once_with(["fail"], check=False)

                command.reset_mock()
                event.reset_mock()
                self.assertEqual(supervisor.main(), 0)
                command.assert_not_called()
                self.assertTrue(
                    any(
                        call.args == ("stage_skipped_failed_marker",)
                        for call in event.call_args_list
                    )
                )

    def test_command_timeout_is_a_retryable_return_code(self) -> None:
        timeout = subprocess.TimeoutExpired(["git", "push"], timeout=30)
        with (
            mock.patch.object(supervisor.subprocess, "run", side_effect=timeout),
            mock.patch.object(supervisor, "_event"),
        ):
            result = supervisor._command(
                ["git", "push"], check=False, timeout_seconds=30
            )
        self.assertEqual(result.returncode, 124)

    def test_supervisor_rejects_obsolete_tree_artifacts(self) -> None:
        names = [
            "lightgbm",
            "xgboost",
            "catboost",
            "hist_gradient_boosting",
            "ensemble",
        ]
        metadata = {
            "training_sampling": "M3 post-feature-sort:[negs1,pos,negs2,pos]",
            "training_sampling_seeds": [10, 20],
            "training_sampling_order": ["building_id", "timestamp"],
            "matrix_dtype": "float64",
            "prediction_dtype": "float64",
            "early_stopping_metric": "roc_auc",
            "score_names": names,
            "fit": {
                "model_contract": {
                    name: {
                        "selection_metric": "roc_auc",
                        "patience": 50 if name == "hist_gradient_boosting" else 200,
                    }
                    for name in names[:-1]
                }
            },
        }
        stored = {name: np.zeros(2, dtype="float64") for name in names}
        self.assertTrue(supervisor._valid_tree_contract(metadata, stored))

        obsolete_metric = json.loads(json.dumps(metadata))
        obsolete_metric["early_stopping_metric"] = "pr_auc"
        self.assertFalse(supervisor._valid_tree_contract(obsolete_metric, stored))

        obsolete_sampling = json.loads(json.dumps(metadata))
        obsolete_sampling["training_sampling"] = "raw-frame-order"
        self.assertFalse(supervisor._valid_tree_contract(obsolete_sampling, stored))

        quantized = dict(stored)
        quantized["ensemble"] = np.zeros(2, dtype="float32")
        self.assertFalse(supervisor._valid_tree_contract(metadata, quantized))


if __name__ == "__main__":
    unittest.main()
