from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lead.data import BASELINE_FEATURE_COLS
from lead.resource_guard import (
    LimitTracker,
    ResourceLimits,
    ResourceSample,
    atomic_write_json,
    terminate_process_tree,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_m5_tabpfn_single_context_scaling.py"
PLOT_SCRIPT = ROOT / "scripts" / "plot_m5_tabpfn_single_context_curves.py"


def load_script():
    spec = importlib.util.spec_from_file_location("m5_single_context", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample(*, gpu: float = 100, ram: float = 1_000) -> ResourceSample:
    return ResourceSample(
        timestamp=1,
        worker_rss_mib=100,
        system_used_mib=ram,
        system_available_mib=3_000,
        system_total_mib=4_000,
        gpu_used_mib=gpu,
        gpu_total_mib=1_000,
        monitoring_scope="process",
    )


class TestTabPFNSingleContextScaling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_script()

    def balanced(self):
        rows = np.arange(2_000, dtype="int64") * 3 + 7
        labels = np.tile(np.array([0, 1], dtype="int8"), 1_000)
        return self.m.nested_balanced_indices(rows, labels, [200, 500], seed=42)

    def test_nested_balanced_indices_are_deterministic(self) -> None:
        first, second = self.balanced(), self.balanced()
        np.testing.assert_array_equal(first[200], second[200])
        np.testing.assert_array_equal(first[500], second[500])

    def test_budgets_are_nested_prefixes(self) -> None:
        indices = self.balanced()
        np.testing.assert_array_equal(indices[200], indices[500][:200])

    def test_balanced_indices_are_unique_and_never_replace(self) -> None:
        self.assertEqual(len(np.unique(self.balanced()[500])), 500)
        with self.assertRaisesRegex(ValueError, "without replacement"):
            self.m.nested_balanced_indices(
                np.arange(20), np.arange(20) % 2, [40], seed=42
            )

    def test_building_split_has_zero_overlap(self) -> None:
        split = self.m.build_split(self.m.synthetic_frame(rows_per_building=4))
        self.assertEqual(
            split["metadata"]["building_overlaps"],
            {"fit_validation": 0, "fit_test": 0, "validation_test": 0},
        )

    def test_feature_count_is_exactly_17(self) -> None:
        self.assertEqual(len(BASELINE_FEATURE_COLS), 17)

    def test_atomic_state_write_replaces_complete_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            atomic_write_json(path, {"status": "first"})
            atomic_write_json(path, {"status": "second", "rows": 500_000})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"rows": 500_000, "status": "second"},
            )
            self.assertFalse(path.with_name("state.json.tmp").exists())

    def test_resume_skips_completed_budgets(self) -> None:
        args = SimpleNamespace(budgets=[100, 200, 500], restart_budget=None)
        self.assertEqual(
            self.m.budgets_to_run(args, {"completed_budgets": [100, 200]}),
            [500],
        )

    def test_budget_limit_pauses_after_requested_count(self) -> None:
        selected, paused = self.m.select_budgets_for_invocation([100, 200, 300], 1)
        self.assertEqual(selected, [100])
        self.assertTrue(paused)

    def test_site_curve_rows_cover_every_site_and_class(self) -> None:
        frame = self.m.synthetic_frame()
        test = frame.loc[frame["building_id"] % 2 == 1]
        first = self.m.stratified_site_curve_indices(test, rows_per_class=2, seed=123)
        second = self.m.stratified_site_curve_indices(test, rows_per_class=2, seed=123)
        np.testing.assert_array_equal(first, second)
        selected = test.loc[first]
        counts = selected.groupby(["site_id", "anomaly"]).size()
        self.assertEqual(len(first), 16 * 2 * 2)
        self.assertTrue((counts == 2).all())

    def test_previous_dead_worker_is_detected(self) -> None:
        state = {"running_budget": 200, "worker_pid": 999_999, "failed_budgets": []}
        summary = {"budget_results": {}}
        with patch.object(self.m, "pid_exists", return_value=False):
            self.m.mark_stale_worker(state, summary)
        self.assertEqual(
            summary["budget_results"]["200"]["status"],
            "interrupted_previous_run",
        )
        self.assertIsNone(state["running_budget"])

    def test_soft_limit_requires_consecutive_polls(self) -> None:
        tracker = LimitTracker(
            ResourceLimits(800, 900, 3_000, 3_500, soft_limit_consecutive_polls=2)
        )
        self.assertEqual(
            tracker.observe(sample(gpu=850), elapsed_seconds=1).action, "continue"
        )
        self.assertEqual(
            tracker.observe(sample(gpu=850), elapsed_seconds=2).action,
            "request_stop",
        )

    def test_hard_limit_terminates_immediately(self) -> None:
        tracker = LimitTracker(ResourceLimits(800, 900, 3_000, 3_500))
        decision = tracker.observe(sample(gpu=901), elapsed_seconds=1)
        self.assertEqual(decision.action, "terminate")
        self.assertEqual(decision.reason, "GPU hard limit exceeded")

    def test_disabled_timeout_never_stops_a_model(self) -> None:
        tracker = LimitTracker(ResourceLimits(800, 900, 3_000, 3_500))
        decision = tracker.observe(sample(gpu=100), elapsed_seconds=10**12)
        self.assertEqual(decision.action, "continue")

    def test_process_tree_termination_kills_survivor(self) -> None:
        class Process:
            def __init__(self, pid: int, children=()):
                self.pid, self._children = pid, list(children)
                self.terminated = self.killed = False

            def children(self, recursive: bool):
                self.recursive = recursive
                return self._children

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True

        child = Process(2)
        parent = Process(1, [child])
        calls = 0

        def wait(processes, timeout):
            nonlocal calls
            calls += 1
            return ([], [child]) if calls == 1 else (processes, [])

        result = terminate_process_tree(
            1,
            grace_seconds=0,
            process_factory=lambda _pid: parent,
            wait_procs=wait,
        )
        self.assertTrue(parent.recursive)
        self.assertTrue(parent.terminated and child.terminated and child.killed)
        self.assertEqual(result["killed"], [2])

    def test_prediction_batch_halving_stops_at_minimum(self) -> None:
        self.assertEqual(self.m.prediction_batch_sizes(256, 32), [256, 128, 64, 32])
        self.assertEqual(self.m.prediction_batch_sizes(100, 32), [100, 50, 32])

    def test_oom_is_serialized(self) -> None:
        with patch.object(self.m.traceback, "format_exc", return_value="trace"):
            result = self.m.serialize_failure(
                RuntimeError("CUDA out of memory"),
                budget=500_000,
                stage="test_predict",
                predict_batch_size=32,
            )
        self.assertEqual(result["status"], "oom")
        self.assertEqual(result["stage"], "test_predict")

    def test_headline_requires_complete_500k_contract(self) -> None:
        complete = {
            "budget_results": {
                "500000": {
                    "status": "completed",
                    "row_contract": {"count": 500_000, "unique_count": 500_000},
                    "context_contract": {
                        "requested_context_rows": 500_000,
                        "effective_context_rows": 500_000,
                        "external_sharding": False,
                        "sample_subsampling_disabled": True,
                        "effective_estimators": 1,
                    },
                    "fit_completed": True,
                    "validation_prediction_completed": True,
                    "test_prediction_completed": True,
                    "prediction_artifact": {"curve_inputs_complete": True},
                    "validation": {"roc_auc": 0.7, "pr_auc": 0.2},
                    "test": {"roc_auc": 0.6, "pr_auc": 0.1},
                }
            }
        }
        self.assertTrue(self.m.headline_500k_success(complete))
        complete["budget_results"]["500000"]["test_prediction_completed"] = False
        self.assertFalse(self.m.headline_500k_success(complete))

    def test_parent_has_no_top_level_torch_or_tabpfn_import(self) -> None:
        self.assertFalse(self.m.parent_has_forbidden_imports())

    def test_fake_model_smoke_completes_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out, state, events = (
                root / "result.json",
                root / "state.json",
                root / "events.jsonl",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--budgets",
                    "200",
                    "500",
                    "--score-rows",
                    "100",
                    "--predict-batch-size",
                    "32",
                    "--site-curve-rows-per-class",
                    "2",
                    "--site-curve-budget",
                    "500",
                    "--smoke",
                    "--out",
                    str(out),
                    "--state-out",
                    str(state),
                    "--events-out",
                    str(events),
                    "--poll-seconds",
                    "0.01",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["budget_results"]["200"]["status"], "completed")
            self.assertEqual(result["budget_results"]["500"]["status"], "completed")
            self.assertEqual(
                result["budget_results"]["500"]["context_contract"][
                    "effective_context_rows"
                ],
                500,
            )
            self.assertFalse(result["headline_500k_success"])
            artifact = Path(
                result["budget_results"]["500"]["prediction_artifact"]["path"]
            )
            self.assertTrue(artifact.is_file())
            with np.load(artifact) as arrays:
                required = {
                    "validation_y",
                    "validation_score",
                    "validation_site_id",
                    "test_y",
                    "test_score",
                    "test_site_id",
                    "site_curve_y",
                    "site_curve_score",
                    "site_curve_site_id",
                }
                self.assertEqual(required - set(arrays.files), set())
            plot_dir = root / "plots"
            plotted = subprocess.run(
                [
                    sys.executable,
                    str(PLOT_SCRIPT),
                    "--summary",
                    str(out),
                    "--output-dir",
                    str(plot_dir),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(plotted.returncode, 0, plotted.stderr)
            self.assertTrue((plot_dir / "m5_tabpfn_context_scaling_roc.png").is_file())
            self.assertTrue(
                (plot_dir / "m5_tabpfn_by_site_precision_recall.png").is_file()
            )

    def test_fake_controller_pauses_after_one_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "result.json"
            state = root / "state.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--budgets",
                    "200",
                    "500",
                    "--score-rows",
                    "100",
                    "--predict-batch-size",
                    "32",
                    "--max-budgets-this-run",
                    "1",
                    "--smoke",
                    "--out",
                    str(out),
                    "--state-out",
                    str(state),
                    "--events-out",
                    str(root / "events.jsonl"),
                    "--poll-seconds",
                    "0.01",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(out.read_text(encoding="utf-8"))
            saved_state = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(set(summary["budget_results"]), {"200"})
            self.assertEqual(saved_state["status"], "paused_after_budget_limit")
            self.assertEqual(saved_state["pending_budgets"], [500])


if __name__ == "__main__":
    unittest.main()
