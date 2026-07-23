from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "supervise_m5_tabpfn_recovery.py"


def load_script():
    spec = importlib.util.spec_from_file_location("m5_tabpfn_recovery", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTabPFNRecoverySupervisor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_script()

    def setUp(self) -> None:
        self.log_directory = tempfile.TemporaryDirectory()
        self.original_log_path = self.m.LOG_PATH
        self.m.LOG_PATH = Path(self.log_directory.name) / "supervisor.log"

    def tearDown(self) -> None:
        self.m.LOG_PATH = self.original_log_path
        self.log_directory.cleanup()

    def test_retry_does_not_stop_after_five_failures(self) -> None:
        attempts = []
        sleeps = []

        def operation() -> bool:
            attempts.append(len(attempts) + 1)
            return len(attempts) >= 7

        with tempfile.TemporaryDirectory() as directory:
            original_log_path = self.m.LOG_PATH
            self.m.LOG_PATH = Path(directory) / "supervisor.log"
            try:
                self.m.retry_until_success(
                    operation,
                    sleep=sleeps.append,
                    delays=(1, 2, 5),
                )
            finally:
                self.m.LOG_PATH = original_log_path

        self.assertEqual(attempts, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(sleeps, [1, 2, 5, 5, 5, 5])

    def test_singleton_lock_rejects_a_duplicate_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "recovery.lock"
            with self.m.SingletonLock(lock_path):
                with self.assertRaises(self.m.SupervisorAlreadyRunning):
                    with self.m.SingletonLock(lock_path):
                        self.fail("duplicate supervisor acquired the same lock")

    def test_pid_alive_recognizes_a_different_live_process(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            self.assertTrue(self.m._pid_alive(process.pid))
        finally:
            process.terminate()
            process.wait()
        self.assertFalse(self.m._pid_alive(process.pid))

    def test_expected_colab_failures_remain_retryable(self) -> None:
        for message in (
            "Service Unavailable",
            "Failed to obtain valid ADC credentials",
            "TooManyAssignments",
            "Session 'lead-tabpfn-tail' not found",
        ):
            with self.subTest(message=message):
                self.assertEqual(self.m.classify_colab_failure(message), "retry")

    def test_colab_failure_with_missing_output_stream_does_not_crash(self) -> None:
        with (
            patch.object(self.m, "_sessions", return_value=("", [])),
            patch.object(
                self.m,
                "_colab",
                return_value=subprocess.CompletedProcess(
                    (), 1, "Service Unavailable", None
                ),
            ),
        ):
            self.assertFalse(self.m.ensure_fresh_colab_session())

    def test_subprocess_output_is_decoded_as_utf8(self) -> None:
        completed = subprocess.CompletedProcess((), 0, "", "")
        with patch.object(self.m.subprocess, "run", return_value=completed) as run:
            self.m._run(["example"])

        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(run.call_args.kwargs["errors"], "replace")

    def test_stale_named_colab_assignment_is_released_before_new_session(self) -> None:
        events = []

        def release(endpoint: str) -> bool:
            events.append(("release", endpoint))
            return True

        def colab(*arguments: str):
            events.append(("colab", *arguments))
            return subprocess.CompletedProcess(arguments, 0, "", "")

        sessions_output = (
            f"[{self.m.SESSION}] gpu-stale-runtime | T4",
            ["gpu-stale-runtime"],
        )
        with (
            patch.object(self.m, "_sessions", return_value=sessions_output),
            patch.object(self.m, "_release_exact_endpoint", side_effect=release),
            patch.object(self.m, "_colab", side_effect=colab),
        ):
            self.assertTrue(self.m.ensure_fresh_colab_session())

        self.assertEqual(
            events,
            [
                ("release", "gpu-stale-runtime"),
                ("colab", "new", "-s", self.m.SESSION, "--gpu", "T4"),
            ],
        )

    def test_colab_rebuild_reuploads_checkpoints_before_launch(self) -> None:
        events = []
        with (
            patch.object(
                self.m,
                "ensure_fresh_colab_session",
                side_effect=lambda: events.append("new") or True,
            ),
            patch.object(
                self.m,
                "restore_colab_files",
                side_effect=lambda: events.append("restore") or True,
            ),
            patch.object(
                self.m,
                "launch_and_verify_colab",
                side_effect=lambda: events.append("launch") or True,
            ),
        ):
            self.assertTrue(self.m.rebuild_colab_from_checkpoints())

        self.assertEqual(events, ["new", "restore", "launch"])

    def test_local_startup_wait_has_no_fixed_health_deadline(self) -> None:
        checks = []
        sleeps = []

        class LiveProcess:
            @staticmethod
            def poll():
                return None

        def healthy() -> bool:
            checks.append(len(checks) + 1)
            return len(checks) >= 7

        with tempfile.TemporaryDirectory() as directory:
            original_log_path = self.m.LOG_PATH
            self.m.LOG_PATH = Path(directory) / "supervisor.log"
            try:
                self.m.wait_for_local_worker_health(
                    LiveProcess(),
                    healthy=healthy,
                    sleep=sleeps.append,
                    delays=(1, 2, 5),
                )
            finally:
                self.m.LOG_PATH = original_log_path

        self.assertEqual(checks, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(sleeps, [1, 2, 5, 5, 5, 5])


if __name__ == "__main__":
    unittest.main()
