from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "supervise_m5_tabpfn_recovery.py"
LAUNCHER = ROOT / "scripts" / "launch_m5_colab_recovery_supervisor.ps1"
ENSURE_LAUNCHER = ROOT / "scripts" / "ensure_m5_colab_recovery_supervisor.ps1"
COLAB_WORKER_LAUNCHER = ROOT / "scripts" / "launch_m5_tabpfn_colab_tail.py"


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

    def test_retry_keeps_supervisor_alive_after_transient_operation_exception(
        self,
    ) -> None:
        attempts = []
        sleeps = []

        def operation() -> bool:
            attempts.append(len(attempts) + 1)
            if len(attempts) == 1:
                raise RuntimeError("temporary upload transport error")
            return True

        with (
            patch.object(self.m, "write_supervisor_status") as status,
            patch.object(self.m, "log"),
        ):
            self.m.retry_until_success(operation, sleep=sleeps.append, delays=(1,))

        self.assertEqual(attempts, [1, 2])
        self.assertEqual(sleeps, [1])
        self.assertTrue(
            any(
                call.args
                and call.args[0] == "operation_exception"
                and call.kwargs.get("attempt") == 1
                and "temporary upload transport error"
                in call.kwargs.get("last_error", "")
                for call in status.mock_calls
            )
        )

    def test_colab_allocation_uses_jittered_exponential_backoff(self) -> None:
        delays = [
            self.m.colab_allocation_delay(
                attempt,
                uniform=lambda lower, upper: (lower + upper) / 2,
            )
            for attempt in range(1, 8)
        ]
        self.assertEqual(delays[:5], [60, 120, 240, 480, 960])
        self.assertEqual(delays[5:], [1350, 1350])

    def test_host_launcher_is_colab_only(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("--scope colab", launcher)
        self.assertNotIn("--scope both", launcher)
        self.assertNotIn("--scope local", launcher)
        self.assertIn('TABPFN_COLAB_HOME = "/home/tonykuo/.colab-hank"', launcher)
        self.assertIn('TABPFN_COLAB_AUTH = "oauth2"', launcher)
        self.assertIn('TABPFN_COLAB_ACCELERATOR = "L4"', launcher)

    def test_colab_worker_launcher_uses_verified_microbatch(self) -> None:
        launcher = COLAB_WORKER_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('"--query-microbatch-size",\n    "1024"', launcher)
        self.assertIn('"--min-query-microbatch-size",\n    "64"', launcher)
        self.assertIn('"--resume"', launcher)

    def test_synced_heartbeat_can_prove_recent_health(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original_results = self.m.TAIL_RESULTS
            self.m.TAIL_RESULTS = Path(directory)
            try:
                (self.m.TAIL_RESULTS / "heartbeat.json").write_text(
                    json.dumps({"timestamp": 900.0}),
                    encoding="utf-8",
                )
                self.assertTrue(
                    self.m.synced_heartbeat_fresh(
                        now=lambda: 1000.0,
                        max_age_seconds=101,
                    )
                )
                self.assertFalse(
                    self.m.synced_heartbeat_fresh(
                        now=lambda: 1000.0,
                        max_age_seconds=99,
                    )
                )
            finally:
                self.m.TAIL_RESULTS = original_results

    def test_completed_shard_is_checked_before_remote_health_probe(self) -> None:
        events = []
        with (
            patch.object(self.m, "colab_formal_shard_complete", return_value=True),
            patch.object(
                self.m,
                "_inspect_colab",
                side_effect=AssertionError("completed shard must not probe remote"),
            ),
            patch.object(self.m, "log", side_effect=events.append),
        ):
            self.m.recover_colab_until_healthy()
        self.assertTrue(
            any("colab_formal_shard_complete=true" in value for value in events)
        )

    def test_ensure_launcher_uses_persistent_singleton_task(self) -> None:
        launcher = ENSURE_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("CodexTabPFNColabRecoverySupervisor", launcher)
        self.assertIn("m5_tabpfn_recovery_supervisor.lock", launcher)
        self.assertIn("ExecutionTimeLimit ([TimeSpan]::Zero)", launcher)
        self.assertIn("Start-ScheduledTask", launcher)

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

    def test_transport_health_rejects_zero_exit_missing_session(self) -> None:
        missing = "[colab] Session 'lead-tabpfn-tail' not found."
        with patch.object(self.m, "_sessions", return_value=(missing, [])):
            self.assertFalse(self.m.colab_session_transport_healthy())

    def test_transport_health_requires_named_server_assignment(self) -> None:
        listing = f"[{self.m.SESSION}] gpu-live-runtime | Hardware: T4 | Variant: GPU"
        with patch.object(
            self.m,
            "_sessions",
            return_value=(listing, ["gpu-live-runtime"]),
        ):
            self.assertTrue(self.m.colab_session_transport_healthy())

    def test_colab_failure_with_missing_output_stream_does_not_crash(self) -> None:
        with (
            patch.object(self.m, "_sessions", return_value=("", [])),
            patch.object(
                self.m,
                "_run",
                return_value=subprocess.CompletedProcess(
                    (), 1, "Service Unavailable", None
                ),
            ),
        ):
            self.assertFalse(self.m.ensure_fresh_colab_session())

    def test_remote_exec_failure_records_return_code_and_sanitized_detail(self) -> None:
        events = []
        result = subprocess.CompletedProcess(
            (), 503, "", "Service Unavailable authorization=secret-value"
        )
        with (
            patch.object(self.m, "_colab", return_value=result),
            patch.object(self.m, "log", side_effect=events.append),
            patch.object(self.m, "write_supervisor_status") as status,
        ):
            self.assertFalse(self.m._remote_exec(self.m.ROOT / ".scratch" / "probe.py"))

        self.assertTrue(any("remote_exec_failed=true" in value for value in events))
        message = next(value for value in events if "remote_exec_failed=true" in value)
        self.assertIn("returncode=503", message)
        self.assertNotIn("secret-value", message)
        self.assertTrue(
            any(
                call.args
                and call.args[0] == "remote_exec_failed"
                and call.kwargs.get("returncode") == 503
                for call in status.mock_calls
            )
        )

    def test_dependency_inspection_failure_is_non_blocking(self) -> None:
        events = []
        with (
            patch.object(self.m, "_remote_exec", side_effect=[True, True, False]),
            patch.object(self.m, "log", side_effect=events.append),
            patch.object(self.m, "write_supervisor_status"),
        ):
            self.assertTrue(self.m.run_runtime_setup())

        self.assertTrue(
            any("runtime_inspect_warning=true" in value for value in events)
        )
        self.assertFalse(any("runtime_setup_failed=true" in value for value in events))

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

        def run(arguments):
            events.append(("run", *arguments))
            return subprocess.CompletedProcess(arguments, 0, "", "")

        sessions_output = (
            f"[{self.m.SESSION}] gpu-stale-runtime | T4",
            ["gpu-stale-runtime"],
        )
        with (
            patch.object(self.m, "_sessions", return_value=sessions_output),
            patch.object(self.m, "_release_exact_endpoint", side_effect=release),
            patch.object(self.m, "_run", side_effect=run),
        ):
            self.assertTrue(self.m.ensure_fresh_colab_session())

        self.assertEqual(
            events,
            [
                ("release", "gpu-stale-runtime"),
                (
                    "run",
                    "wsl.exe",
                    "-d",
                    "Ubuntu",
                    "--",
                    "env",
                    f"HOME={self.m.COLAB_HOME}",
                    self.m.COLAB_PYTHON,
                    self.m._wsl_path(self.m.COLAB_CREATE_HELPER),
                    "--session",
                    self.m.SESSION,
                    "--gpu",
                    self.m.COLAB_ACCELERATOR,
                    "--auth",
                    self.m.COLAB_AUTH,
                ),
            ],
        )

    def test_other_named_sessions_are_never_released(self) -> None:
        events = []
        listing = (
            "[other-worker] gpu-other-runtime | Hardware: L4 | Variant: GPU",
            ["gpu-other-runtime"],
        )

        def run(arguments):
            events.append(("run", *arguments))
            return subprocess.CompletedProcess(arguments, 0, "", "")

        with (
            patch.object(self.m, "_sessions", return_value=listing),
            patch.object(
                self.m,
                "_release_exact_endpoint",
                side_effect=AssertionError("must not release another worker"),
            ),
            patch.object(self.m, "_run", side_effect=run),
        ):
            self.assertTrue(self.m.ensure_fresh_colab_session())

        self.assertEqual(len(events), 1)

    def test_target_release_preserves_other_named_sessions(self) -> None:
        events = []
        listing = (
            "\n".join(
                (
                    f"[{self.m.SESSION}] gpu-target-runtime | Hardware: L4 | Variant: GPU",
                    "[other-worker] gpu-other-runtime | Hardware: L4 | Variant: GPU",
                )
            ),
            ["gpu-target-runtime", "gpu-other-runtime"],
        )

        def release(endpoint: str) -> bool:
            events.append(("release", endpoint))
            return True

        with (
            patch.object(self.m, "_sessions", return_value=listing),
            patch.object(self.m, "_release_exact_endpoint", side_effect=release),
            patch.object(
                self.m,
                "_run",
                return_value=subprocess.CompletedProcess((), 0, "", ""),
            ),
        ):
            self.assertTrue(self.m.ensure_fresh_colab_session())

        self.assertEqual(events, [("release", "gpu-target-runtime")])

    def test_colab_rebuild_reuploads_checkpoints_before_launch(self) -> None:
        events = []
        with (
            patch.object(
                self.m,
                "colab_session_transport_healthy",
                return_value=False,
            ),
            patch.object(
                self.m,
                "ensure_fresh_colab_session",
                side_effect=lambda: events.append("new") or True,
            ),
            patch.object(
                self.m,
                "retry_until_success",
                side_effect=lambda operation, **kwargs: operation(),
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

    def test_live_new_session_is_reused_after_interrupted_upload(self) -> None:
        events = []
        with (
            patch.object(
                self.m,
                "colab_session_transport_healthy",
                return_value=True,
            ),
            patch.object(
                self.m,
                "ensure_fresh_colab_session",
                side_effect=AssertionError("live session must not be released"),
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

        self.assertEqual(events, ["restore", "launch"])

    def test_forced_rebuild_replaces_transport_even_when_assignment_lists(self) -> None:
        events = []
        with (
            patch.object(self.m, "colab_session_transport_healthy", return_value=True),
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
            patch.object(
                self.m,
                "retry_until_success",
                side_effect=lambda operation, **kwargs: operation(),
            ),
        ):
            self.assertTrue(
                self.m.rebuild_colab_from_checkpoints(force_fresh_session=True)
            )

        self.assertEqual(events, ["new", "restore", "launch"])

    def test_failed_rebuild_accepts_durable_progress_before_retry(self) -> None:
        episode = {
            "status": "active",
            "baseline_completed_rows": 1_197_155,
            "baseline_valid_chunks": 60,
        }
        with (
            patch.object(self.m, "rebuild_colab_from_checkpoints", return_value=False),
            patch.object(self.m, "recovery_episode_advanced", return_value=True),
            patch.object(self.m, "log") as log,
        ):
            accepted = self.m.rebuild_once_or_accept_durable_progress(
                episode,
                lambda: self.m.rebuild_colab_from_checkpoints(force_fresh_session=True),
            )

        self.assertTrue(accepted)
        self.assertTrue(
            any(
                call.args
                and "rebuild_failed_after_durable_advance=true" in call.args[0]
                for call in log.mock_calls
            )
        )

    def test_remote_exec_waits_longer_than_the_remote_deadline(self) -> None:
        captured: dict[str, object] = {}

        def run(command, *, timeout_seconds=180):
            captured["command"] = list(command)
            captured["timeout_seconds"] = timeout_seconds
            return subprocess.CompletedProcess(list(command), 0, "", "")

        with patch.object(self.m, "_run", side_effect=run):
            self.assertTrue(
                self.m._remote_exec(self.m.ROOT / ".scratch" / "install.py", 900)
            )

        self.assertIn("--timeout", captured["command"])
        remote_deadline = captured["command"][
            captured["command"].index("--timeout") + 1
        ]
        self.assertEqual(remote_deadline, "900")
        self.assertGreater(captured["timeout_seconds"], float(remote_deadline))

    def test_upload_is_not_killed_by_the_default_command_budget(self) -> None:
        captured: dict[str, object] = {}

        def run(command, *, timeout_seconds=180):
            captured["timeout_seconds"] = timeout_seconds
            return subprocess.CompletedProcess(list(command), 0, "", "")

        with patch.object(self.m, "_run", side_effect=run):
            self.assertTrue(self.m._upload(SCRIPT, "/content/probe.py"))

        self.assertEqual(captured["timeout_seconds"], self.m.UPLOAD_TIMEOUT_SECONDS)
        self.assertGreater(self.m.UPLOAD_TIMEOUT_SECONDS, 180)

    def test_failed_rebuild_rechecks_durable_progress_after_sync_interval(self) -> None:
        episode = {
            "status": "active",
            "baseline_completed_rows": 1_777_155,
            "baseline_valid_chunks": 89,
        }
        sleeps: list[float] = []
        with (
            patch.object(self.m, "rebuild_colab_from_checkpoints", return_value=False),
            patch.object(
                self.m, "recovery_episode_advanced", side_effect=[False, True]
            ),
            patch.object(self.m, "_durable_frontier", return_value=(1_797_155, 90)),
            patch.object(self.m, "write_supervisor_status"),
            patch.object(self.m, "log"),
        ):
            accepted = self.m.rebuild_once_or_accept_durable_progress(
                episode,
                lambda: self.m.rebuild_colab_from_checkpoints(force_fresh_session=True),
                sleep=sleeps.append,
            )

        self.assertTrue(accepted)
        self.assertEqual(sleeps, [90])

    def test_failed_rebuild_without_durable_progress_still_fails(self) -> None:
        episode = {
            "status": "active",
            "baseline_completed_rows": 1_777_155,
            "baseline_valid_chunks": 89,
        }
        sleeps: list[float] = []
        with (
            patch.object(self.m, "rebuild_colab_from_checkpoints", return_value=False),
            patch.object(self.m, "recovery_episode_advanced", return_value=False),
            patch.object(self.m, "log"),
        ):
            accepted = self.m.rebuild_once_or_accept_durable_progress(
                episode,
                lambda: self.m.rebuild_colab_from_checkpoints(force_fresh_session=True),
                sleep=sleeps.append,
            )

        self.assertFalse(accepted)
        self.assertEqual(sleeps, [90])

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

    def test_recovery_episode_reuses_persisted_baseline_until_advanced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original_episode_path = self.m.EPISODE_PATH
            self.m.EPISODE_PATH = Path(directory) / "episode.json"
            frontiers = iter(((237155, 12), (257155, 13)))
            try:
                with patch.object(
                    self.m, "_durable_frontier", return_value=(237155, 12)
                ):
                    first = self.m.load_or_start_recovery_episode()
                with patch.object(
                    self.m, "_durable_frontier", return_value=(999999, 99)
                ):
                    second = self.m.load_or_start_recovery_episode()
                self.assertEqual(first["baseline_completed_rows"], 237155)
                self.assertEqual(first["baseline_valid_chunks"], 12)
                self.assertEqual(second["baseline_completed_rows"], 237155)
                self.assertEqual(second["baseline_valid_chunks"], 12)
                with patch.object(self.m, "_durable_frontier", side_effect=frontiers):
                    self.assertFalse(self.m.recovery_episode_advanced(first))
                    self.assertTrue(self.m.recovery_episode_advanced(first))
            finally:
                self.m.EPISODE_PATH = original_episode_path

    def test_valid_chunk_count_rejects_incomplete_npz(self) -> None:
        required = (
            "raw_index.npy",
            "anomaly.npy",
            "score.npy",
            "site_id.npy",
            "building_id.npy",
        )
        with tempfile.TemporaryDirectory() as directory:
            chunks = Path(directory)
            valid = chunks / "rows_00000000_00020000.npz"
            invalid = chunks / "rows_00020000_00040000.npz"
            with zipfile.ZipFile(valid, "w") as archive:
                for name in required:
                    archive.writestr(name, b"test")
            with zipfile.ZipFile(invalid, "w") as archive:
                archive.writestr("score.npy", b"test")
            with patch.object(
                self.m, "_local_tail_chunks", return_value=[valid, invalid]
            ):
                self.assertEqual(self.m._valid_local_tail_chunks(), [valid])

    def test_monitors_start_only_after_worker_is_healthy(self) -> None:
        events = []
        episode = {
            "status": "active",
            "baseline_completed_rows": 237155,
            "baseline_valid_chunks": 12,
        }
        with tempfile.TemporaryDirectory() as directory:
            original_episode_path = self.m.EPISODE_PATH
            self.m.EPISODE_PATH = Path(directory) / "episode.json"
            self.m.EPISODE_PATH.write_text(json.dumps(episode), encoding="utf-8")
            try:
                with (
                    patch.object(
                        self.m,
                        "_inspect_colab",
                        side_effect=[None, {"alive": True}],
                    ),
                    patch.object(
                        self.m,
                        "synced_heartbeat_fresh",
                        return_value=False,
                    ),
                    patch.object(
                        self.m,
                        "local_sync_reports_missing_remote_work",
                        return_value=False,
                    ),
                    patch.object(
                        self.m,
                        "retry_until_success",
                        side_effect=lambda operation: events.append("rebuild"),
                    ),
                    patch.object(
                        self.m,
                        "ensure_colab_monitors",
                        side_effect=lambda: events.append("monitors"),
                    ),
                    patch.object(
                        self.m,
                        "recovery_episode_advanced",
                        return_value=True,
                    ),
                    patch.object(
                        self.m,
                        "close_recovery_episode",
                        side_effect=lambda value: events.append("close"),
                    ),
                ):
                    self.m.recover_colab_until_healthy(
                        return_after_episode_success=True
                    )
            finally:
                self.m.EPISODE_PATH = original_episode_path

        self.assertEqual(events, ["rebuild", "monitors", "close"])

    def test_monitor_children_remain_attached_to_persistent_supervisor(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        monitor_launcher = source.split("def _launch_detached_monitor", 1)[1].split(
            "def ensure_colab_monitors", 1
        )[0]
        self.assertIn("CREATE_NEW_PROCESS_GROUP", monitor_launcher)
        self.assertIn("CREATE_NO_WINDOW", monitor_launcher)
        self.assertNotIn("DETACHED_PROCESS", monitor_launcher)


if __name__ == "__main__":
    unittest.main()
