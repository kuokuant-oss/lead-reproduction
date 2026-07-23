from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
LOCAL_WORK = PROC / "m5_tabpfn_canonical_full_test_context100000.work"
LOCAL_STATE = PROC / "m5_tabpfn_canonical_full_test_context100000.state.json"
LOCAL_STDOUT = (
    PROC / "m5_tabpfn_canonical_full_test_context100000.controller.stdout.log"
)
LOCAL_STDERR = (
    PROC / "m5_tabpfn_canonical_full_test_context100000.controller.stderr.log"
)
TAIL_INPUT = PROC / "m5_tabpfn_distributed_context100000" / "tail"
TAIL_RESULTS = PROC / "m5_tabpfn_distributed_context100000" / "tail-results"
UPLOAD_PARTS = PROC / "m5_tabpfn_upload_parts"
LOCK_PATH = PROC / "m5_tabpfn_recovery_supervisor.lock"
LOG_PATH = PROC / "m5_tabpfn_recovery_supervisor.log"
SESSION = "lead-tabpfn-tail"
COLAB = "/home/tonykuo/.local/bin/colab"
COLAB_PYTHON = "/home/tonykuo/.local/share/uv/tools/google-colab-cli/bin/python"
REMOTE_ROOT = "/content/lead_tabpfn_tail"
EXPECTED_LOCAL_ROWS = 5_060_000
EXPECTED_LOCAL_CHECKPOINTS = 253
EXPECTED_COLAB_CHECKPOINTS = 254
ENDPOINT_RE = re.compile(r"^gpu-[a-z0-9-]+$")


class SupervisorAlreadyRunning(RuntimeError):
    pass


class RecoveryInvariantError(RuntimeError):
    pass


def log(message: str) -> None:
    line = f"{int(time.time())} {message}"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class SingletonLock:
    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                try:
                    payload = json.loads(self.path.read_text(encoding="utf-8"))
                    owner_pid = int(payload.get("pid", 0))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    owner_pid = 0
                if owner_pid and _pid_alive(owner_pid):
                    raise SupervisorAlreadyRunning(
                        f"recovery supervisor already running as PID {owner_pid}"
                    )
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid(), "started_at": time.time()}, handle)
            self.acquired = True
            return self
        raise SupervisorAlreadyRunning("could not acquire recovery supervisor lock")

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False


def retry_until_success(
    operation: Callable[[], bool],
    *,
    sleep: Callable[[float], None] = time.sleep,
    delays: Sequence[float] = (30, 60, 60, 60),
) -> None:
    if not delays or any(delay < 0 for delay in delays):
        raise ValueError("delays must contain nonnegative values")
    attempt = 0
    while True:
        attempt += 1
        if operation():
            return
        delay = delays[min(attempt - 1, len(delays) - 1)]
        log(f"retry_pending=true attempt={attempt} delay_seconds={delay}")
        sleep(delay)


def classify_colab_failure(message: str) -> str:
    expected = (
        "service unavailable",
        "adc credentials",
        "toomanyassignments",
        "not found",
        "no active sessions",
        "appears to be lost",
        "temporarily unavailable",
    )
    lowered = message.lower()
    return "retry" if any(value in lowered for value in expected) else "investigate"


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _output_text(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stdout or "") + (result.stderr or "")


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise RecoveryInvariantError(f"expected Windows drive path: {resolved}")
    tail = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def _colab(*arguments: str) -> subprocess.CompletedProcess[str]:
    return _run(["wsl.exe", "-d", "Ubuntu", "--", COLAB, "--auth", "adc", *arguments])


def _sessions() -> tuple[str, list[str]]:
    result = _colab("sessions")
    text = _output_text(result)
    endpoints = []
    for line in text.splitlines():
        match = re.search(r"\[(?:[^]]*)\]\s+(gpu-[a-z0-9-]+)\s+\|", line)
        if match:
            endpoints.append(match.group(1))
    return text, endpoints


def _release_exact_endpoint(endpoint: str) -> bool:
    if not ENDPOINT_RE.fullmatch(endpoint):
        raise RecoveryInvariantError(f"unsafe Colab endpoint: {endpoint}")
    code = (
        "from colab_cli.auth import AuthProvider; "
        "from colab_cli.common import state; "
        "state.auth_provider=AuthProvider.ADC; "
        f"state.client.unassign('{endpoint}')"
    )
    result = _run(
        [
            "wsl.exe",
            "-d",
            "Ubuntu",
            "--",
            COLAB_PYTHON,
            "-c",
            code,
        ]
    )
    if result.returncode == 0:
        log(f"released_exact_endpoint={endpoint}")
        return True
    log(f"endpoint_release_failed=true endpoint={endpoint}")
    return False


def ensure_fresh_colab_session() -> bool:
    """Discard any unusable assignment before allocating a brand-new T4."""
    _, endpoints = _sessions()
    if len(endpoints) > 1:
        raise RecoveryInvariantError(
            "multiple Colab assignments; refusing broad cleanup"
        )
    if endpoints and not _release_exact_endpoint(endpoints[0]):
        return False
    result = _colab("new", "-s", SESSION, "--gpu", "T4")
    if result.returncode == 0:
        log("colab_session_ready=true accelerator=T4")
        return True
    category = classify_colab_failure(_output_text(result))
    log(f"colab_session_create_failed=true category={category}")
    return False


def _remote_exec(script: Path, setup_timeout: int = 900) -> bool:
    result = _colab(
        "exec",
        "-s",
        SESSION,
        "-f",
        _wsl_path(script),
        "--timeout",
        str(setup_timeout),
    )
    return result.returncode == 0


def _upload(local: Path, remote: str) -> bool:
    result = _colab("upload", "-s", SESSION, _wsl_path(local), remote)
    return result.returncode == 0


def _local_tail_chunks() -> list[Path]:
    return sorted((TAIL_RESULTS / "chunks").glob("rows_*.npz"))


def _tail_durable_rows() -> int:
    path = TAIL_RESULTS / "progress.json"
    if not path.exists():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8-sig"))["completed_rows"])


def restore_colab_files() -> bool:
    if not _remote_exec(ROOT / ".scratch" / "create_colab_tail_dirs.py", 120):
        return False
    uploads: list[tuple[Path, str]] = []
    uploads.extend(
        (path, f"{REMOTE_ROOT}/{path.name}")
        for path in sorted(UPLOAD_PARTS.iterdir())
        if path.is_file()
    )
    uploads.extend(
        (
            TAIL_INPUT / name,
            f"{REMOTE_ROOT}/{name}",
        )
        for name in ("metadata.npz", "model.portable.tabpfn_fit", "manifest.json")
    )
    uploads.append(
        (
            ROOT / "scripts" / "run_m5_tabpfn_portable_shard.py",
            f"{REMOTE_ROOT}/run_m5_tabpfn_portable_shard.py",
        )
    )
    uploads.extend(
        (path, f"{REMOTE_ROOT}/work/chunks/{path.name}")
        for path in _local_tail_chunks()
    )
    for local, remote in uploads:
        if not local.is_file():
            raise RecoveryInvariantError(f"missing recovery input: {local}")
        if not _upload(local, remote):
            log(f"upload_failed=true file={local.name}")
            return False
    if not _remote_exec(ROOT / ".scratch" / "reassemble_colab_tail.py", 300):
        raise RecoveryInvariantError("Colab input SHA-256 verification failed")
    for script in (
        "install_colab_tabpfn.py",
        "install_colab_exact_runtime.py",
        "inspect_colab_python_deps.py",
    ):
        if not _remote_exec(ROOT / ".scratch" / script, 900):
            log(f"runtime_setup_failed=true script={script}")
            return False
    return True


def _inspect_colab() -> dict[str, Any] | None:
    result = _colab(
        "exec",
        "-s",
        SESSION,
        "-f",
        _wsl_path(ROOT / ".scratch" / "inspect_colab_tail_status.py"),
        "--timeout",
        "120",
    )
    if result.returncode != 0:
        return None
    for line in (result.stdout or "").splitlines():
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None


def launch_and_verify_colab(sleep: Callable[[float], None] = time.sleep) -> bool:
    if not _remote_exec(ROOT / ".scratch" / "launch_colab_tail.py", 120):
        return False
    durable_rows = _tail_durable_rows()
    durable_chunks = len(_local_tail_chunks())
    sleep(45)
    first = _inspect_colab()
    if not first or not first.get("alive"):
        return False
    if int(first.get("chunk_count", -1)) < durable_chunks:
        raise RecoveryInvariantError("remote checkpoint count regressed")
    first_rows = int(first.get("heartbeat.json", {}).get("completed_rows", -1))
    if first_rows < durable_rows:
        raise RecoveryInvariantError("remote heartbeat resumed behind durable rows")
    sleep(45)
    second = _inspect_colab()
    if not second or not second.get("alive"):
        return False
    second_rows = int(second.get("heartbeat.json", {}).get("completed_rows", -1))
    if second_rows <= first_rows:
        return False
    log(
        "colab_recovery_verified=true "
        f"durable_rows={durable_rows} heartbeat_rows={second_rows}"
    )
    return True


def rebuild_colab_from_checkpoints() -> bool:
    """Create a fresh runtime, restore all durable inputs, and verify resume."""
    if not ensure_fresh_colab_session():
        return False
    if not restore_colab_files():
        return False
    return launch_and_verify_colab()


def recover_colab_until_healthy() -> None:
    status = _inspect_colab()
    if status and status.get("alive"):
        log("colab_worker_already_healthy=true")
        return
    retry_until_success(rebuild_colab_from_checkpoints)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def local_worker_healthy() -> bool:
    heartbeat_path = LOCAL_WORK / "heartbeat.json"
    if not heartbeat_path.exists():
        return False
    heartbeat = _read_json(heartbeat_path)
    return _pid_alive(int(heartbeat.get("pid", 0)))


def wait_for_local_worker_health(
    process: subprocess.Popen[Any],
    *,
    healthy: Callable[[], bool] = local_worker_healthy,
    sleep: Callable[[float], None] = time.sleep,
    delays: Sequence[float] = (15, 30, 60),
) -> None:
    """Wait without a wall-time deadline while the launched controller remains alive."""
    attempt = 0
    while not healthy():
        exit_code = process.poll()
        if exit_code is not None:
            raise RecoveryInvariantError(
                f"local recovery controller exited before health verification: {exit_code}"
            )
        delay = delays[min(attempt, len(delays) - 1)]
        attempt += 1
        log(f"local_startup_pending=true health_sample={attempt} delay_seconds={delay}")
        sleep(delay)


def ensure_local_worker(sleep: Callable[[float], None] = time.sleep) -> None:
    progress_path = LOCAL_WORK / "progress.json"
    if progress_path.exists():
        progress = _read_json(progress_path)
        if int(progress.get("rows_completed", 0)) >= EXPECTED_LOCAL_ROWS:
            log("local_complete=true")
            return
    if local_worker_healthy():
        log("local_worker_already_healthy=true")
        return
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        str(ROOT / "scripts" / "run_m5_tabpfn_canonical_full_test.py"),
        "--context-rows",
        "100000",
        "--query-microbatch-size",
        "384",
        "--min-query-microbatch-size",
        "256",
        "--checkpoint-rows",
        "20000",
        "--resume",
    ]
    creation_flags = 0
    if os.name == "nt":
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW
        )
    with (
        LOCAL_STDOUT.open("ab", buffering=0) as stdout,
        LOCAL_STDERR.open("ab", buffering=0) as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
            close_fds=True,
        )
    wait_for_local_worker_health(process, sleep=sleep)
    progress = _read_json(progress_path)
    if progress.get("fit_action") != "loaded":
        raise RecoveryInvariantError("local recovery did not load fitted state")
    log(f"local_recovery_verified=true rows={progress.get('rows_completed', 0)}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=("both", "local", "colab"),
        default="both",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        with SingletonLock(LOCK_PATH):
            log(f"supervisor_started=true scope={args.scope}")
            if args.scope in ("both", "local"):
                ensure_local_worker()
            if args.scope in ("both", "colab"):
                recover_colab_until_healthy()
            log("supervisor_completed=true")
    except SupervisorAlreadyRunning as error:
        log(f"duplicate_supervisor_skipped=true reason={error}")
        return 0
    except RecoveryInvariantError as error:
        log(f"invariant_failure=true reason={error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
