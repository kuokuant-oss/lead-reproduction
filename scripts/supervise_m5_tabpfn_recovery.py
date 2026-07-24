from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import re
import subprocess
import time
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
COLAB_SHARD = os.environ.get("TABPFN_COLAB_SHARD", "tail")
if COLAB_SHARD not in {"head", "tail"}:
    raise ValueError(f"unsupported Colab shard: {COLAB_SHARD}")
LOCAL_WORK = PROC / "m5_tabpfn_canonical_full_test_context100000.work"
LOCAL_STATE = PROC / "m5_tabpfn_canonical_full_test_context100000.state.json"
LOCAL_STDOUT = (
    PROC / "m5_tabpfn_canonical_full_test_context100000.controller.stdout.log"
)
LOCAL_STDERR = (
    PROC / "m5_tabpfn_canonical_full_test_context100000.controller.stderr.log"
)
TAIL_INPUT = PROC / "m5_tabpfn_distributed_context100000" / COLAB_SHARD
TAIL_RESULTS = PROC / "m5_tabpfn_distributed_context100000" / f"{COLAB_SHARD}-results"
UPLOAD_PARTS = PROC / "m5_tabpfn_upload_parts"
HEAD_UPLOAD_PARTS = PROC / "m5_tabpfn_head_upload_parts"
STATE_STEM = (
    "m5_tabpfn_recovery" if COLAB_SHARD == "tail" else "m5_tabpfn_colab_head_recovery"
)
LOCK_PATH = PROC / f"{STATE_STEM}_supervisor.lock"
LOG_PATH = PROC / f"{STATE_STEM}_supervisor.log"
EPISODE_PATH = PROC / f"{STATE_STEM}_episode.json"
MONITORS_PATH = PROC / f"{STATE_STEM}_monitors.json"
STATUS_PATH = PROC / f"{STATE_STEM}_status.json"
SYNC_SCRIPT = (
    ROOT / "scripts" / "sync_m5_tabpfn_colab_tail.ps1"
    if COLAB_SHARD == "tail"
    else ROOT / "scripts" / "sync_m5_tabpfn_colab_head.ps1"
)
KEEPALIVE_SCRIPT = (
    ROOT / "scripts" / "monitor_m5_tabpfn_colab_keepalive.ps1"
    if COLAB_SHARD == "tail"
    else ROOT / "scripts" / "monitor_m5_tabpfn_colab_head_keepalive.ps1"
)
SESSION = "lead-tabpfn-tail" if COLAB_SHARD == "tail" else "lead-tabpfn-tail-2"
COLAB = "/home/tonykuo/.local/bin/colab"
COLAB_PYTHON = "/home/tonykuo/.local/share/uv/tools/google-colab-cli/bin/python"
COLAB_CREATE_HELPER = ROOT / "scripts" / "create_colab_session.py"
COLAB_HOME = os.environ.get("TABPFN_COLAB_HOME", "/home/tonykuo")
COLAB_AUTH = os.environ.get("TABPFN_COLAB_AUTH", "adc")
COLAB_ACCELERATOR = os.environ.get("TABPFN_COLAB_ACCELERATOR", "T4")
REMOTE_ROOT = f"/content/lead_tabpfn_{COLAB_SHARD}"
EXPECTED_LOCAL_ROWS = 5_060_000
EXPECTED_LOCAL_CHECKPOINTS = 253
EXPECTED_COLAB_CHECKPOINTS = 254 if COLAB_SHARD == "tail" else 253
REMOTE_DIR_SCRIPT = (
    ROOT / ".scratch" / "create_colab_tail_dirs.py"
    if COLAB_SHARD == "tail"
    else ROOT / "scripts" / "create_m5_tabpfn_colab_head_dirs.py"
)
REASSEMBLE_SCRIPT = (
    ROOT / ".scratch" / "reassemble_colab_tail.py"
    if COLAB_SHARD == "tail"
    else ROOT / "scripts" / "reassemble_m5_tabpfn_colab_head.py"
)
INSPECT_SCRIPT = (
    ROOT / ".scratch" / "inspect_colab_tail_status.py"
    if COLAB_SHARD == "tail"
    else ROOT / "scripts" / "inspect_m5_tabpfn_colab_head.py"
)
WORKER_LAUNCHER = ROOT / "scripts" / f"launch_m5_tabpfn_colab_{COLAB_SHARD}.py"
ENDPOINT_RE = re.compile(r"^gpu-[a-z0-9-]+$")
# The local subprocess must outlive the deadline handed to the remote command,
# so a slow-but-healthy step is not reported as a deterministic failure.
LOCAL_TIMEOUT_MARGIN_SECONDS = 120
UPLOAD_TIMEOUT_SECONDS = 900
INSPECT_TIMEOUT_SECONDS = 120
# A resumed worker revalidates every uploaded checkpoint before its first
# heartbeat, so the startup window scales with the shard's checkpoint count.
LAUNCH_HEARTBEAT_POLL_SECONDS = 45
LAUNCH_HEARTBEAT_MAX_SAMPLES = 20


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


def write_supervisor_status(phase: str, **fields: Any) -> None:
    """Publish an atomic, machine-readable status so a stuck loop is visible."""
    payload: dict[str, Any] = {
        "phase": phase,
        "timestamp": time.time(),
        "pid": os.getpid(),
        "shard": COLAB_SHARD,
        "session": SESSION,
    }
    if STATUS_PATH.is_file():
        try:
            previous = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
            if isinstance(previous, dict):
                payload.update(
                    {
                        key: value
                        for key, value in previous.items()
                        if key not in {"phase", "timestamp", "pid"}
                    }
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    payload.update(fields)
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, STATUS_PATH)


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
    delay_for_attempt: Callable[[int], float] | None = None,
) -> None:
    if delay_for_attempt is None and (not delays or any(delay < 0 for delay in delays)):
        raise ValueError("delays must contain nonnegative values")
    attempt = 0
    while True:
        attempt += 1
        write_supervisor_status("attempting", attempt=attempt, last_error=None)
        try:
            succeeded = operation()
        except RecoveryInvariantError:
            raise
        except Exception as error:  # keep the persistent supervisor alive on transient CLI/runtime bugs
            detail = f"{type(error).__name__}: {error}".replace("\n", " ")[:500]
            write_supervisor_status(
                "operation_exception",
                attempt=attempt,
                last_error=detail,
            )
            log(f"operation_exception=true attempt={attempt} detail={detail}")
            succeeded = False
        if succeeded:
            write_supervisor_status("step_succeeded", attempt=attempt)
            return
        if delay_for_attempt is None:
            delay = delays[min(attempt - 1, len(delays) - 1)]
        else:
            delay = delay_for_attempt(attempt)
            if delay < 0:
                raise ValueError("delay_for_attempt returned a negative delay")
        next_retry_at = time.time() + delay
        write_supervisor_status(
            "retrying",
            attempt=attempt,
            next_retry_at=next_retry_at,
        )
        log(f"retry_pending=true attempt={attempt} delay_seconds={delay}")
        sleep(delay)


def colab_allocation_delay(
    attempt: int,
    *,
    uniform: Callable[[float, float], float] = random.uniform,
) -> float:
    """Back off T4 allocation without slowing post-allocation repair steps."""
    if attempt <= 0:
        raise ValueError("attempt must be positive")
    exponential_minutes = (1, 2, 4, 8, 16)
    if attempt <= len(exponential_minutes):
        return float(exponential_minutes[attempt - 1] * 60)
    return float(round(uniform(15 * 60, 30 * 60)))


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


def _run(
    command: Sequence[str], *, timeout_seconds: float = 180
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            cwd=ROOT,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return subprocess.CompletedProcess(
            list(command),
            124,
            error.stdout or "",
            f"supervisor subprocess timeout after {timeout_seconds:g}s",
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


def _colab(
    *arguments: str, timeout_seconds: float = 180
) -> subprocess.CompletedProcess[str]:
    """Run one Colab CLI command.

    ``timeout_seconds`` bounds the local subprocess.  Callers that also grant
    the remote side a budget must pass a larger value here, otherwise the
    supervisor kills a healthy command before the remote deadline it asked
    for and a transient slow step looks like a deterministic failure.
    """
    return _run(
        [
            "wsl.exe",
            "-d",
            "Ubuntu",
            "--",
            "env",
            f"HOME={COLAB_HOME}",
            COLAB,
            "--auth",
            COLAB_AUTH,
            *arguments,
        ],
        timeout_seconds=timeout_seconds,
    )


def _sessions() -> tuple[str, list[str]]:
    result = _colab("sessions")
    text = _output_text(result)
    endpoints = []
    for line in text.splitlines():
        match = re.search(r"\[(?:[^]]*)\]\s+(gpu-[a-z0-9-]+)\s+\|", line)
        if match:
            endpoints.append(match.group(1))
    return text, endpoints


def _named_session_endpoint(text: str, session: str = SESSION) -> str | None:
    match = re.search(
        rf"^\[{re.escape(session)}\]\s+(gpu-[a-z0-9-]+)\s+\|",
        text,
        re.MULTILINE,
    )
    return match.group(1) if match else None


def _release_exact_endpoint(endpoint: str) -> bool:
    if not ENDPOINT_RE.fullmatch(endpoint):
        raise RecoveryInvariantError(f"unsafe Colab endpoint: {endpoint}")
    if COLAB_AUTH not in {"adc", "oauth2"}:
        raise RecoveryInvariantError(f"unsupported Colab auth provider: {COLAB_AUTH}")
    provider = "AuthProvider.ADC" if COLAB_AUTH == "adc" else "AuthProvider.OAUTH2"
    code = (
        "from colab_cli.auth import AuthProvider; "
        "from colab_cli.common import state; "
        f"state.auth_provider={provider}; "
        f"state.client.unassign('{endpoint}')"
    )
    result = _run(
        [
            "wsl.exe",
            "-d",
            "Ubuntu",
            "--",
            "env",
            f"HOME={COLAB_HOME}",
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
    """Replace only this supervisor's named assignment, preserving all others."""
    text, _ = _sessions()
    endpoint = _named_session_endpoint(text)
    if endpoint and not _release_exact_endpoint(endpoint):
        return False
    result = _run(
        [
            "wsl.exe",
            "-d",
            "Ubuntu",
            "--",
            "env",
            f"HOME={COLAB_HOME}",
            COLAB_PYTHON,
            _wsl_path(COLAB_CREATE_HELPER),
            "--session",
            SESSION,
            "--gpu",
            COLAB_ACCELERATOR,
            "--auth",
            COLAB_AUTH,
        ]
    )
    if result.returncode == 0:
        log(f"colab_session_ready=true accelerator={COLAB_ACCELERATOR}")
        return True
    output = _output_text(result)
    category = classify_colab_failure(output)
    detail = next(
        (line for line in output.splitlines() if line.startswith("{")),
        "{}",
    )
    log(f"colab_session_create_failed=true category={category} detail={detail}")
    return False


def colab_session_transport_healthy() -> bool:
    """Return true only when the named assignment exists on the server.

    colab-cli 0.6.0 exits zero for ``status -s NAME`` even when NAME is not
    found.  The server-backed sessions listing is therefore the authoritative
    transport check; an exit code alone would create an infinite upload loop
    against a missing runtime.
    """
    text, _ = _sessions()
    return _named_session_endpoint(text) is not None


def _remote_exec(script: Path, setup_timeout: int = 900) -> bool:
    result = _colab(
        "exec",
        "-s",
        SESSION,
        "-f",
        _wsl_path(script),
        "--timeout",
        str(setup_timeout),
        timeout_seconds=setup_timeout + LOCAL_TIMEOUT_MARGIN_SECONDS,
    )
    if result.returncode == 0:
        return True
    # Keep the failure actionable.  Previously a non-zero remote exec was
    # silently collapsed into ``False``, so a transient Colab transport error
    # looked identical to a deterministic setup bug.
    detail = " ".join(_output_text(result).split())
    detail = re.sub(
        r"(?i)(authorization|access[_-]?token|refresh[_-]?token|client_secret)\s*[:=]\s*\S+",
        r"\1=<redacted>",
        detail,
    )[:800]
    log(
        "remote_exec_failed=true "
        f"script={script.name} returncode={result.returncode} detail={detail or '<empty>'}"
    )
    write_supervisor_status(
        "remote_exec_failed",
        script=script.name,
        returncode=result.returncode,
        last_error=detail or f"remote exec exited {result.returncode}",
    )
    return False


def _upload(local: Path, remote: str) -> bool:
    # A 64 MiB features/foundation part can legitimately exceed the default
    # command budget on a slow link; killing it discards a verified restore.
    result = _colab(
        "upload",
        "-s",
        SESSION,
        _wsl_path(local),
        remote,
        timeout_seconds=UPLOAD_TIMEOUT_SECONDS,
    )
    return result.returncode == 0


def _local_tail_chunks() -> list[Path]:
    return sorted((TAIL_RESULTS / "chunks").glob("rows_*.npz"))


def _valid_local_tail_chunks() -> list[Path]:
    required = {
        "raw_index.npy",
        "anomaly.npy",
        "score.npy",
        "site_id.npy",
        "building_id.npy",
    }
    valid = []
    for path in _local_tail_chunks():
        try:
            with zipfile.ZipFile(path) as archive:
                if set(archive.namelist()) == required and archive.testzip() is None:
                    valid.append(path)
        except (OSError, zipfile.BadZipFile):
            continue
    return valid


def _tail_durable_rows() -> int:
    path = TAIL_RESULTS / "progress.json"
    if not path.exists():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8-sig"))["completed_rows"])


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _durable_frontier() -> tuple[int, int]:
    return _tail_durable_rows(), len(_valid_local_tail_chunks())


def load_or_start_recovery_episode() -> dict[str, Any]:
    if EPISODE_PATH.exists():
        episode = _read_json(EPISODE_PATH)
        if episode.get("status") == "active":
            return episode
    rows, chunks = _durable_frontier()
    episode = {
        "status": "active",
        "baseline_completed_rows": rows,
        "baseline_valid_chunks": chunks,
        "started_at": time.time(),
    }
    _atomic_write_json(EPISODE_PATH, episode)
    log(f"recovery_episode_started=true baseline_rows={rows} baseline_chunks={chunks}")
    return episode


def recovery_episode_advanced(episode: dict[str, Any]) -> bool:
    rows, chunks = _durable_frontier()
    return rows > int(episode["baseline_completed_rows"]) and chunks > int(
        episode["baseline_valid_chunks"]
    )


def colab_formal_shard_complete() -> bool:
    return (
        len(_valid_local_tail_chunks()) >= EXPECTED_COLAB_CHECKPOINTS
        and (TAIL_RESULTS / "result.json").is_file()
    )


def close_recovery_episode(episode: dict[str, Any]) -> None:
    rows, chunks = _durable_frontier()
    completed = dict(episode)
    completed.update(
        status="completed",
        completed_at=time.time(),
        completed_rows=rows,
        completed_valid_chunks=chunks,
    )
    _atomic_write_json(EPISODE_PATH, completed)
    log(f"recovery_episode_completed=true durable_rows={rows} valid_chunks={chunks}")


def _monitor_command(script: Path) -> list[str]:
    return [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
    ]


def _launch_detached_monitor(script: Path) -> subprocess.Popen[Any]:
    if not script.is_file():
        raise RecoveryInvariantError(f"missing recovery monitor: {script}")
    creation_flags = 0
    if os.name == "nt":
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    return subprocess.Popen(
        _monitor_command(script),
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
        close_fds=True,
    )


def ensure_colab_monitors(sleep: Callable[[float], None] = time.sleep) -> None:
    state: dict[str, Any] = {}
    if MONITORS_PATH.exists():
        try:
            state = _read_json(MONITORS_PATH)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            state = {}
    specifications = {
        "sync_pid": SYNC_SCRIPT,
        "keepalive_pid": KEEPALIVE_SCRIPT,
    }
    launched: dict[str, subprocess.Popen[Any]] = {}
    for key, script in specifications.items():
        pid = int(state.get(key, 0))
        if pid and _pid_alive(pid):
            continue
        launched[key] = _launch_detached_monitor(script)
        state[key] = launched[key].pid
    state.update(session=SESSION, verified_at=time.time())
    _atomic_write_json(MONITORS_PATH, state)
    if launched:
        sleep(3)
        for key, process in launched.items():
            if process.poll() is not None or not _pid_alive(process.pid):
                raise RecoveryInvariantError(
                    f"Colab monitor failed to stay alive: {key}"
                )
        log(
            "colab_monitors_verified=true "
            + " ".join(f"{key}={state[key]}" for key in sorted(specifications))
        )


def run_runtime_setup() -> bool:
    """Install the pinned runtime; dependency inspection is diagnostic only."""
    for script in (
        "install_colab_tabpfn.py",
        "install_colab_exact_runtime.py",
        "inspect_colab_python_deps.py",
    ):
        if _remote_exec(ROOT / ".scratch" / script, 900):
            log(f"runtime_setup_complete=true script={script}")
            continue
        if script == "inspect_colab_python_deps.py":
            # This script only prints versions.  A transient exec failure here
            # must not discard a verified upload or force a full rebuild.
            log(f"runtime_inspect_warning=true script={script}")
            write_supervisor_status("runtime_inspect_warning", script=script)
            continue
        log(f"runtime_setup_failed=true script={script}")
        write_supervisor_status("restore_failed", step="runtime_setup", script=script)
        return False
    return True


def restore_colab_files() -> bool:
    write_supervisor_status("restore_start")
    log(f"restore_start=true shard={COLAB_SHARD} session={SESSION}")
    if not _remote_exec(REMOTE_DIR_SCRIPT, 120):
        write_supervisor_status("restore_failed", step="create_remote_dirs")
        log("restore_failed=true step=create_remote_dirs")
        return False
    uploads: list[tuple[Path, str]] = []
    if COLAB_SHARD == "tail":
        part_paths = sorted(UPLOAD_PARTS.iterdir())
    else:
        part_paths = [
            *sorted(HEAD_UPLOAD_PARTS.glob("features.float32.npy.part*")),
            *sorted(UPLOAD_PARTS.glob("tabpfn-v3-classifier-v3_default.ckpt.part*")),
        ]
    uploads.extend(
        (path, f"{REMOTE_ROOT}/{path.name}") for path in part_paths if path.is_file()
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
        for path in _valid_local_tail_chunks()
    )
    log(
        f"restore_upload_plan=true files={len(uploads)} durable_chunks={len(_valid_local_tail_chunks())}"
    )
    for local, remote in uploads:
        if not local.is_file():
            raise RecoveryInvariantError(f"missing recovery input: {local}")
        write_supervisor_status("uploading", file=local.name)
        if not _upload(local, remote):
            log(f"upload_failed=true file={local.name}")
            write_supervisor_status("restore_failed", step="upload", file=local.name)
            return False
        log(f"upload_complete=true file={local.name}")
    if not _remote_exec(REASSEMBLE_SCRIPT, 300):
        write_supervisor_status("restore_failed", step="sha256_reassemble")
        raise RecoveryInvariantError("Colab input SHA-256 verification failed")
    log("restore_sha256_verified=true")
    if not run_runtime_setup():
        return False
    write_supervisor_status("restore_complete", uploaded_files=len(uploads))
    log("restore_complete=true")
    return True


def _inspect_colab() -> dict[str, Any] | None:
    result = _colab(
        "exec",
        "-s",
        SESSION,
        "-f",
        _wsl_path(INSPECT_SCRIPT),
        "--timeout",
        str(INSPECT_TIMEOUT_SECONDS),
        timeout_seconds=INSPECT_TIMEOUT_SECONDS + LOCAL_TIMEOUT_MARGIN_SECONDS,
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


def _await_resumed_heartbeat(
    durable_rows: int,
    durable_chunks: int,
    sleep: Callable[[float], None],
) -> int | None:
    """Sample until the resumed worker reaches the durable frontier.

    A restored session starts without ``heartbeat.json`` -- only fixed inputs
    and ``rows_*.npz`` are uploaded -- and ``--resume`` revalidates every
    uploaded checkpoint before it predicts again.  Both effects make the first
    samples read as missing or behind, and the window grows with the checkpoint
    count, so a single fixed-delay sample turns a healthy late-shard resume into
    a fatal invariant.  Only a worker still behind the frontier after the whole
    grace period has actually regressed.
    """
    for _ in range(LAUNCH_HEARTBEAT_MAX_SAMPLES):
        sleep(LAUNCH_HEARTBEAT_POLL_SECONDS)
        sample = _inspect_colab()
        if not sample or not sample.get("alive"):
            write_supervisor_status("launch_failed", step="first_health_sample")
            log("launch_failed=true step=first_health_sample")
            return None
        if int(sample.get("chunk_count", -1)) < durable_chunks:
            raise RecoveryInvariantError("remote checkpoint count regressed")
        rows = int(sample.get("heartbeat.json", {}).get("completed_rows", -1))
        if rows >= durable_rows:
            return rows
        log(f"resume_heartbeat_pending=true rows={rows} durable_rows={durable_rows}")
    raise RecoveryInvariantError("remote heartbeat resumed behind durable rows")


def launch_and_verify_colab(sleep: Callable[[float], None] = time.sleep) -> bool:
    write_supervisor_status("launch_start")
    log(f"launch_start=true shard={COLAB_SHARD} session={SESSION}")
    if not _remote_exec(WORKER_LAUNCHER, 120):
        write_supervisor_status("launch_failed", step="worker_launcher")
        log("launch_failed=true step=worker_launcher")
        return False
    log("worker_launcher_complete=true")
    durable_rows = _tail_durable_rows()
    durable_chunks = len(_valid_local_tail_chunks())
    first_rows = _await_resumed_heartbeat(durable_rows, durable_chunks, sleep)
    if first_rows is None:
        return False
    sleep(45)
    second = _inspect_colab()
    if not second or not second.get("alive"):
        write_supervisor_status("launch_failed", step="second_health_sample")
        log("launch_failed=true step=second_health_sample")
        return False
    second_rows = int(second.get("heartbeat.json", {}).get("completed_rows", -1))
    if second_rows <= first_rows:
        write_supervisor_status(
            "launch_failed",
            step="heartbeat_not_advancing",
            first_rows=first_rows,
            second_rows=second_rows,
        )
        log(
            f"launch_failed=true step=heartbeat_not_advancing first_rows={first_rows} second_rows={second_rows}"
        )
        return False
    log(
        "colab_recovery_verified=true "
        f"durable_rows={durable_rows} heartbeat_rows={second_rows}"
    )
    return True


def synced_heartbeat_fresh(
    *,
    now: Callable[[], float] = time.time,
    max_age_seconds: float = 600,
) -> bool:
    path = TAIL_RESULTS / "heartbeat.json"
    if not path.is_file():
        return False
    try:
        heartbeat = _read_json(path)
        timestamp = float(heartbeat.get("timestamp", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return timestamp > 0 and now() - timestamp <= max_age_seconds


def local_sync_reports_missing_remote_work() -> bool:
    """Detect a reclaimed remote work tree without probing a stuck exec call."""
    path = TAIL_RESULTS / "sync.log"
    if not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    for line in reversed(lines):
        recent = line.lower()
        if (
            "checkpoints=" in recent
            or "downloaded=" in recent
            or "completed=true" in recent
        ):
            return False
        if ("session '" in recent and "not found" in recent) or (
            "file or directory not found" in recent
            and f"{REMOTE_ROOT}/work/chunks".lower() in recent
        ):
            return True
    return False


def rebuild_colab_from_checkpoints(*, force_fresh_session: bool = False) -> bool:
    """Create a fresh runtime, restore all durable inputs, and verify resume."""
    if force_fresh_session or not colab_session_transport_healthy():
        retry_until_success(
            ensure_fresh_colab_session,
            delay_for_attempt=colab_allocation_delay,
        )
    if not restore_colab_files():
        return False
    return launch_and_verify_colab()


def rebuild_once_or_accept_durable_progress(
    episode: dict[str, Any],
    operation: Callable[[], bool],
    *,
    sleep: Callable[[float], None] = time.sleep,
    resync_grace_seconds: float = 90,
) -> bool:
    """Stop retrying when a failed rebuild already advanced durable output.

    A runtime can reclaim immediately after the worker writes and syncs a new
    checkpoint.  In that case the remote health sample may return ``False``
    even though the recovery episode has met its success contract.  Treating
    that result as an ordinary failed attempt causes an unnecessary second
    full upload and loses the episode boundary.

    A transient remote exec failure can also fail the health sample while the
    worker is still predicting normally.  The sync monitor only lists remote
    chunks once per minute, so a checkpoint finished moments earlier is still
    in flight and the immediate durable check misses it.  Re-check once after
    a sync interval before discarding a runtime that may still be producing.
    The acceptance contract is unchanged: durable rows *and* structurally
    valid chunk count must both exceed the episode baseline.
    """
    succeeded = operation()
    if succeeded:
        return True
    if _accept_durable_advance(episode):
        return True
    sleep(resync_grace_seconds)
    return _accept_durable_advance(episode)


def _accept_durable_advance(episode: dict[str, Any]) -> bool:
    if not recovery_episode_advanced(episode):
        return False
    log("rebuild_failed_after_durable_advance=true")
    rows, chunks = _durable_frontier()
    write_supervisor_status(
        "recovery_episode_advanced",
        durable_rows=rows,
        durable_chunks=chunks,
    )
    return True


def recover_colab_until_healthy(
    *,
    return_after_episode_success: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    episode: dict[str, Any] | None = None
    if EPISODE_PATH.exists():
        candidate = _read_json(EPISODE_PATH)
        if candidate.get("status") == "active":
            episode = candidate
    while True:
        if colab_formal_shard_complete():
            log(f"colab_formal_shard_complete=true shard={COLAB_SHARD}")
            return
        missing_remote_work = local_sync_reports_missing_remote_work()
        if missing_remote_work:
            status = None
            log(f"remote_work_missing_from_sync=true shard={COLAB_SHARD}")
        elif synced_heartbeat_fresh():
            status = {"alive": True, "source": "recent_synced_heartbeat"}
        else:
            status = _inspect_colab()
        if missing_remote_work or (
            (not status or not status.get("alive")) and not synced_heartbeat_fresh()
        ):
            if episode is None:
                episode = load_or_start_recovery_episode()
            retry_until_success(
                lambda: rebuild_once_or_accept_durable_progress(
                    episode,
                    lambda: rebuild_colab_from_checkpoints(
                        force_fresh_session=missing_remote_work
                    ),
                )
            )
        else:
            log(f"colab_worker_already_healthy=true shard={COLAB_SHARD}")
        ensure_colab_monitors()
        if episode is not None and recovery_episode_advanced(episode):
            close_recovery_episode(episode)
            episode = None
            if return_after_episode_success:
                return
        sleep(60)


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
