"""Resource monitoring primitives for isolated experiment workers.

This module intentionally never imports torch or TabPFN, so a controller can
use it without creating a CUDA context.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import psutil


MIB = 1024 * 1024


@dataclass(frozen=True)
class ResourceLimits:
    gpu_soft_mib: float | None
    gpu_hard_mib: float | None
    ram_soft_mib: float
    ram_hard_mib: float
    soft_limit_consecutive_polls: int = 4
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        for soft, hard, name in (
            (self.gpu_soft_mib, self.gpu_hard_mib, "GPU"),
            (self.ram_soft_mib, self.ram_hard_mib, "RAM"),
        ):
            if soft is not None and hard is not None and not 0 < soft < hard:
                raise ValueError(f"{name} soft limit must be below hard limit")
        if self.soft_limit_consecutive_polls < 1:
            raise ValueError("soft-limit consecutive polls must be positive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("worker timeout must be positive")


@dataclass(frozen=True)
class ResourceSample:
    timestamp: float
    worker_rss_mib: float
    system_used_mib: float
    system_available_mib: float
    system_total_mib: float
    gpu_used_mib: float | None
    gpu_total_mib: float | None
    monitoring_scope: str


@dataclass(frozen=True)
class LimitDecision:
    action: str
    reason: str | None = None


def atomic_write_json(path: Path, payload: Any) -> None:
    """Durably replace a JSON document without exposing partial content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def resolve_limits(
    *,
    gpu_total_mib: float | None,
    ram_total_mib: float,
    gpu_soft_fraction: float,
    gpu_hard_fraction: float,
    ram_soft_fraction: float,
    ram_hard_fraction: float,
    gpu_soft_mib: float | None,
    gpu_hard_mib: float | None,
    ram_soft_mib: float | None,
    ram_hard_mib: float | None,
    soft_limit_consecutive_polls: int,
    timeout_seconds: float | None,
) -> ResourceLimits:
    """Resolve limits with explicit MiB values taking precedence."""
    for fraction, name in (
        (gpu_soft_fraction, "gpu soft"),
        (gpu_hard_fraction, "gpu hard"),
        (ram_soft_fraction, "ram soft"),
        (ram_hard_fraction, "ram hard"),
    ):
        if not 0 < fraction <= 1:
            raise ValueError(f"{name} fraction must be in (0, 1]")
    return ResourceLimits(
        gpu_soft_mib=(
            gpu_soft_mib
            if gpu_soft_mib is not None
            else gpu_total_mib * gpu_soft_fraction
            if gpu_total_mib is not None
            else None
        ),
        gpu_hard_mib=(
            gpu_hard_mib
            if gpu_hard_mib is not None
            else gpu_total_mib * gpu_hard_fraction
            if gpu_total_mib is not None
            else None
        ),
        ram_soft_mib=(
            ram_soft_mib
            if ram_soft_mib is not None
            else ram_total_mib * ram_soft_fraction
        ),
        ram_hard_mib=(
            ram_hard_mib
            if ram_hard_mib is not None
            else ram_total_mib * ram_hard_fraction
        ),
        soft_limit_consecutive_polls=soft_limit_consecutive_polls,
        timeout_seconds=timeout_seconds,
    )


class LimitTracker:
    """Turn resource samples into deterministic continue/stop decisions."""

    def __init__(self, limits: ResourceLimits) -> None:
        self.limits = limits
        self.soft_polls = 0

    def observe(
        self, sample: ResourceSample, *, elapsed_seconds: float
    ) -> LimitDecision:
        if (
            self.limits.timeout_seconds is not None
            and elapsed_seconds >= self.limits.timeout_seconds
        ):
            return LimitDecision("terminate", "worker timeout")
        if (
            self.limits.gpu_hard_mib is not None
            and sample.gpu_used_mib is not None
            and sample.gpu_used_mib >= self.limits.gpu_hard_mib
        ):
            return LimitDecision("terminate", "GPU hard limit exceeded")
        if sample.system_used_mib >= self.limits.ram_hard_mib:
            return LimitDecision("terminate", "RAM hard limit exceeded")

        gpu_soft = (
            self.limits.gpu_soft_mib is not None
            and sample.gpu_used_mib is not None
            and sample.gpu_used_mib >= self.limits.gpu_soft_mib
        )
        ram_soft = sample.system_used_mib >= self.limits.ram_soft_mib
        if gpu_soft or ram_soft:
            self.soft_polls += 1
            if self.soft_polls >= self.limits.soft_limit_consecutive_polls:
                names = "+".join(
                    name
                    for name, exceeded in (("GPU", gpu_soft), ("RAM", ram_soft))
                    if exceeded
                )
                return LimitDecision("request_stop", f"{names} soft limit exceeded")
        else:
            self.soft_polls = 0
        return LimitDecision("continue")


def _run_nvidia_smi(
    arguments: list[str],
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    return command_runner(
        ["nvidia-smi", *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )


def query_gpu_memory(
    worker_pid: int,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Read worker GPU usage, falling back to whole-device WDDM usage."""
    process_query = _run_nvidia_smi(
        [
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        command_runner=command_runner,
    )
    if process_query.returncode == 0:
        matching: list[float] = []
        for line in process_query.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            try:
                if int(parts[0]) == worker_pid:
                    matching.append(float(parts[1]))
            except ValueError:
                continue
        if matching:
            return {
                "used_mib": float(sum(matching)),
                "total_mib": _query_gpu_total(command_runner=command_runner),
                "monitoring_scope": "process",
                "available": True,
            }

    device_query = _run_nvidia_smi(
        ["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        command_runner=command_runner,
    )
    if device_query.returncode == 0:
        parts = [
            part.strip()
            for part in next(iter(device_query.stdout.splitlines()), "").split(",")
        ]
        if len(parts) == 2:
            try:
                return {
                    "used_mib": float(parts[0]),
                    "total_mib": float(parts[1]),
                    "monitoring_scope": "device_total",
                    "available": True,
                }
            except ValueError:
                pass
    return {
        "used_mib": None,
        "total_mib": None,
        "monitoring_scope": "unavailable",
        "available": False,
    }


def _query_gpu_total(
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> float | None:
    result = _run_nvidia_smi(
        ["--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        command_runner=command_runner,
    )
    if result.returncode != 0:
        return None
    try:
        return float(next(iter(result.stdout.splitlines())).strip())
    except (StopIteration, ValueError):
        return None


def sample_resources(
    worker_pid: int,
    *,
    process_factory: Callable[[int], Any] = psutil.Process,
    virtual_memory: Callable[[], Any] = psutil.virtual_memory,
    gpu_query: Callable[[int], dict[str, Any]] = query_gpu_memory,
    clock: Callable[[], float] = time.time,
) -> ResourceSample:
    process = process_factory(worker_pid)
    memory = virtual_memory()
    gpu = gpu_query(worker_pid)
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except psutil.Error:
        pass
    worker_rss = 0
    for item in processes:
        try:
            worker_rss += int(item.memory_info().rss)
        except psutil.Error:
            continue
    return ResourceSample(
        timestamp=float(clock()),
        worker_rss_mib=float(worker_rss / MIB),
        system_used_mib=float(memory.used / MIB),
        system_available_mib=float(memory.available / MIB),
        system_total_mib=float(memory.total / MIB),
        gpu_used_mib=gpu.get("used_mib"),
        gpu_total_mib=gpu.get("total_mib"),
        monitoring_scope=str(gpu.get("monitoring_scope", "unavailable")),
    )


def sample_dict(sample: ResourceSample) -> dict[str, Any]:
    return asdict(sample)


def pid_exists(pid: int, *, exists: Callable[[int], bool] = psutil.pid_exists) -> bool:
    return pid > 0 and bool(exists(pid))


def terminate_process_tree(
    pid: int,
    *,
    grace_seconds: float,
    process_factory: Callable[[int], Any] = psutil.Process,
    wait_procs: Callable[..., Any] = psutil.wait_procs,
) -> dict[str, Any]:
    """Terminate descendants and parent, then kill surviving processes."""
    try:
        parent = process_factory(pid)
    except psutil.Error:
        return {"terminated": [], "killed": [], "already_gone": True}
    processes = [*parent.children(recursive=True), parent]
    terminated: list[int] = []
    for process in processes:
        try:
            process.terminate()
            terminated.append(int(process.pid))
        except psutil.Error:
            continue
    _, alive = wait_procs(processes, timeout=max(0.0, grace_seconds))
    killed: list[int] = []
    for process in alive:
        try:
            process.kill()
            killed.append(int(process.pid))
        except psutil.Error:
            continue
    if alive:
        wait_procs(alive, timeout=max(0.0, grace_seconds))
    return {"terminated": terminated, "killed": killed, "already_gone": False}
