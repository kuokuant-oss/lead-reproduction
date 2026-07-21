"""Pure sampling and observability helpers for M6 TabPFN context curves."""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Return a deterministic, vectorised 64-bit permutation."""
    z = np.asarray(values, dtype="uint64").copy()
    with np.errstate(over="ignore"):
        z = (z + np.uint64(0x9E3779B97F4A7C15)) & UINT64_MASK
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z &= UINT64_MASK
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z &= UINT64_MASK
        return z ^ (z >> np.uint64(31))


def stable_row_priority(raw_index: np.ndarray, *, seed: int) -> np.ndarray:
    """Assign a seed-specific priority without target labels or mutable RNG state."""
    values = np.asarray(raw_index, dtype="uint64")
    seed_bits = np.uint64(int(seed) & int(UINT64_MASK))
    return _splitmix64(values ^ _splitmix64(np.asarray([seed_bits]))[0])


def nested_stratified_context_positions(
    raw_index: np.ndarray,
    labels: np.ndarray,
    *,
    context_rows: int,
    seed: int,
) -> np.ndarray:
    """Choose a deterministic natural-prevalence context.

    Rows are ranked once per class using stable raw-row identities. Increasing
    ``context_rows`` therefore only appends rows within each class: a 10k
    context is an exact row-identity subset of the corresponding 100k context.
    """
    raw_index = np.asarray(raw_index, dtype="int64")
    labels = np.asarray(labels)
    if len(raw_index) != len(labels):
        raise ValueError("raw_index and labels must have equal length")
    if context_rows <= 0:
        raise ValueError("context_rows must be positive")
    if context_rows > len(raw_index):
        raise ValueError(
            f"context_rows {context_rows:,} exceeds source rows {len(raw_index):,}"
        )
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError("TabPFN context requires at least two classes")

    priorities = stable_row_priority(raw_index, seed=seed)
    target_counts = np.floor(context_rows * counts / len(labels)).astype("int64")
    remainder = int(context_rows - target_counts.sum())
    fractional = context_rows * counts / len(labels) - target_counts
    for class_pos in np.argsort(-fractional, kind="stable")[:remainder]:
        target_counts[class_pos] += 1

    chosen: list[np.ndarray] = []
    for class_value, target_count in zip(classes, target_counts, strict=True):
        candidates = np.flatnonzero(labels == class_value)
        if target_count <= 0 or target_count > len(candidates):
            raise ValueError("class allocation cannot satisfy requested context")
        order = np.lexsort((raw_index[candidates], priorities[candidates]))
        chosen.append(candidates[order[:target_count]])

    positions = np.concatenate(chosen)
    order = np.lexsort((raw_index[positions], priorities[positions]))
    return positions[order].astype("int64", copy=False)


@dataclass(frozen=True)
class TimedRegion:
    wall_seconds: float
    cpu_user_seconds: float
    cpu_system_seconds: float
    gpu_seconds: float | None


class RegionTimer:
    """Measure wall, process CPU, and synchronized CUDA-event elapsed time."""

    def __init__(self, *, use_cuda_events: bool = True) -> None:
        self.use_cuda_events = use_cuda_events
        self._process: Any = None
        self._torch: Any = None
        self._start_event: Any = None
        self._end_event: Any = None

    def __enter__(self) -> "RegionTimer":
        import psutil

        self._process = psutil.Process()
        cpu = self._process.cpu_times()
        self._cpu_user = float(cpu.user)
        self._cpu_system = float(cpu.system)
        self._wall = time.perf_counter()
        if self.use_cuda_events:
            try:
                import torch

                if torch.cuda.is_available():
                    self._torch = torch
                    self._start_event = torch.cuda.Event(enable_timing=True)
                    self._end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    self._start_event.record()
            except (ImportError, RuntimeError):
                self._torch = None
        return self

    def __exit__(self, *_: object) -> None:
        gpu_seconds: float | None = None
        if self._torch is not None:
            self._end_event.record()
            self._torch.cuda.synchronize()
            gpu_seconds = float(self._start_event.elapsed_time(self._end_event) / 1000)
        cpu = self._process.cpu_times()
        self.result = TimedRegion(
            wall_seconds=float(time.perf_counter() - self._wall),
            cpu_user_seconds=float(cpu.user - self._cpu_user),
            cpu_system_seconds=float(cpu.system - self._cpu_system),
            gpu_seconds=gpu_seconds,
        )


class ResourceMonitor:
    """Sample process CPU/RAM and NVIDIA GPU utilisation to a CSV sidecar."""

    FIELDS = (
        "timestamp_unix",
        "elapsed_seconds",
        "process_cpu_percent",
        "process_rss_bytes",
        "gpu_util_percent",
        "gpu_memory_used_bytes",
    )

    def __init__(self, path: Path, *, interval_seconds: float = 1.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("monitor interval must be positive")
        self.path = Path(path)
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.samples = 0
        self.peak_gpu_memory_bytes: int | None = None
        self.peak_process_rss_bytes = 0

    def _gpu_reader(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            def read() -> tuple[int, int]:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return int(util.gpu), int(memory.used)

            return read
        except Exception as error:  # noqa: BLE001 - monitoring must be non-fatal
            self.error = f"GPU monitoring unavailable: {type(error).__name__}: {error}"
            return None

    def _run(self) -> None:
        import psutil

        self.path.parent.mkdir(parents=True, exist_ok=True)
        process = psutil.Process()
        process.cpu_percent(None)
        gpu_read = self._gpu_reader()
        started = time.perf_counter()
        try:
            with self.path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.FIELDS)
                if stream.tell() == 0:
                    writer.writeheader()
                while not self._stop.is_set():
                    gpu_util = None
                    gpu_memory = None
                    if gpu_read is not None:
                        try:
                            gpu_util, gpu_memory = gpu_read()
                        except Exception as error:  # noqa: BLE001
                            self.error = (
                                "GPU monitoring failed: "
                                f"{type(error).__name__}: {error}"
                            )
                            gpu_read = None
                    rss = int(process.memory_info().rss)
                    self.peak_process_rss_bytes = max(self.peak_process_rss_bytes, rss)
                    if gpu_memory is not None:
                        self.peak_gpu_memory_bytes = max(
                            self.peak_gpu_memory_bytes or 0, gpu_memory
                        )
                    writer.writerow(
                        {
                            "timestamp_unix": time.time(),
                            "elapsed_seconds": time.perf_counter() - started,
                            "process_cpu_percent": process.cpu_percent(None),
                            "process_rss_bytes": rss,
                            "gpu_util_percent": gpu_util,
                            "gpu_memory_used_bytes": gpu_memory,
                        }
                    )
                    stream.flush()
                    self.samples += 1
                    self._stop.wait(self.interval_seconds)
        except Exception as error:  # noqa: BLE001
            self.error = f"resource monitor failed: {type(error).__name__}: {error}"

    def __enter__(self) -> "ResourceMonitor":
        self._thread = threading.Thread(
            target=self._run,
            name="m6-tabpfn-resource-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, 2 * self.interval_seconds))

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "interval_seconds": self.interval_seconds,
            "samples": int(self.samples),
            "peak_process_rss_bytes": int(self.peak_process_rss_bytes),
            "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes,
            "error": self.error,
        }


def timing_dict(timer: RegionTimer) -> dict[str, float | None]:
    return asdict(timer.result)
