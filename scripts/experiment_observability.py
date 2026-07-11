"""Read-only hardware and timing provenance for experiment runners."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from importlib import metadata
from typing import Any


DEPENDENCIES = (
    "catboost",
    "lightgbm",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "tabpfn",
    "torch",
    "xgboost",
)


def _cpu_name() -> str | None:
    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER")
    return name.strip() if name else None


def _total_ram_bytes() -> int | None:
    if platform.system() == "Windows":

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)
    except (AttributeError, OSError, ValueError):
        return None


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _nvidia_driver_versions() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def host_environment() -> dict[str, Any]:
    """Return stable, read-only host facts needed to interpret runtime costs."""
    total_ram = _total_ram_bytes()
    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": {
            "name": _cpu_name(),
            "logical_cores": os.cpu_count(),
            "processor_architecture": os.environ.get("PROCESSOR_ARCHITECTURE"),
        },
        "memory": {
            "total_bytes": total_ram,
            "total_gib": round(total_ram / (1024**3), 3) if total_ram else None,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "dependency_versions": dependency_versions(),
        "execution_controls": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "nvidia_driver_versions": _nvidia_driver_versions(),
    }


def timing_protocol() -> dict[str, Any]:
    """Describe timing semantics without changing experiment execution."""
    return {
        "clock": "time.perf_counter",
        "unit": "seconds",
        "elapsed_seconds_scope": "runner main after argument parsing through result assembly before JSON write",
        "fit_predict_seconds_scope": "model-local initialization/fit/predict region as recorded by each runner",
        "includes_data_loading_and_feature_generation_in_elapsed": True,
        "includes_json_write_in_elapsed": False,
        "cuda_events_used": False,
        "cuda_synchronization_added": False,
        "note": "Observability only: model inputs, seeds, hyperparameters, execution order, and predictions are unchanged.",
    }
