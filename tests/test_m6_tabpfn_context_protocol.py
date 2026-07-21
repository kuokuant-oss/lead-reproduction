from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PROTOCOL = SCRIPTS / "m6_tabpfn_context_protocol.py"
SUITE = SCRIPTS / "run_m6_tabpfn_context_suite.ps1"


def load_protocol():
    scripts_dir = str(SCRIPTS)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "m6_tabpfn_context_protocol", PROTOCOL
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {PROTOCOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestM6TabPFNContextProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_protocol()

    def test_context_is_deterministic_natural_prevalence_and_nested(self) -> None:
        raw_index = np.arange(50_000, dtype="int64") * 3 + 7
        labels = np.zeros(len(raw_index), dtype="int8")
        labels[::10] = 1
        small = self.m.nested_stratified_context_positions(
            raw_index,
            labels,
            context_rows=1_000,
            seed=42,
        )
        repeated = self.m.nested_stratified_context_positions(
            raw_index,
            labels,
            context_rows=1_000,
            seed=42,
        )
        large = self.m.nested_stratified_context_positions(
            raw_index,
            labels,
            context_rows=10_000,
            seed=42,
        )
        np.testing.assert_array_equal(small, repeated)
        self.assertEqual(len(small), 1_000)
        self.assertEqual(int(labels[small].sum()), 100)
        self.assertEqual(int(labels[large].sum()), 1_000)
        self.assertTrue(set(raw_index[small]) < set(raw_index[large]))

    def test_context_rejects_invalid_support(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two classes"):
            self.m.nested_stratified_context_positions(
                np.arange(10),
                np.zeros(10),
                context_rows=5,
                seed=42,
            )
        with self.assertRaisesRegex(ValueError, "exceeds source rows"):
            self.m.nested_stratified_context_positions(
                np.arange(10),
                np.arange(10) % 2,
                context_rows=11,
                seed=42,
            )

    def test_region_timer_records_wall_and_cpu_without_cuda(self) -> None:
        with self.m.RegionTimer(use_cuda_events=False) as timer:
            sum(value * value for value in range(10_000))
            time.sleep(0.01)
        result = timer.result
        self.assertGreaterEqual(result.wall_seconds, 0.01)
        self.assertGreaterEqual(result.cpu_user_seconds, 0.0)
        self.assertGreaterEqual(result.cpu_system_seconds, 0.0)
        self.assertIsNone(result.gpu_seconds)

    def test_resource_monitor_is_nonfatal_and_writes_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "monitor.csv"
            with self.m.ResourceMonitor(path, interval_seconds=0.01) as monitor:
                time.sleep(0.05)
            summary = monitor.summary()
            self.assertTrue(path.exists())
            self.assertGreaterEqual(summary["samples"], 1)
            self.assertGreater(summary["peak_process_rss_bytes"], 0)

    def test_powershell_suite_has_no_wall_timeout_or_silent_retry(self) -> None:
        source = SUITE.read_text(encoding="utf-8")
        self.assertIn("& $Python @Arguments", source)
        self.assertIn("timeout_seconds = $null", source)
        self.assertIn("retry_on_silence = $false", source)
        self.assertNotIn("Wait-Process -Timeout", source)
        self.assertNotIn("Start-Sleep", source)
        self.assertNotIn("$Process = Start-Process", source)


if __name__ == "__main__":
    unittest.main()
