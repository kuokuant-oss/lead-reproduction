from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "experiment_observability.py"
RUNNERS = (
    "run_m5_phaseC_tabpfn_spike.py",
    "run_m5_phaseD_foundation_vs_gbdt.py",
    "run_m6_phaseD_50_50_full_models.py",
    "run_m5_phaseD_deep_comparison.py",
)


def load_module():
    spec = importlib.util.spec_from_file_location("experiment_observability", MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExperimentObservability(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_module()

    def test_host_environment_records_minimum_cost_context(self) -> None:
        env = self.m.host_environment()
        self.assertIn("os", env)
        self.assertIn("cpu", env)
        self.assertIn("memory", env)
        self.assertIn("python", env)
        self.assertIn("dependency_versions", env)
        self.assertGreater(env["cpu"]["logical_cores"], 0)
        self.assertGreater(env["memory"]["total_bytes"], 0)

    def test_timing_protocol_explicitly_preserves_experiment_behavior(self) -> None:
        protocol = self.m.timing_protocol()
        self.assertEqual(protocol["clock"], "time.perf_counter")
        self.assertFalse(protocol["includes_json_write_in_elapsed"])
        self.assertFalse(protocol["cuda_synchronization_added"])
        self.assertIn("unchanged", protocol["note"])

    def test_every_related_runner_writes_observability_to_selected_output(self) -> None:
        for filename in RUNNERS:
            with self.subTest(runner=filename):
                source = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
                self.assertIn('"timing_protocol": timing_protocol()', source)
                self.assertIn('"environment": env', source)
                self.assertIn("write_json_with_provenance(", source)
                self.assertIn("args.out", source)


if __name__ == "__main__":
    unittest.main()
