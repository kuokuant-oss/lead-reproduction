from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PLOTTER = SCRIPTS / "plot_m6_seen_vs_unseen.py"


def load_plotter():
    scripts_dir = str(SCRIPTS)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "plot_m6_seen_vs_unseen_for_test", PLOTTER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {PLOTTER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTabPFNOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_plotter()

    @staticmethod
    def payload(*, status: str = "completed", rows: int = 553_357) -> dict:
        return {
            "status": status,
            "target_site": 1,
            "context": {"context_rows": 10_000},
            "query": {"rows": rows, "anomalies": 77_779},
            "metrics": {
                "tabpfn": {
                    "pr_auc": 0.92,
                    "threshold_0_5": {
                        "precision": 0.91,
                        "recall": 0.80,
                        "f1": 0.85,
                    },
                }
            },
        }

    def test_only_completed_full_site_result_is_observed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = self.m.PROC
            self.m.PROC = Path(directory)
            try:
                stem = "m6_tabpfn_b1_a1_meters50_seed42_context10000"
                (self.m.PROC / f"{stem}.json").write_text(
                    json.dumps(self.payload()), encoding="utf-8"
                )
                partial = "m6_tabpfn_b1_a1_meters100_seed42_context10000"
                (self.m.PROC / f"{partial}.json").write_text(
                    json.dumps(self.payload(status="running")), encoding="utf-8"
                )

                observed = self.m.observe_tabpfn()

                rows = observed["10000"]["1"]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["budget"], "50")
                self.assertEqual(rows[0]["n_rows"], 553_357)
            finally:
                self.m.PROC = previous

    def test_completed_result_must_cover_exact_target_site(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = self.m.PROC
            self.m.PROC = Path(directory)
            try:
                stem = "m6_tabpfn_b1_a1_meters50_seed42_context10000"
                (self.m.PROC / f"{stem}.json").write_text(
                    json.dumps(self.payload(rows=553_356)), encoding="utf-8"
                )
                with self.assertRaisesRegex(SystemExit, "full-site support drifted"):
                    self.m.observe_tabpfn()
            finally:
                self.m.PROC = previous

    def test_tabpfn_seed_is_not_plottable_until_all_budgets_complete(self) -> None:
        incomplete = [
            {"seed": 42, "budget": budget}
            for budget in self.m.BUDGETS
            if budget != "all"
        ]
        self.assertEqual(self.m._complete_tabpfn_seeds(incomplete), ())

        complete = incomplete + [{"seed": 42, "budget": "all"}]
        self.assertEqual(self.m._complete_tabpfn_seeds(complete), (42,))


if __name__ == "__main__":
    unittest.main()
