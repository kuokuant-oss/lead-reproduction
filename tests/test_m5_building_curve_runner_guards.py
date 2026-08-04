from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_m5_building_curve_tabpfn_cell import main as tabpfn_main
from scripts.run_m5_building_curve_tabpfn_cell import parse_args as tabpfn_args
from scripts.run_m5_building_curve_tree_cell import main as tree_main
from scripts.run_m5_building_curve_tree_cell import parse_args as tree_args


def manifest(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "sampling_profile": "representative",
                "cells": {
                    "10": {
                        "available_buildings": list(range(0, 20, 2)),
                        "tree_fit_buildings": list(range(0, 16, 2)),
                        "tree_early_stop_buildings": [16, 18],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


class TestM5BuildingCurveRunnerGuards(unittest.TestCase):
    def test_plan_mode_never_loads_frame_or_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = manifest(Path(temporary) / "ladder.json")
            common = ["--building-manifest", str(path), "--building-budget", "10"]
            self.assertEqual(tree_main(common), 0)
            self.assertEqual(tabpfn_main(common), 0)

    def test_validation_requires_deterministic_caps(self) -> None:
        common = ["--building-manifest", "ladder.json", "--building-budget", "10"]
        with self.assertRaisesRegex(ValueError, "three positive row caps"):
            tree_args([*common, "--mode", "validation"])
        with self.assertRaisesRegex(ValueError, "context and holdout row caps"):
            tabpfn_args([*common, "--mode", "validation"])

    def test_formal_rejects_validation_caps(self) -> None:
        common = ["--building-manifest", "ladder.json", "--building-budget", "10"]
        with self.assertRaisesRegex(ValueError, "only allowed"):
            tree_args([*common, "--mode", "formal", "--max-fit-rows", "10"])
        with self.assertRaisesRegex(ValueError, "only allowed"):
            tabpfn_args([*common, "--mode", "formal", "--max-context-rows", "10"])


if __name__ == "__main__":
    unittest.main()
