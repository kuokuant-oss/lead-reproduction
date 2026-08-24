from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.m5_building_count_v4_protocol import (
    BUDGETS,
    SCHEDULED_BUDGETS,
    k_major_contexts,
)
from scripts import launch_m5_building_count_v4 as launcher

scheduler = launcher.scheduler


class TestM5BuildingCountV4Scheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        launcher._configure()

    def test_order_completes_k400_seed0_then_k200_then_remaining_k400(self) -> None:
        contexts = k_major_contexts()
        self.assertEqual(len(contexts), 40)
        self.assertEqual(len(set(contexts)), 40)
        self.assertEqual(BUDGETS, (50, 100, 200, 300, 400))
        self.assertEqual(SCHEDULED_BUDGETS, (50, 100, 400, 200))
        self.assertEqual([item[2] for item in contexts[:10]], [50] * 10)
        self.assertEqual([item[2] for item in contexts[10:20]], [100] * 10)
        self.assertEqual(contexts[20:22], [(0, 0, 400), (0, 1, 400)])
        self.assertEqual([item[2] for item in contexts[22:32]], [200] * 10)
        self.assertEqual([item[2] for item in contexts[32:]], [400] * 8)
        self.assertEqual(
            [item[0] for item in contexts[32:]],
            [1, 1, 2, 2, 3, 3, 4, 4],
        )
        self.assertEqual(set(item[2] for item in contexts), set(SCHEDULED_BUDGETS))
        self.assertNotIn(300, [item[2] for item in contexts])

    def test_formal_plan_has_80_units_and_dedicated_adapters(self) -> None:
        def context(path: Path, budget: int) -> SimpleNamespace:
            name = path.stem
            building_seed = int(name.split("building_seed", 1)[1].split("_", 1)[0])
            row_seed = int(name.rsplit("row_seed", 1)[1])
            return SimpleNamespace(
                source_manifest_path=Path(
                    f"/audit/building_ladder_seed{building_seed}.json"
                ),
                building_seed=building_seed,
                row_seed=row_seed,
            )

        with patch.object(scheduler, "load_fixed_context", side_effect=context):
            units = scheduler.build_units(
                Path("/audit"),
                Path("/formal"),
                mode="formal",
                model_seed=42,
                model_path=Path("/model.ckpt"),
                validation_context_rows=200,
                validation_holdout_rows=200,
            )
        self.assertEqual(len(units), 80)
        self.assertEqual([u["identity"]["K"] for u in units[:20]], [50] * 20)
        self.assertEqual([u["identity"]["K"] for u in units[20:40]], [100] * 20)
        self.assertEqual([u["identity"]["K"] for u in units[40:44]], [400] * 4)
        self.assertEqual(
            [u["identity"]["building_seed"] for u in units[40:44]],
            [0] * 4,
        )
        self.assertEqual([u["identity"]["K"] for u in units[44:64]], [200] * 20)
        self.assertEqual([u["identity"]["K"] for u in units[64:]], [400] * 16)
        self.assertEqual(
            [
                (
                    unit["identity"]["building_seed"],
                    unit["identity"]["row_seed"],
                    unit["identity"]["K"],
                )
                for unit in units[::2]
            ],
            k_major_contexts(),
        )
        self.assertNotIn(300, [u["identity"]["K"] for u in units])
        for tree, tabpfn in zip(units[0::2], units[1::2], strict=True):
            self.assertIn("run_m5_building_count_v4_tree_cell.py", tree["command"][1])
            self.assertIn(
                "run_m5_building_count_v4_tabpfn_cell.py", tabpfn["command"][1]
            )
            self.assertIn("--balanced-context-manifest", tree["command"])
            self.assertNotIn("--experiment-version", tree["command"])
            self.assertIn("--experiment-version", tabpfn["command"])

    def test_validation_is_bounded_to_two_extreme_contexts(self) -> None:
        contexts = scheduler.selected_contexts("validation")
        self.assertEqual(contexts, [(0, 0, 50), (4, 1, 400)])

    def test_default_mode_is_non_launching_plan(self) -> None:
        self.assertEqual(scheduler.parse_args([]).mode, "plan")


if __name__ == "__main__":
    unittest.main()
