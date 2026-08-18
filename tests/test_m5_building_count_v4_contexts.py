from __future__ import annotations

import unittest

import numpy as np

from scripts.prepare_m5_building_count_v4_fixed_10k import (
    BUDGETS,
    build_ladder,
    draw_context,
)


class TestM5BuildingCountV4Contexts(unittest.TestCase):
    def test_building_ladder_is_deterministic_unique_and_nested(self) -> None:
        candidates = np.arange(0, 1000, 2, dtype="int64")
        permutation, cells = build_ladder(candidates, building_seed=0)
        repeated, repeated_cells = build_ladder(candidates, building_seed=0)
        self.assertTrue(np.array_equal(permutation, repeated))
        previous = np.empty(0, dtype="int64")
        for budget in BUDGETS:
            selected = cells[budget]
            self.assertEqual(len(selected), budget)
            self.assertEqual(len(np.unique(selected)), budget)
            self.assertTrue(np.array_equal(selected, repeated_cells[budget]))
            self.assertTrue(np.array_equal(selected[: len(previous)], previous))
            previous = selected

    def test_row_draw_is_exact_balanced_unique_and_seeded(self) -> None:
        buildings = np.repeat(np.arange(0, 100, 2, dtype="int32"), 500)
        anomaly = np.tile(np.array([0, 1], dtype="int8"), len(buildings) // 2)
        selected_buildings = np.unique(buildings)
        first, support = draw_context(
            buildings,
            anomaly,
            selected_buildings,
            building_seed=0,
            row_seed=0,
            budget=50,
        )
        repeated, _ = draw_context(
            buildings,
            anomaly,
            selected_buildings,
            building_seed=0,
            row_seed=0,
            budget=50,
        )
        second, _ = draw_context(
            buildings,
            anomaly,
            selected_buildings,
            building_seed=0,
            row_seed=1,
            budget=50,
        )
        self.assertEqual(len(first), 10_000)
        self.assertEqual(len(np.unique(first)), 10_000)
        self.assertEqual(int(anomaly[first].sum()), 5_000)
        self.assertTrue(np.array_equal(first, repeated))
        self.assertFalse(np.array_equal(first, second))
        self.assertGreaterEqual(support["full_anomalies"], 5_000)
        self.assertGreaterEqual(support["full_normals"], 5_000)

    def test_short_class_support_fails_without_redraw(self) -> None:
        buildings = np.zeros(10_000, dtype="int32")
        anomaly = np.zeros(10_000, dtype="int8")
        anomaly[:4_999] = 1
        with self.assertRaisesRegex(ValueError, "insufficient unique class support"):
            draw_context(
                buildings,
                anomaly,
                np.array([0]),
                building_seed=0,
                row_seed=0,
                budget=50,
            )


if __name__ == "__main__":
    unittest.main()
