from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.m5_building_curve_protocol import stable_priority
from scripts.prepare_m5_building_count_v3_balanced_contexts import (
    BUDGETS,
    CONTEXT_ROWS,
    atomic_savez,
    build_nested_balanced_contexts,
    validate_source_manifest,
)


def synthetic_rows(rows_per_building: int = 120) -> tuple[np.ndarray, np.ndarray]:
    buildings = np.repeat(np.arange(0, 200, 2, dtype="int32"), rows_per_building)
    labels = np.tile(
        np.resize(np.asarray([0, 1], dtype="int8"), rows_per_building), 100
    )
    return buildings, labels


def nested_buildings() -> dict[int, np.ndarray]:
    all_buildings = np.arange(0, 200, 2, dtype="int64")
    return {budget: all_buildings[:budget].copy() for budget in BUDGETS}


def small_targets() -> dict[int, int]:
    return {10: 20, 20: 40, 50: 100, 100: 200}


class TestM5BuildingCountV3BalancedContexts(unittest.TestCase):
    def test_contexts_are_deterministic_balanced_unique_and_nested(self) -> None:
        building, labels = synthetic_rows()
        first, support = build_nested_balanced_contexts(
            building,
            labels,
            nested_buildings(),
            balance_seed=42,
            context_rows=small_targets(),
        )
        second, _ = build_nested_balanced_contexts(
            building,
            labels,
            nested_buildings(),
            balance_seed=42,
            context_rows=small_targets(),
        )
        previous = np.empty(0, dtype="int64")
        for budget in BUDGETS:
            rows = first[budget]
            np.testing.assert_array_equal(rows, second[budget])
            np.testing.assert_array_equal(rows[: len(previous)], previous)
            self.assertEqual(len(np.unique(rows)), len(rows))
            self.assertEqual(int(labels[rows].sum()), len(rows) // 2)
            self.assertEqual(support[budget]["selected_anomalies"], len(rows) // 2)
            self.assertTrue(np.isin(building[rows], nested_buildings()[budget]).all())
            previous = rows

    def test_seed_changes_the_random_draw(self) -> None:
        building, labels = synthetic_rows()
        first, _ = build_nested_balanced_contexts(
            building,
            labels,
            nested_buildings(),
            balance_seed=42,
            context_rows=small_targets(),
        )
        second, _ = build_nested_balanced_contexts(
            building,
            labels,
            nested_buildings(),
            balance_seed=43,
            context_rows=small_targets(),
        )
        self.assertFalse(np.array_equal(first[100], second[100]))

    def test_no_building_anchor_is_inserted(self) -> None:
        labels = np.asarray([0, 1] * 1_000, dtype="int8")
        building_order = np.arange(0, 200, 2, dtype="int64")
        building_order[[9, 99]] = building_order[[99, 9]]
        building = np.resize(building_order, len(labels)).astype("int32")
        building[building == 198] = 0
        priority = stable_priority(np.arange(len(labels)), seed=42)
        worst_normal = int(
            np.flatnonzero(labels == 0)[np.argmax(priority[labels == 0])]
        )
        worst_anomaly = int(
            np.flatnonzero(labels == 1)[np.argmax(priority[labels == 1])]
        )
        building[[worst_normal, worst_anomaly]] = 198
        sets = {budget: building_order[:budget].copy() for budget in BUDGETS}
        contexts, _ = build_nested_balanced_contexts(
            building,
            labels,
            sets,
            context_rows={10: 2, 20: 4, 50: 10, 100: 20},
        )
        self.assertNotIn(198, set(map(int, building[contexts[10]])))

    def test_insufficient_unique_class_support_fails_without_replacement(self) -> None:
        building, labels = synthetic_rows(rows_per_building=2)
        labels[:] = 0
        labels[0] = 1
        with self.assertRaisesRegex(ValueError, "insufficient unique class support"):
            build_nested_balanced_contexts(
                building,
                labels,
                nested_buildings(),
                context_rows=small_targets(),
            )

    def test_building_ladder_must_be_an_ordered_prefix(self) -> None:
        building, labels = synthetic_rows()
        broken = nested_buildings()
        broken[20] = broken[20].copy()
        broken[20][0], broken[20][10] = broken[20][10], broken[20][0]
        with self.assertRaisesRegex(ValueError, "strict prefixes"):
            build_nested_balanced_contexts(
                building,
                labels,
                broken,
                context_rows=small_targets(),
            )

    def test_source_manifest_rejects_odd_or_non_nested_buildings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "building_ladder_seed42.json"
            cells = {str(k): {"available_buildings": list(range(k))} for k in BUDGETS}
            manifest_path.write_text(
                json.dumps(
                    {"building_seed": 42, "budgets": list(BUDGETS), "cells": cells}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "odd holdout"):
                validate_source_manifest(manifest_path, expected_seed=42)

    def test_atomic_savez_replaces_complete_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.npz"
            atomic_savez(path, raw_index=np.asarray([1, 2], dtype="int64"))
            atomic_savez(path, raw_index=np.asarray([3, 4, 5], dtype="int64"))
            with np.load(path) as payload:
                np.testing.assert_array_equal(payload["raw_index"], [3, 4, 5])
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_frozen_production_targets_are_exact(self) -> None:
        self.assertEqual(
            CONTEXT_ROWS,
            {10: 5_000, 20: 10_000, 50: 25_000, 100: 50_000},
        )


if __name__ == "__main__":
    unittest.main()
