from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.m5_building_curve_protocol import (
    add_cell_composition,
    add_proportional_row_quotas,
    build_building_profiles,
    build_nested_building_ladder,
    cell_indices,
    resolve_cell_indices,
    validate_ladder,
)
from scripts.m5_tree_early_stopping import (
    MODEL_ORDER,
    ensemble_probabilities,
    fit_early_stopped_models,
    refit_models_at_selected_iterations,
)


def synthetic_frame(buildings: int = 120, rows_per_building: int = 32) -> pd.DataFrame:
    ids = np.arange(0, buildings * 2, 2, dtype="int64")
    building = np.repeat(ids, rows_per_building)
    row = np.tile(np.arange(rows_per_building), buildings)
    meter = row % 4
    anomaly = ((row + building // 2) % 11 == 0).astype("int8")
    # Ensure both labels within every building and every 2-building ES prefix.
    anomaly[row == 0] = 1
    anomaly[row == 1] = 0
    return pd.DataFrame(
        {
            "building_id": building,
            "site_id": (building // 2) % 8,
            "meter": meter,
            "anomaly": anomaly,
            "primary_use": np.where((building // 2) % 2, "Office", "Education"),
            "x0": np.sin(row) + anomaly * 2,
            "x1": np.cos(row / 3) - anomaly,
        }
    )


class TestM5BuildingCurveProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = synthetic_frame()
        cls.profiles = build_building_profiles(cls.frame)

    def test_profiles_are_even_building_training_only(self) -> None:
        self.assertTrue(self.profiles["building_id"].mod(2).eq(0).all())
        self.assertEqual(len(self.profiles), self.frame["building_id"].nunique())
        self.assertEqual(
            set(self.profiles["anomaly_bin"].unique())
            <= {"zero", "positive_low", "positive_mid", "positive_high"},
            True,
        )

    def test_odd_building_source_is_rejected(self) -> None:
        broken = self.frame.copy()
        broken.loc[broken.index[:4], "building_id"] = 1
        with self.assertRaisesRegex(ValueError, "even-building"):
            build_building_profiles(broken)

    def test_ladder_is_deterministic_strictly_nested_and_roles_are_nested(self) -> None:
        first, manifest = build_nested_building_ladder(
            self.profiles, [10, 20, 50, 100], seed=42
        )
        second, second_manifest = build_nested_building_ladder(
            self.profiles, [10, 20, 50, 100], seed=42
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(manifest["cells"], second_manifest["cells"])
        validate_ladder(first, manifest)
        previous_available: set[int] = set()
        previous_fit: set[int] = set()
        previous_es: set[int] = set()
        for budget in (10, 20, 50, 100):
            cell = manifest["cells"][str(budget)]
            available = set(cell["available_buildings"])
            fit = set(cell["tree_fit_buildings"])
            early_stop = set(cell["tree_early_stop_buildings"])
            self.assertEqual(
                (len(fit), len(early_stop)), (budget * 4 // 5, budget // 5)
            )
            self.assertTrue(
                previous_available < available if previous_available else True
            )
            self.assertTrue(previous_fit < fit if previous_fit else True)
            self.assertTrue(previous_es < early_stop if previous_es else True)
            previous_available, previous_fit, previous_es = available, fit, early_stop

    def test_seed_changes_order_but_not_even_building_gate(self) -> None:
        first, _ = build_nested_building_ladder(self.profiles, [20], seed=42)
        second, _ = build_nested_building_ladder(self.profiles, [20], seed=43)
        self.assertFalse(np.array_equal(first["building_id"], second["building_id"]))
        self.assertTrue(first["building_id"].mod(2).eq(0).all())
        self.assertTrue(second["building_id"].mod(2).eq(0).all())

    def test_cell_rows_partition_available_and_composition_is_recorded(self) -> None:
        _, manifest = build_nested_building_ladder(self.profiles, [10, 20], seed=42)
        resolved = cell_indices(self.frame, manifest, 10)
        self.assertEqual(
            set(resolved["available_rows"]),
            set(resolved["tree_fit_rows"]) | set(resolved["tree_early_stop_rows"]),
        )
        enriched = add_cell_composition(self.frame, manifest)
        cell = enriched["cells"]["10"]
        self.assertEqual(cell["available_rows"], 10 * 32)
        self.assertEqual(set(cell["available_meter_counts"]), {"0", "1", "2", "3"})

    def test_requested_budget_must_preserve_exact_role_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiples"):
            build_nested_building_ladder(self.profiles, [12], seed=42)

    def test_average_cap_is_nested_deterministic_and_role_partitioned(self) -> None:
        _, manifest = build_nested_building_ladder(self.profiles, [10, 20], seed=42)
        manifest["row_policy"] = "average_building_cap"
        manifest["average_rows_per_building_limit"] = 20
        manifest["max_context_rows"] = 400
        manifest["row_selection_seed"] = 42
        manifest = add_proportional_row_quotas(self.frame, manifest)
        first = resolve_cell_indices(self.frame, manifest, 10)
        second = resolve_cell_indices(self.frame, manifest, 10)
        larger = resolve_cell_indices(self.frame, manifest, 20)
        np.testing.assert_array_equal(first["available_rows"], second["available_rows"])
        self.assertEqual(len(first["available_rows"]), 200)
        counts = self.frame.loc[first["available_rows"], "building_id"].value_counts()
        self.assertTrue(counts.eq(20).all())
        self.assertEqual(len(first["tree_fit_rows"]), 160)
        self.assertEqual(len(first["tree_early_stop_rows"]), 40)
        self.assertLess(
            set(first["available_rows"]),
            set(larger["available_rows"]),
        )


class TestM5TreeEarlyStopping(unittest.TestCase):
    def test_all_components_use_external_validation_and_record_best_iteration(
        self,
    ) -> None:
        rng = np.random.default_rng(7)
        x_fit = rng.normal(size=(160, 4)).astype("float32")
        y_fit = (x_fit[:, 0] + x_fit[:, 1] * 0.3 > 0).astype("int8")
        x_es = rng.normal(size=(80, 4)).astype("float32")
        y_es = (x_es[:, 0] + x_es[:, 1] * 0.3 > 0).astype("int8")
        ceilings = {
            "lightgbm": 8,
            "xgboost": 8,
            "catboost": 8,
            "hist_gradient_boosting": 8,
        }
        models, records, contract = fit_early_stopped_models(
            x_fit,
            y_fit,
            x_es,
            y_es,
            seed=42,
            patience=2,
            hist_patience=2,
            ceilings=ceilings,
        )
        self.assertEqual(tuple(models), MODEL_ORDER)
        for name in MODEL_ORDER:
            self.assertEqual(contract[name]["selection_metric"], "roc_auc")
            self.assertGreaterEqual(records[name]["best_iteration"], 1)
            self.assertLessEqual(records[name]["best_iteration"], ceilings[name])
            self.assertIn("history", records[name])
            self.assertIn(
                records[name]["stop_reason"], {"early_stopping", "iteration_ceiling"}
            )
        scores = ensemble_probabilities(models, x_es)
        self.assertEqual(set(scores), {*MODEL_ORDER, "ensemble"})
        self.assertTrue(np.isfinite(scores["ensemble"]).all())
        refit = refit_models_at_selected_iterations(
            np.vstack([x_fit, x_es]),
            np.concatenate([y_fit, y_es]),
            records,
            contract,
        )
        self.assertEqual(tuple(refit), MODEL_ORDER)

    def test_single_class_early_stop_is_rejected(self) -> None:
        x = np.zeros((8, 2), dtype="float32")
        with self.assertRaisesRegex(ValueError, "early-stop rows"):
            fit_early_stopped_models(
                x,
                np.tile([0, 1], 4),
                x,
                np.zeros(8),
                ceilings={name: 2 for name in MODEL_ORDER},
            )


if __name__ == "__main__":
    unittest.main()
