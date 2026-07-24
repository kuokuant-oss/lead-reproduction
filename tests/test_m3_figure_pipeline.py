from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestM3FigureObserver(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_module(
            "run_m3_figure_observations",
            "run_m3_figure_observations.py",
        )

    def test_frozen_model_contract_matches_m3_4_declarations(self) -> None:
        contract = self.m.frozen_model_contract(seed=42)
        self.assertEqual(
            contract,
            {
                "lightgbm": {
                    "class": "LGBMClassifier",
                    "params": {
                        "n_estimators": 100,
                        "verbose": -1,
                        "random_state": 42,
                    },
                },
                "xgboost": {
                    "class": "XGBClassifier",
                    "params": {
                        "n_estimators": 100,
                        "eval_metric": "logloss",
                        "verbosity": 0,
                        "random_state": 42,
                    },
                },
                "catboost": {
                    "class": "CatBoostClassifier",
                    "params": {
                        "iterations": 1000,
                        "verbose": False,
                        "random_seed": 42,
                        "allow_writing_files": False,
                    },
                },
                "hist_gradient_boosting": {
                    "class": "HistGradientBoostingClassifier",
                    "params": {"max_iter": 100, "random_state": 42},
                },
            },
        )
        self.assertEqual(self.m.VALUE_CHANGE_REGIME, "timestamp_merge")
        self.assertEqual(self.m.EXPECTED_SPLIT["train_buildings"], 725)
        self.assertEqual(self.m.EXPECTED_SPLIT["validation_buildings"], 724)

    def test_confusion_orientation_is_ground_truth_by_prediction(self) -> None:
        summary = self.m.evaluation_summary(
            np.array([0, 0, 1, 1]),
            np.array([0.1, 0.8, 0.2, 0.9]),
        )
        self.assertEqual(
            summary["threshold_0_5"]["confusion_matrix"],
            {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
        )

    def test_stratified_sample_is_deterministic_and_keeps_both_classes(self) -> None:
        y = np.array([0] * 90 + [1] * 10)
        first = self.m.stratified_sample_indices(y, sample_size=20, seed=42)
        second = self.m.stratified_sample_indices(y, sample_size=20, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(int(y[first].sum()), 2)

    def test_feature_screening_does_not_claim_removal(self) -> None:
        rows = [
            {
                "feature": "lag_value_diff_1",
                "roc_auc_decrease_mean": -0.0001,
                "roc_auc_decrease_std": 0.0002,
                "pr_auc_decrease_mean": 0.0,
                "pr_auc_decrease_std": 0.0002,
                "repeats": 3,
            }
        ]
        screened = self.m.screen_importance_candidates(rows)
        self.assertEqual(screened[0]["screen_label"], "redundant_or_replaceable")
        self.assertNotIn("remove", screened[0]["screen_label"])
        self.assertEqual(
            self.m.correlated_group_for_feature(
                "lag_value_diff_1",
                [
                    "lag_value_diff_0",
                    "lag_value_diff_1",
                    "lag_value_diff_2",
                    "lag_value_ratio_1",
                ],
            ),
            [
                "lag_value_diff_0",
                "lag_value_diff_1",
                "lag_value_diff_2",
                "lag_value_ratio_1",
            ],
        )

    def test_constantized_prediction_does_not_mutate_read_only_input(self) -> None:
        class SumProbabilityModel:
            def predict_proba(self, values):
                positive = np.clip(values.sum(axis=1) / 10, 0, 1)
                return np.column_stack([1 - positive, positive])

        values = np.arange(12, dtype=float).reshape(4, 3)
        original = values.copy()
        values.setflags(write=False)
        prediction = self.m.predict_probability_with_constant_columns(
            "lightgbm",
            SumProbabilityModel(),
            values,
            [1],
            batch_size=2,
        )
        np.testing.assert_array_equal(values, original)
        np.testing.assert_allclose(prediction, [0.2, 0.8, 1.0, 1.0])


class TestM3FigureRendering(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_module("plot_m3_figures", "plot_m3_figures.py")

    @staticmethod
    def synthetic_observation() -> dict:
        x = np.linspace(0, 1, 30).tolist()
        roc = {"x": x, "y": np.sqrt(np.linspace(0, 1, 30)).tolist()}
        pr = {"x": x, "y": (1 - 0.35 * np.linspace(0, 1, 30)).tolist()}
        keys = [
            *TestM3FigureRendering.m.MODEL_ORDER,
            TestM3FigureRendering.m.FEATURE_BASELINE_KEY,
        ]
        metrics = {
            key: {
                "roc_auc": 0.99,
                "pr_auc": 0.81,
                "threshold_0_5": {
                    "precision": 0.8,
                    "recall": 0.7,
                    "f1": 0.75,
                    "confusion_matrix": {
                        "tn": 800,
                        "fp": 20,
                        "fn": 10,
                        "tp": 70,
                    },
                },
            }
            for key in keys
        }
        curves = {key: {"roc": roc, "precision_recall": pr} for key in keys}
        importance_rows = [
            {
                "feature": f"feature_{i}",
                "roc_auc_decrease_mean": 0.02 / (i + 1),
                "roc_auc_decrease_std": 0.0001,
                "pr_auc_decrease_mean": 0.01 / (i + 1),
                "pr_auc_decrease_std": 0.0001,
                "repeats": 3,
            }
            for i in range(12)
        ]
        consensus = [
            {
                "feature": row["feature"],
                "models_in_top10": 4,
                "mean_rank": i + 1,
                "rank_std": 0.0,
                "mean_roc_auc_decrease": row["roc_auc_decrease_mean"],
            }
            for i, row in enumerate(importance_rows)
        ]
        return {
            "metrics": metrics,
            "curves": curves,
            "value_change_illustration": {
                "building_id": 12,
                "meter": 0,
                "anchor_timestamp": "2016-01-02T00:00:00",
                "anchor_difference": 15.0,
                "anchor_ratio": 4.0,
                "points": [
                    {
                        "timestamp": f"2016-01-01T{i:02d}:00:00",
                        "meter_reading": float(i + (15 if i == 12 else 0)),
                        "anomaly": int(i == 12),
                        "difference": float(15 if i == 12 else 1),
                        "ratio": float(4 if i == 12 else 1.05),
                    }
                    for i in range(24)
                ],
            },
            "permutation_importance": {
                "models": {
                    key: importance_rows
                    for key in TestM3FigureRendering.m.MODEL_ORDER[:4]
                },
                "ensemble": importance_rows,
                "consensus": consensus,
            },
        }

    def test_all_thirteen_renderers_create_readable_pngs(self) -> None:
        data = self.synthetic_observation()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            paths = [output / f"figure_{index}.png" for index in range(13)]
            self.m.render_confusion(data, paths[0])
            self.m.render_value_change(data, paths[1])
            self.m.render_workflow(paths[2])
            curve_specs = (
                ("models", "precision_recall"),
                ("models", "roc"),
                ("feature_engineering", "precision_recall"),
                ("feature_engineering", "roc"),
            )
            for path, (comparison, curve_type) in zip(
                paths[3:7], curve_specs, strict=True
            ):
                self.m.render_discrimination_curve(
                    data,
                    path,
                    comparison=comparison,
                    curve_type=curve_type,
                )
            importance_sources = (
                "lightgbm",
                "xgboost",
                "catboost",
                "hist_gradient_boosting",
                "ensemble",
                "consensus",
            )
            for path, source in zip(paths[7:], importance_sources, strict=True):
                self.m.render_importance_figure(data, path, source=source)
            for path in paths:
                with self.subTest(path=path.name):
                    self.assertGreater(path.stat().st_size, 10_000)
                    with Image.open(path) as image:
                        self.assertEqual(image.format, "PNG")
                        self.assertGreaterEqual(image.width, 1000)

    def test_importance_order_uses_the_value_drawn_in_each_figure(self) -> None:
        rows = [
            {"feature": "consensus_first", "roc_auc_decrease_mean": 0.01},
            {"feature": "model_first", "roc_auc_decrease_mean": 0.03},
            {"feature": "middle", "roc_auc_decrease_mean": 0.02},
        ]
        self.assertEqual(
            self.m._importance_feature_order(rows, "roc_auc_decrease_mean"),
            ["model_first", "middle", "consensus_first"],
        )

    def test_feature_comparison_uses_tree_ensemble_predictions(self) -> None:
        data = {"metrics": {}, "curves": {}, "split": {"validation_rows": 4}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.npz"
            np.savez(
                path,
                anomaly=np.array([0, 0, 1, 1]),
                ensemble=np.array([0.1, 0.2, 0.8, 0.9]),
            )
            self.m.add_17_feature_ensemble_comparison(data, path)

        baseline = self.m.FEATURE_BASELINE_KEY
        self.assertEqual(data["metrics"][baseline]["roc_auc"], 1.0)
        self.assertEqual(data["metrics"][baseline]["pr_auc"], 1.0)
        self.assertIn("roc", data["curves"][baseline])
        self.assertIn("precision_recall", data["curves"][baseline])

    def test_importance_panel_has_no_error_bars(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.container import ErrorbarContainer

        fig, ax = plt.subplots()
        rows = [
            {
                "feature": "feature_a",
                "roc_auc_decrease_mean": 0.03,
                "roc_auc_decrease_std": 0.02,
            }
        ]
        self.m._importance_panel(
            ax,
            rows,
            title="",
            value_key="roc_auc_decrease_mean",
            color="#000000",
            feature_order=["feature_a"],
            x_limit=0.05,
        )
        self.assertFalse(
            any(isinstance(container, ErrorbarContainer) for container in ax.containers)
        )
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
