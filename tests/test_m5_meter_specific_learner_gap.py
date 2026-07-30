from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.analyze_m5_meter_specific_learner_gap import (
    CONTEXTS,
    assert_same_identity,
    bootstrap_metrics,
    context_responses,
    learner_gaps,
    leave_one_building_unit,
    loo_unit_manifest,
    per_meter_metrics,
    _validation_provenance,
    validate_prediction_fields,
)


def fixture_base() -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    raw = 0
    for meter, name in enumerate(("electricity", "chilledwater", "steam", "hotwater")):
        for building in (meter * 10 + 1, meter * 10 + 2):
            for anomaly in (0, 1):
                rows.append(
                    {
                        "raw_index": raw,
                        "anomaly": anomaly,
                        "site_id": meter,
                        "building_id": building,
                        "meter": meter,
                        "meter_name": name,
                    }
                )
                raw += 1
    return pd.DataFrame(rows)


def fixture_scores(base: pd.DataFrame) -> dict[str, dict[int, np.ndarray]]:
    y = base["anomaly"].to_numpy()
    meter = base["meter"].to_numpy()
    result = {"tabpfn": {}, "trees": {}}
    for position, context in enumerate(CONTEXTS):
        # Electricity is intentionally a tree-favouring counterexample.
        tab = (
            np.where(y == 1, 0.75, 0.25).astype("float32")
            + meter * 0.002
            + position * 0.001
        )
        tree = np.where(y == 1, 0.70, 0.30).astype("float32") + meter * 0.002
        tab[meter == 0] = (
            np.array([0.80, 0.60, 0.20, 0.40], dtype="float32") + position * 0.001
        )
        tree[meter == 0] = np.where(y[meter == 0] == 1, 0.90, 0.10)
        tree[meter == 2] = np.array([0.80, 0.60, 0.20, 0.40], dtype="float32")
        result["tabpfn"][context] = tab
        result["trees"][context] = tree
    return result


class TestM5MeterSpecificLearnerGap(unittest.TestCase):
    def test_interruption_control_is_not_result_affecting_provenance(self) -> None:
        audit = pd.DataFrame(
            [{"model": "tabpfn", "context_rows": 5000, "sha256": "input"}]
        )
        common = {
            "seed": 20260730,
            "bootstrap_draws": 1,
            "loo_buildings": 3,
            "segment_draws": 1,
        }
        interrupted = SimpleNamespace(**common, validation_stop_after_units=12)
        resumed = SimpleNamespace(**common, validation_stop_after_units=None)
        self.assertEqual(
            _validation_provenance(interrupted, audit),
            _validation_provenance(resumed, audit),
        )

    def test_identity_drift_duplicate_indices_and_nonfinite_scores_hard_fail(
        self,
    ) -> None:
        reference = {
            "raw_index": np.array([1, 2]),
            "anomaly": np.array([0, 1]),
            "site_id": np.array([0, 0]),
            "building_id": np.array([4, 4]),
        }
        drifted = {**reference, "anomaly": np.array([1, 0])}
        with self.assertRaisesRegex(AssertionError, "anomaly identity drift"):
            assert_same_identity(reference, drifted, source="fixture")
        with self.assertRaisesRegex(AssertionError, "duplicate raw indices"):
            validate_prediction_fields(
                {**reference, "raw_index": np.array([1, 1])},
                np.array([0.1, 0.2]),
                source="fixture",
            )
        with self.assertRaisesRegex(AssertionError, "non-finite"):
            validate_prediction_fields(
                reference, np.array([0.1, np.nan]), source="fixture"
            )

    def test_paired_gaps_keep_electricity_counterexample(self) -> None:
        base = fixture_base()
        metrics, _ = per_meter_metrics(fixture_scores(base), base)
        gaps = learner_gaps(metrics)
        electricity = gaps.loc[
            (gaps.meter == "electricity")
            & (gaps.context_rows == 5_000)
            & (gaps.metric == "pr_auc"),
            "tabpfn_minus_trees",
        ].iloc[0]
        steam = gaps.loc[
            (gaps.meter == "steam")
            & (gaps.context_rows == 5_000)
            & (gaps.metric == "pr_auc"),
            "tabpfn_minus_trees",
        ].iloc[0]
        self.assertLess(electricity, 0)
        self.assertGreater(steam, 0)
        self.assertEqual(
            set(metrics.meter), {"electricity", "chilledwater", "steam", "hotwater"}
        )

    def test_context_slope_uses_log10_context_rows(self) -> None:
        rows = []
        for model in ("tabpfn", "trees"):
            for context in CONTEXTS:
                rows.append(
                    {
                        "model": model,
                        "context_rows": context,
                        "meter": "steam",
                        "pr_auc": np.log10(context) if model == "tabpfn" else 0.0,
                        "roc_auc": 0.0,
                        "anomaly_within_meter_percentile_rank": 0.0,
                    }
                )
        metrics = pd.DataFrame(rows)
        gaps = learner_gaps(metrics)
        slopes = context_responses(metrics, gaps)
        result = slopes.loc[
            (slopes.meter == "steam")
            & (slopes.quantity == "learner_gap")
            & (slopes.metric == "pr_auc")
            & (slopes.contrast == "5k_to_100k")
        ].iloc[0]
        self.assertAlmostEqual(result.slope_log10_context, 1.0, places=10)
        self.assertAlmostEqual(
            result.endpoint_change, np.log10(100_000) - np.log10(5_000), places=10
        )

    def test_building_bootstrap_is_matched_across_models_and_contexts(self) -> None:
        base = fixture_base()
        draws, invalid = bootstrap_metrics(fixture_scores(base), base, draws=3, seed=7)
        self.assertEqual(invalid["single_class"], 0)
        for draw in range(3):
            subset = draws.loc[
                (draws.draw == draw)
                & (draws.meter == "steam")
                & (draws.quantity == "tabpfn_minus_trees")
                & (draws.metric == "pr_auc")
            ]
            self.assertEqual(set(subset.context_rows), set(CONTEXTS))
        self.assertIn("learner_gap_change_5k_to_100k", set(draws.quantity))

    def test_invalid_single_class_bootstrap_draws_are_counted(self) -> None:
        base = fixture_base().loc[lambda frame: frame.meter == 0].copy()
        base.loc[:, "anomaly"] = 1
        scores = {
            model: {
                context: np.full(len(base), 0.5, dtype="float32")
                for context in CONTEXTS
            }
            for model in ("tabpfn", "trees")
        }
        draws, invalid = bootstrap_metrics(scores, base, draws=2, seed=1)
        self.assertTrue(draws.empty)
        self.assertEqual(invalid["single_class"], 2)

    def test_script_is_reader_only_and_uses_new_output_root(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "analyze_m5_meter_specific_learner_gap.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [
            node.names[0].name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import) and node.names
        ]
        self.assertNotIn("tabpfn", imports)
        self.assertIn("m5_meter_specific_learner_gap", source)
        self.assertNotIn('m5_hotwater_label_factorial" / "independent_query', source)

    def test_safe_mode_requires_bounded_validation_and_formal_is_gated(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "analyze_m5_meter_specific_learner_gap.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--validation-mode", source)
        self.assertIn("--loo-buildings", source)
        self.assertIn("--segment-draws", source)
        self.assertIn("AUTHORIZE E0 FORMAL RUN", source)
        self.assertNotIn("timeout=", source)

    def test_loo_checkpoint_payload_is_one_meter_by_one_building(self) -> None:
        base = fixture_base()
        scores = fixture_scores(base)
        metrics, ranks = per_meter_metrics(scores, base)
        gaps = learner_gaps(metrics)
        result = leave_one_building_unit(
            scores, base, ranks, gaps, code=0, building_id=1
        )
        self.assertEqual(set(result["building_id"]), {1})
        self.assertEqual(set(result["meter"]), {"electricity"})

    def test_validation_stop_is_validation_only_and_units_are_granular(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "analyze_m5_meter_specific_learner_gap.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--validation-stop-after-units", source)
        self.assertIn("EXPECTED_VALIDATION_INTERRUPTION", source)
        self.assertIn("__draw__", source)
        self.assertIn("__building__", source)

    def test_three_buildings_per_meter_produce_twelve_loo_units(self) -> None:
        base = fixture_base()
        additional = base.groupby("meter", group_keys=False).head(2).copy()
        additional["building_id"] = additional["meter"] * 10 + 3
        additional["raw_index"] += 1000
        expanded = pd.concat([base, additional], ignore_index=True)
        selected, units = loo_unit_manifest(expanded, loo_buildings=3)
        self.assertEqual(sum(len(values) for values in selected.values()), 12)
        self.assertEqual(len(units), 12)
        self.assertTrue(all("__building__" in unit for unit in units))


if __name__ == "__main__":
    unittest.main()
