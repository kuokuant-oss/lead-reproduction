from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m3_figure_observations import frozen_model_contract
from scripts.run_m5_building_count_v2 import (
    EXPERIMENT_VERSION,
    build_units,
    main as v2_main,
    matched_context_gate,
)
from scripts.run_m5_building_count_v2_tree_cell import parse_args as v2_tree_args
from scripts.run_m5_building_curve_tabpfn_cell import parse_args as tabpfn_args
from scripts.update_m5_building_count_v2_report import main as report_main


def audit_summary() -> dict[str, object]:
    return {
        "status": "audit_passed_ready_for_model_evaluation",
        "sampling_profile": "site_stratified_random",
        "building_seeds": [42, 43, 44, 45, 46],
        "budgets": [10, 20, 50, 100],
        "row_seed": 42,
    }


class TestM5BuildingCountV2(unittest.TestCase):
    def test_sweep_has_40_collision_free_seed_aware_units(self) -> None:
        units = build_units(
            Path("/audit"),
            Path("/out"),
            audit_summary(),
            families=["tree", "tabpfn"],
            mode="formal",
            model_seed=42,
            validation_context_rows=200,
            validation_holdout_rows=200,
        )
        self.assertEqual(len(units), 40)
        identities = {tuple(sorted(unit["identity"].items())) for unit in units}
        outputs = {unit["output"] for unit in units}
        self.assertEqual(len(identities), 40)
        self.assertEqual(len(outputs), 40)
        for unit in units:
            identity = unit["identity"]
            self.assertEqual(identity["experiment_version"], EXPERIMENT_VERSION)
            self.assertEqual(identity["row_seed"], 42)
            self.assertEqual(identity["model_seed"], 42)
            self.assertIn(f"building_seed{identity['building_seed']}", unit["output"])
            command = unit["command"]
            self.assertNotIn("--early-stopping", command)
            if identity["model"] == "tree":
                self.assertIn("scripts/run_m5_building_count_v2_tree_cell.py", command)
            else:
                self.assertIn("--experiment-version", command)
                self.assertIn(EXPERIMENT_VERSION, command)

    def test_frozen_tree_contract_has_no_early_stopping_dependency(self) -> None:
        contract = frozen_model_contract(42)
        self.assertEqual(contract["lightgbm"]["params"]["n_estimators"], 100)
        self.assertEqual(contract["xgboost"]["params"]["n_estimators"], 100)
        self.assertEqual(contract["catboost"]["params"]["iterations"], 1000)
        self.assertEqual(contract["hist_gradient_boosting"]["params"]["max_iter"], 100)
        serialized = json.dumps(contract)
        self.assertNotIn("early_stopping", serialized)
        self.assertNotIn("eval_set", serialized)

    def test_v2_tree_and_tabpfn_arguments_preserve_separate_identity(self) -> None:
        manifest = Path("/audit/building_ladder_seed43.json")
        tree = v2_tree_args(
            [
                "--building-manifest",
                str(manifest),
                "--building-budget",
                "20",
                "--mode",
                "plan",
                "--out-root",
                "/out/building_seed43/tree_no_es_k20_f137",
            ]
        )
        tabpfn = tabpfn_args(
            [
                "--building-manifest",
                str(manifest),
                "--building-budget",
                "20",
                "--mode",
                "plan",
                "--experiment-version",
                EXPERIMENT_VERSION,
                "--out-root",
                "/out/building_seed43/tabpfn_k20_f137",
            ]
        )
        self.assertEqual(tree.model_seed, 42)
        self.assertEqual(tabpfn.model_seed, 42)
        self.assertEqual(tabpfn.experiment_version, EXPERIMENT_VERSION)
        self.assertFalse(hasattr(tree, "early_stopping_rounds"))
        self.assertFalse(hasattr(tree, "early_stop_rows"))

    def test_plan_mode_does_not_start_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit_root = Path(directory)
            (audit_root / "summary.json").write_text(
                json.dumps(audit_summary()), encoding="utf-8"
            )
            output = io.StringIO()
            with redirect_stdout(output):
                result = v2_main(
                    [
                        "--audit-root",
                        str(audit_root),
                        "--out-root",
                        str(audit_root / "out"),
                        "--mode",
                        "plan",
                    ]
                )
        self.assertEqual(result, 0)
        census = json.loads(output.getvalue())
        self.assertEqual(census["units"], 40)
        self.assertEqual(census["completed"], 0)
        self.assertIn("no early stopping", census["tree_training"])
        self.assertIn("byte-identical", census["matched_context"])

    def test_matched_context_gate_checks_metadata_and_prediction_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            units = []
            common = {
                "experiment_version": EXPERIMENT_VERSION,
                "building_seed": 42,
                "building_budget": 10,
                "features": 137,
                "row_seed": 42,
                "model_seed": 42,
                "row_policy": "average_building_cap",
                "context_rows": 5000,
                "context_row_sha256": "context",
                "holdout_row_sha256": "holdout",
                "training_sampling": ("exact_manifest_available_rows_no_resampling"),
                "class_ratio_policy": "natural_prevalence_of_manifest_available_rows",
            }
            arrays = {
                "validation_raw_index": np.array([3, 5], dtype="int64"),
                "anomaly": np.array([0, 1], dtype="int8"),
                "building_id": np.array([1, 3], dtype="int16"),
                "site_id": np.array([0, 1], dtype="int8"),
                "meter": np.array([0, 2], dtype="int8"),
            }
            for family in ("tree", "tabpfn"):
                output = root / family
                output.mkdir()
                (output / "cell.json").write_text(json.dumps(common), encoding="utf-8")
                np.savez(output / "predictions.npz", **arrays)
                units.append(
                    {
                        "identity": {
                            "building_seed": 42,
                            "K": 10,
                            "model": family,
                        },
                        "output": str(output),
                    }
                )
            records = matched_context_gate(units)
            self.assertEqual(records[0]["context_row_sha256"], "context")
            self.assertTrue(records[0]["passed"])

            changed = dict(common)
            changed["context_row_sha256"] = "different"
            (root / "tabpfn" / "cell.json").write_text(
                json.dumps(changed), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                AssertionError, "matched-context metadata mismatch"
            ):
                matched_context_gate(units)

    def test_report_requires_and_preserves_all_raw_seed_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_root = root / "audit"
            aggregate_root = root / "sweep" / "aggregate"
            audit_root.mkdir()
            aggregate_root.mkdir(parents=True)
            summary_payload = audit_summary()
            (audit_root / "summary.json").write_text(
                json.dumps(summary_payload), encoding="utf-8"
            )
            prefix_rows = [
                {
                    "building_seed": seed,
                    "K": budget,
                    "sampling_attempt": 0,
                    "attempts_used": 1,
                    "constraint_pass": True,
                    "reproducibility_digest": f"{seed}-{budget}",
                }
                for seed in summary_payload["building_seeds"]
                for budget in summary_payload["budgets"]
            ]
            pd.DataFrame(prefix_rows).to_csv(
                audit_root / "sampling_prefix_audit.csv", index=False
            )
            raw_rows = [
                {
                    "grouping": "overall",
                    "model": model,
                    "building_budget": budget,
                    "building_seed": seed,
                    "pr_auc": 0.5 + seed / 1000,
                    "roc_auc": 0.7 + budget / 1000,
                }
                for model in ("tabpfn", "ensemble")
                for budget in summary_payload["budgets"]
                for seed in summary_payload["building_seeds"]
            ]
            pd.DataFrame(raw_rows).to_csv(aggregate_root / "metrics.csv", index=False)
            seed_rows = [
                {
                    "sampling_profile": "site_stratified_random",
                    "features": 137,
                    "building_budget": budget,
                    "model": model,
                    "n_building_seeds": 5,
                    "pr_auc_mean": 0.5,
                    "pr_auc_std": 0.01,
                    "pr_auc_min": 0.48,
                    "pr_auc_max": 0.52,
                    "roc_auc_mean": 0.7,
                    "roc_auc_std": 0.02,
                }
                for model in ("tabpfn", "ensemble")
                for budget in summary_payload["budgets"]
            ]
            pd.DataFrame(seed_rows).to_csv(
                aggregate_root / "building_seed_summary.csv", index=False
            )
            (aggregate_root.parent / "matched_context_gate.json").write_text(
                json.dumps({"passed": True, "checked_cells": 20}),
                encoding="utf-8",
            )
            report = root / "report.md"
            result = report_main(
                [
                    "--audit-root",
                    str(audit_root),
                    "--aggregate-root",
                    str(aggregate_root),
                    "--out",
                    str(report),
                ]
            )
            self.assertEqual(result, 0)
            text = report.read_text(encoding="utf-8")
            self.assertIn("**Status:** complete", text)
            self.assertIn("Per-seed overall results", text)
            self.assertIn("| ensemble | 100 | 46 |", text)

            pd.DataFrame(seed_rows[:-1]).to_csv(
                aggregate_root / "building_seed_summary.csv", index=False
            )
            with self.assertRaisesRegex(
                SystemExit, "headline cross-seed summary is incomplete"
            ):
                report_main(
                    [
                        "--audit-root",
                        str(audit_root),
                        "--aggregate-root",
                        str(aggregate_root),
                        "--out",
                        str(report),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
