from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts import run_m5_building_candidate_sensitivity_overnight as supervisor
from scripts.update_m5_building_candidate_sensitivity_report import (
    BEGIN,
    END,
    render_section,
    replace_section,
)


def _unit(root: Path, *, seed: int, model: str = "tree") -> dict:
    return {
        "identity": {
            "sampling_profile": "representative",
            "building_seed": seed,
            "K": 10,
            "features": 137,
            "model": model,
        },
        "output": str(root / f"seed{seed}_{model}"),
        "command": [f"run-{seed}-{model}"],
    }


def _args(audit_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        audit_root=audit_root,
        unit_retries=1,
        retry_delay=0,
        gpu_wait_checks=1,
    )


class TestSensitivityOvernightSupervisor(unittest.TestCase):
    def test_supervisor_defaults_to_non_launching_plan_mode(self) -> None:
        args = supervisor.parse_args([])
        self.assertEqual(args.mode, "plan")

    def test_attempt_count_is_atomic_and_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supervisor._record_attempt(root, "seed42_k10_tree", 2)
            self.assertEqual(supervisor._attempt_count(root, "seed42_k10_tree"), 2)
            self.assertFalse(
                supervisor._attempt_path(root, "seed42_k10_tree")
                .with_suffix(".json.tmp")
                .exists()
            )

    def test_exhausted_unit_is_marked_and_later_units_continue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_root = root / "audit"
            units = [_unit(root, seed=42), _unit(root, seed=43)]

            def complete(unit: dict) -> bool:
                return unit["identity"]["building_seed"] == 43

            failed_result = subprocess.CompletedProcess(["run"], 1)
            with (
                mock.patch.object(supervisor, "_complete", side_effect=complete),
                mock.patch.object(
                    supervisor.subprocess, "run", return_value=failed_result
                ) as command,
                mock.patch.object(supervisor, "_event"),
            ):
                failed = supervisor._run_units(
                    _args(audit_root), units, audit_root / "overnight"
                )
            self.assertEqual(failed, ["building_seed42_k10_f137_tree"])
            self.assertEqual(command.call_count, 2)
            marker = (
                audit_root
                / "overnight"
                / "failed_stages"
                / "building_seed42_k10_f137_tree.json"
            )
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(payload["attempts"], 2)
            self.assertTrue(payload["requires_review"])

            with (
                mock.patch.object(supervisor, "_complete", side_effect=complete),
                mock.patch.object(supervisor.subprocess, "run") as resumed_command,
                mock.patch.object(supervisor, "_event"),
            ):
                resumed = supervisor._run_units(
                    _args(audit_root), units, audit_root / "overnight"
                )
            self.assertEqual(resumed, ["building_seed42_k10_f137_tree"])
            resumed_command.assert_not_called()

    def test_finalize_stage_has_a_bounded_retry_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(finalize_retries=2, retry_delay=0)
            action = mock.Mock(return_value=1)
            self.assertFalse(
                supervisor._bounded_stage(
                    args,
                    root,
                    stage="aggregate",
                    action=action,
                    valid=lambda: False,
                )
            )
            self.assertEqual(action.call_count, 3)
            failure = json.loads(
                (root / "failed_stages" / "aggregate.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["attempts"], 3)

    def test_watchdog_waits_for_existing_formal_owner(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "ensure_m5_building_candidate_sensitivity_overnight.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("pgrep -f", source)
        self.assertIn("existing formal sweep owns execution", source)
        self.assertIn("--unit-retries 2", source)
        self.assertIn("--mode formal", source)
        self.assertIn("--publish-results", source)
        self.assertNotIn("kill-session", source)


class TestSensitivityReportUpdater(unittest.TestCase):
    def test_report_section_is_complete_and_idempotently_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seeds = [42, 43, 44]
            budgets = [10, 20, 50, 100]
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "audit_passed_ready_for_model_evaluation",
                        "building_seeds": seeds,
                        "budgets": budgets,
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "K": budget,
                        "seed_a": left,
                        "seed_b": right,
                        "intersection_count": 5,
                        "jaccard_similarity": 1 / 3,
                    }
                    for budget in budgets
                    for left, right in ((42, 43), (42, 44), (43, 44))
                ]
            ).to_csv(root / "building_overlap.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "building_seed": seed,
                        "K": budget,
                        "total_available_rows": 1000,
                        "total_anomaly_rows": 50,
                        "natural_anomaly_prevalence": 0.05,
                        "prefix_discrepancy": 0.01,
                        "quality_gate_pass": True,
                    }
                    for seed in seeds
                    for budget in budgets
                ]
            ).to_csv(root / "composition_audit.csv", index=False)
            aggregate = root / "model_results" / "building_seed_sweep_42-43-44"
            aggregate.mkdir(parents=True)
            (aggregate / "summary.json").write_text(
                json.dumps(
                    {
                        "cells": [
                            {"seed": seed, "K": budget, "model": model}
                            for seed in seeds
                            for budget in budgets
                            for model in ("tree", "tabpfn")
                        ]
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "sampling_profile": "representative",
                        "features": 137,
                        "building_budget": budget,
                        "model": model,
                        "n_building_seeds": 3,
                        "pr_auc_mean": 0.7,
                        "pr_auc_std": 0.01,
                        "pr_auc_min": 0.69,
                        "pr_auc_max": 0.71,
                        "roc_auc_mean": 0.95,
                        "roc_auc_std": 0.005,
                    }
                    for budget in budgets
                    for model in ("ensemble", "tabpfn")
                ]
            ).to_csv(aggregate / "building_seed_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "grouping": "overall",
                        "model": model,
                        "building_budget": budget,
                        "building_seed": seed,
                        "pr_auc": 0.7,
                        "roc_auc": 0.95,
                    }
                    for model in ("ensemble", "tabpfn")
                    for budget in budgets
                    for seed in seeds
                ]
            ).to_csv(aggregate / "metrics.csv", index=False)

            section = render_section(root)
            first = replace_section("# Existing report\n", section)
            second = replace_section(first, section)
            self.assertEqual(first, second)
            self.assertEqual(first.count(BEGIN), 1)
            self.assertEqual(first.count(END), 1)
            self.assertIn("Cross-seed overall results", first)
            self.assertIn("Per-seed overall results", first)


if __name__ == "__main__":
    unittest.main()
