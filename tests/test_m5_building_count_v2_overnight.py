from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_m5_building_count_v2_overnight as supervisor
from scripts.run_m5_building_count_v2 import (
    EXPERIMENT_VERSION,
    budget_major_seed_pairs,
    build_units,
    ordered_seed_budget_pairs,
)
from scripts.update_m5_building_count_v2_progress import (
    BEGIN,
    END,
    render_progress,
    replace_progress,
)


def summary() -> dict[str, object]:
    return {
        "status": "audit_passed_ready_for_model_evaluation",
        "sampling_profile": "site_stratified_random",
        "building_seeds": [42, 43, 44, 45, 46],
        "budgets": [10, 20, 50, 100],
        "row_seed": 42,
    }


def unit(root: Path, *, seed: int, budget: int, model: str) -> dict:
    return {
        "identity": {
            "experiment_version": EXPERIMENT_VERSION,
            "sampling_profile": "site_stratified_random",
            "building_seed": seed,
            "K": budget,
            "features": 137,
            "model": model,
            "row_seed": 42,
            "model_seed": 42,
        },
        "output": str(root / f"seed{seed}_k{budget}_{model}"),
        "command": [f"run-{seed}-{budget}-{model}"],
    }


def args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        audit_root=root / "audit",
        report=root / "report.md",
        retry_delay=0,
        unit_retries=2,
        finalize_retries=2,
        push_retries=3,
        git_push_timeout=10,
        gpu_wait_checks=1,
    )


class TestM5BuildingCountV2Overnight(unittest.TestCase):
    def test_pair_order_finishes_seed42_then_sweeps_each_k(self) -> None:
        expected = [
            (42, 10),
            (42, 20),
            (42, 50),
            (42, 100),
            (43, 10),
            (44, 10),
            (45, 10),
            (46, 10),
            (43, 20),
            (44, 20),
            (45, 20),
            (46, 20),
            (43, 50),
            (44, 50),
            (45, 50),
            (46, 50),
            (43, 100),
            (44, 100),
            (45, 100),
            (46, 100),
        ]
        self.assertEqual(ordered_seed_budget_pairs(summary()), expected)
        units = build_units(
            Path("/audit"),
            Path("/out"),
            summary(),
            families=["tree", "tabpfn"],
            mode="formal",
            model_seed=42,
            validation_context_rows=200,
            validation_holdout_rows=200,
        )
        observed = [
            (item["identity"]["building_seed"], item["identity"]["K"])
            for item in units[::2]
        ]
        self.assertEqual(observed, expected)
        for offset in range(0, len(units), 2):
            self.assertEqual(
                [
                    units[offset]["identity"]["model"],
                    units[offset + 1]["identity"]["model"],
                ],
                ["tree", "tabpfn"],
            )

    def test_budget_major_order_sweeps_k10_k20_k50_k100(self) -> None:
        extension = {
            **summary(),
            "building_seeds": [47, 48, 49, 50, 51],
        }
        expected = [
            (seed, budget)
            for budget in (10, 20, 50, 100)
            for seed in (47, 48, 49, 50, 51)
        ]
        self.assertEqual(budget_major_seed_pairs(extension), expected)
        parsed = supervisor.parse_args(["--pair-order", "budget-major"])
        self.assertEqual(parsed.pair_order, "budget-major")

    def test_defaults_are_non_launching_and_bounded(self) -> None:
        parsed = supervisor.parse_args([])
        self.assertEqual(parsed.mode, "plan")
        self.assertEqual(parsed.pair_order, "scientific")
        self.assertEqual(parsed.unit_retries, 2)
        self.assertEqual(parsed.finalize_retries, 2)
        self.assertEqual(parsed.push_retries, 5)
        self.assertFalse(parsed.publish_results)

    def test_exhausted_unit_is_marked_after_three_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = unit(root, seed=42, budget=10, model="tree")
            failed_result = subprocess.CompletedProcess(["run"], 1)
            with (
                mock.patch.object(supervisor, "_complete", return_value=False),
                mock.patch.object(
                    supervisor.subprocess, "run", return_value=failed_result
                ) as command,
                mock.patch.object(supervisor, "_event"),
            ):
                complete = supervisor._run_unit(
                    args(root),
                    current,
                    root / "overnight",
                    [current],
                    failed_units=[],
                    failed_publications=[],
                    pair_index=1,
                    pair_count=20,
                )
            self.assertFalse(complete)
            self.assertEqual(command.call_count, 3)
            marker = (
                root / "overnight" / "failed_stages" / "building_seed42_k10_tree.json"
            )
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(payload["attempts"], 3)
            self.assertEqual(payload["reason"], "unit_retry_limit_exhausted")

    def test_push_retries_are_bounded_and_marked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed_result = subprocess.CompletedProcess(["git", "push"], 1)
            with (
                mock.patch.object(
                    supervisor.subprocess, "run", return_value=failed_result
                ) as command,
                mock.patch.object(supervisor, "_event"),
            ):
                self.assertFalse(
                    supervisor._push_with_retries(
                        args(root),
                        root / "overnight",
                        "building_seed42_k10",
                    )
                )
            self.assertEqual(command.call_count, 3)
            marker = (
                root / "overnight" / "failed_publications" / "building_seed42_k10.json"
            )
            self.assertTrue(marker.is_file())

    def test_progress_report_records_complete_pair_and_is_replaceable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small_summary = {
                **summary(),
                "building_seeds": [42, 43],
                "budgets": [10],
            }
            units = build_units(
                root / "audit",
                root / "out",
                small_summary,
                families=["tree", "tabpfn"],
                mode="formal",
                model_seed=42,
                validation_context_rows=200,
                validation_holdout_rows=200,
            )
            for item in units[:2]:
                output = Path(item["output"])
                output.mkdir(parents=True)
                identity = item["identity"]
                score = "ensemble" if identity["model"] == "tree" else "tabpfn"
                (output / "cell.json").write_text(
                    json.dumps(
                        {
                            "experiment_version": EXPERIMENT_VERSION,
                            "building_seed": identity["building_seed"],
                            "building_budget": identity["K"],
                            "features": 137,
                            "row_seed": 42,
                            "model_seed": 42,
                            "metrics": {score: {"pr_auc": 0.7, "roc_auc": 0.9}},
                        }
                    ),
                    encoding="utf-8",
                )
                (output / "COMPLETE.json").write_text("{}", encoding="utf-8")
                (output / "predictions.npz").write_bytes(b"placeholder")
            section = render_progress(
                small_summary,
                units,
                root / "overnight",
                last_pair="building_seed42_k10",
            )
            first = replace_progress("# V2\n", section)
            second = replace_progress(first, section)
            self.assertEqual(first, second)
            self.assertEqual(first.count(BEGIN), 1)
            self.assertEqual(first.count(END), 1)
            self.assertIn("| 1 | 42 | 10 | complete | yes | yes |", first)
            self.assertIn("Completed seed/K pairs: 1/2", first)

    def test_watchdog_restores_formal_resume_and_publication(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "ensure_m5_building_count_v2_overnight.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("run_m5_building_count_v2_overnight.py", source)
        self.assertIn("--mode formal", source)
        self.assertIn("--unit-retries 2", source)
        self.assertIn("--finalize-retries 2", source)
        self.assertIn("--push-retries 5", source)
        self.assertIn("--publish-results", source)
        self.assertIn("pgrep -f", source)
        self.assertNotIn("kill-session", source)


if __name__ == "__main__":
    unittest.main()
