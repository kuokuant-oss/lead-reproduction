from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.m5_building_count_v3_protocol import (
    BUDGETS,
    BUILDING_SEEDS,
    CLASS_RATIO_POLICY,
    EXPERIMENT_VERSION,
    SAMPLING_PROFILE,
    TRAINING_CONTEXT_POLICY,
    grouped_budget_major_pairs,
)
from scripts import run_m5_building_count_v3 as scheduler


def identity(*, seed: int, budget: int, model: str) -> dict[str, object]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "building_seed": seed,
        "K": budget,
        "model": model,
        "features": 137,
        "balance_seed": 42,
        "model_seed": 42,
        "expected_context_rows": 4,
        "expected_holdout_rows": 2,
    }


def metadata(*, seed: int = 42, budget: int = 10, model: str) -> dict[str, object]:
    experiment = (
        "m5_building_count_v3_balanced_context_tree_cell"
        if model == "tree"
        else "m5_building_count_v3_balanced_context_tabpfn_cell"
    )
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "experiment": experiment,
        "mode": "NON_SCIENTIFIC_VALIDATION",
        "building_seed": seed,
        "building_budget": budget,
        "features": 137,
        "row_seed": 42,
        "model_seed": 42,
        "sampling_profile": SAMPLING_PROFILE,
        "training_sampling": TRAINING_CONTEXT_POLICY,
        "class_ratio_policy": CLASS_RATIO_POLICY,
        "row_policy": "frozen_balanced_context_artifact",
        "context_rows": 4,
        "context_row_sha256": "a" * 64,
        "context_label_sha256": "b" * 64,
        "context_feature_matrix_sha256": "c" * 64,
        "holdout_rows": 2,
        "holdout_row_sha256": "d" * 64,
        "validation_feature_scope": "bounded_context_and_holdout_rows",
        "manifest_sha256": "e" * 64,
        "source_building_manifest_sha256": "f" * 64,
    }


def write_complete(root: Path, *, model: str) -> dict[str, object]:
    root.mkdir(parents=True)
    (root / "COMPLETE.json").write_text("{}", encoding="utf-8")
    (root / "cell.json").write_text(json.dumps(metadata(model=model)), encoding="utf-8")
    np.savez(
        root / "predictions.npz",
        validation_raw_index=np.array([1, 3], dtype="int64"),
        anomaly=np.array([0, 1], dtype="int8"),
        building_id=np.array([1, 3], dtype="int16"),
        site_id=np.array([0, 1], dtype="int8"),
        meter=np.array([0, 2], dtype="int8"),
    )
    return {
        "identity": identity(seed=42, budget=10, model=model),
        "context_manifest": str(root.parent / "balanced_context_seed42.json"),
        "output": str(root),
        "command": ["never-run"],
    }


class TestM5BuildingCountV3Scheduler(unittest.TestCase):
    def test_grouped_order_finishes_42_46_before_47_51(self) -> None:
        pairs = grouped_budget_major_pairs()
        self.assertEqual(len(pairs), 40)
        self.assertEqual(pairs[:5], [(seed, 10) for seed in range(42, 47)])
        self.assertEqual(pairs[5:10], [(seed, 20) for seed in range(42, 47)])
        self.assertEqual(pairs[15:20], [(seed, 100) for seed in range(42, 47)])
        self.assertEqual(pairs[20:25], [(seed, 10) for seed in range(47, 52)])
        self.assertEqual(pairs[-5:], [(seed, 100) for seed in range(47, 52)])

    def test_default_mode_is_non_launching_plan(self) -> None:
        parsed = scheduler.parse_args([])
        self.assertEqual(parsed.mode, "plan")
        self.assertFalse(parsed.authorize_formal)
        unit = {
            "identity": identity(seed=42, budget=10, model="tree"),
            "output": "/not/used",
        }
        with (
            patch.object(
                scheduler, "verify_training_context_gate", return_value={"passed": True}
            ),
            patch.object(scheduler, "build_units", return_value=[unit]),
            patch.object(scheduler.subprocess, "run") as run,
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(scheduler.main([]), 0)
        run.assert_not_called()
        census = json.loads(output.getvalue())
        self.assertEqual(census["mode"], "plan")

    def test_build_units_has_80_shared_context_commands(self) -> None:
        def context(path: Path, budget: int) -> SimpleNamespace:
            seed = int(path.stem.removeprefix("balanced_context_seed"))
            return SimpleNamespace(
                source_manifest_path=Path(f"/source/building_ladder_seed{seed}.json")
            )

        with patch.object(scheduler, "load_balanced_context", side_effect=context):
            units = scheduler.build_units(
                Path("/audit"),
                Path("/validation"),
                mode="validation",
                model_seed=42,
                model_path=Path("/models/tabpfn.ckpt"),
                validation_context_rows=200,
                validation_holdout_rows=200,
            )
        self.assertEqual(len(units), 80)
        for offset in range(0, 80, 2):
            tree, tabpfn = units[offset : offset + 2]
            self.assertEqual(tree["identity"]["model"], "tree")
            self.assertEqual(tabpfn["identity"]["model"], "tabpfn")
            self.assertEqual(tree["context_manifest"], tabpfn["context_manifest"])
            self.assertIn("--balanced-context-manifest", tree["command"])
            self.assertIn("--balanced-context-manifest", tabpfn["command"])
            self.assertIn("--max-context-rows", tree["command"])
            self.assertIn("--max-holdout-rows", tabpfn["command"])
            self.assertIn("--model-path", tabpfn["command"])
            self.assertIn("/models/tabpfn.ckpt", tabpfn["command"])
        pair_order = [
            (units[index]["identity"]["building_seed"], units[index]["identity"]["K"])
            for index in range(0, len(units), 2)
        ]
        self.assertEqual(pair_order, grouped_budget_major_pairs())

    def test_validation_and_formal_roots_are_isolated(self) -> None:
        validation = scheduler.parse_args(["--mode", "validation"])
        formal = scheduler.parse_args(["--mode", "formal"])
        self.assertNotEqual(validation.out_root, formal.out_root)
        self.assertIn("NON_SCIENTIFIC_VALIDATION", str(validation.out_root))
        self.assertNotIn("NON_SCIENTIFIC_VALIDATION", str(formal.out_root))

    def test_mode_guards_reject_invalid_caps_and_authorization(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive and even"):
            scheduler.parse_args(["--validation-context-rows", "201"])
        with self.assertRaisesRegex(ValueError, "only valid in formal"):
            scheduler.parse_args(["--authorize-formal"])
        with self.assertRaisesRegex(ValueError, "visibly non-scientific"):
            scheduler.parse_args(
                ["--mode", "validation", "--out-root", "/tmp/formal-looking"]
            )

    def test_complete_rejects_missing_corrupt_and_provenance_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tree"
            unit = write_complete(root, model="tree")
            self.assertTrue(scheduler.complete(unit, mode="validation"))
            changed = metadata(model="tree")
            changed["row_seed"] = 99
            (root / "cell.json").write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            scheduler.parse_args(["--validation-context-rows", "6000"])
            self.assertFalse(scheduler.complete(unit, mode="validation"))
            (root / "cell.json").write_text("{", encoding="utf-8")
            self.assertFalse(scheduler.complete(unit, mode="validation"))

    def test_matched_gate_checks_row_label_feature_and_holdout_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            units = [
                write_complete(root / family, model=family)
                for family in ("tree", "tabpfn")
            ]
            fake_context = SimpleNamespace(
                manifest={"cells": {"10": {"raw_index_sha256": "formal"}}}
            )
            with patch.object(
                scheduler, "load_balanced_context", return_value=fake_context
            ):
                records = scheduler.matched_context_gate(units, mode="validation")
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]["passed"])

            changed = metadata(model="tabpfn")
            changed["context_feature_matrix_sha256"] = "different"
            (root / "tabpfn" / "cell.json").write_text(
                json.dumps(changed), encoding="utf-8"
            )
            with (
                patch.object(
                    scheduler, "load_balanced_context", return_value=fake_context
                ),
                self.assertRaisesRegex(AssertionError, "metadata mismatch"),
            ):
                scheduler.matched_context_gate(units, mode="validation")

    def test_missing_pairs_cannot_pass_final_gate(self) -> None:
        self.assertEqual(scheduler.matched_context_gate([], mode="validation"), [])
        expected = len(BUILDING_SEEDS) * len(BUDGETS)
        self.assertNotEqual(len([]), expected)

    def test_atomic_status_write_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            scheduler._atomic_json(path, {"status": "running", "completed": 1})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["completed"], 1
            )
            self.assertFalse(path.with_name("status.json.tmp").exists())

    def test_formal_preflight_requires_explicit_flag_and_clean_validation(self) -> None:
        with self.assertRaisesRegex(SystemExit, "explicit --authorize-formal"):
            scheduler._formal_preflight(SimpleNamespace(authorize_formal=False))
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.ckpt"
            model_path.write_bytes(b"checkpoint")
            clean = subprocess.CompletedProcess(["git"], 0, stdout="", stderr="")
            with (
                patch.object(
                    scheduler, "_validation_gate", return_value={"passed": True}
                ),
                patch.object(scheduler.subprocess, "run", return_value=clean),
            ):
                scheduler._formal_preflight(
                    SimpleNamespace(authorize_formal=True, model_path=model_path)
                )


if __name__ == "__main__":
    unittest.main()
