from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import prepare_m5_building_count_v5_fixed_50k as prepare
from scripts import run_m5_building_count_v5 as launcher
protocol = launcher.protocol


class TestM5BuildingCountV5Preparation(unittest.TestCase):
    def test_three_phase_context_order_and_census(self) -> None:
        contexts = prepare.planned_contexts()
        self.assertEqual(len(contexts), 42)
        self.assertEqual(len({context.key for context in contexts}), 42)
        self.assertEqual(
            [(c.budget, c.building_seed, c.row_seed) for c in contexts[:2]],
            [(725, None, 0), (725, None, 1)],
        )
        phase2 = contexts[2:26]
        self.assertEqual(set(c.building_seed for c in phase2), {0, 1, 2})
        self.assertEqual(
            [c.budget for c in phase2],
            [400] * 6 + [200] * 6 + [100] * 6 + [50] * 6,
        )
        phase3 = contexts[26:]
        self.assertEqual(set(c.building_seed for c in phase3), {3, 4})
        self.assertEqual(
            [c.budget for c in phase3],
            [400] * 4 + [200] * 4 + [100] * 4 + [50] * 4,
        )

    def test_row_draw_is_balanced_unique_deterministic_and_seeded(self) -> None:
        building = np.zeros(60_000, dtype="int32")
        anomaly = np.tile(np.array([1, 0], dtype="int8"), 30_000)
        support = np.array([0], dtype="int64")
        identity0 = prepare.ContextIdentity(725, None, 0, prepare.FULL_SUPPORT_ID)
        identity1 = prepare.ContextIdentity(725, None, 1, prepare.FULL_SUPPORT_ID)
        first, capacity = prepare.draw_context(
            building, anomaly, support, identity0
        )
        repeated, _ = prepare.draw_context(building, anomaly, support, identity0)
        second, _ = prepare.draw_context(building, anomaly, support, identity1)
        self.assertEqual(len(first), 50_000)
        self.assertEqual(len(np.unique(first)), 50_000)
        self.assertEqual(int(anomaly[first].sum()), 25_000)
        self.assertEqual(capacity, {"full_anomalies": 30_000, "full_normals": 30_000})
        self.assertTrue(np.array_equal(first, repeated))
        self.assertFalse(np.array_equal(first, second))

    def test_short_class_support_fails_without_repair(self) -> None:
        building = np.zeros(49_999, dtype="int32")
        anomaly = np.zeros(49_999, dtype="int8")
        anomaly[:24_999] = 1
        identity = prepare.ContextIdentity(725, None, 0, prepare.FULL_SUPPORT_ID)
        with self.assertRaisesRegex(ValueError, "insufficient unique class support"):
            prepare.draw_context(
                building, anomaly, np.array([0], dtype="int64"), identity
            )

    def test_atomic_npz_has_no_temporary_residue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.npz"
            prepare.atomic_savez(path, raw_index=np.arange(5, dtype="int64"))
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_name("context.npz.tmp").exists())

    def test_full_support_provenance_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.json"
            source_path.write_text("{}", encoding="utf-8")
            selected = np.arange(0, 1450, 2, dtype="int64")
            support = prepare.SourceSupport(
                selected_buildings=selected,
                source_manifest_path=source_path,
                source_manifest_sha256=prepare.sha256_file(source_path),
                source_cell_selected_sha256=prepare.int_array_sha256(selected),
            )
            raw = {"train_csv_sha256": "a" * 64, "bad_meter_readings_csv_sha256": "b" * 64}
            prepare.prepare_full_support_manifest(
                support,
                audit_root=root,
                raw_inputs=raw,
                preparation_commit="c" * 40,
            )
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                prepare.prepare_full_support_manifest(
                    support,
                    audit_root=root,
                    raw_inputs=raw,
                    preparation_commit="d" * 40,
                )

    def test_missing_contexts_refuse_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing contexts"):
                prepare.finalize_gate(
                    [],
                    contexts=prepare.planned_contexts(),
                    raw_inputs={},
                    source={},
                    evaluation_holdout={},
                    preparation_commit="c" * 40,
                    audit_root=Path(directory),
                    mode="FORMAL_INPUT_PREPARATION",
                )

    def test_mode_guards_keep_default_non_launching(self) -> None:
        self.assertEqual(prepare.parse_args([]).mode, "plan")
        with self.assertRaises(SystemExit):
            prepare.parse_args(["--mode", "prepare"])
        with self.assertRaises(SystemExit):
            prepare.parse_args(["--mode", "validation", "--max-units", "2"])


class TestM5BuildingCountV5Scheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        launcher._configure()

    def test_scheduler_context_order(self) -> None:
        contexts = protocol.queued_contexts()
        self.assertEqual(len(contexts), 42)
        self.assertEqual(contexts[:2], [(725, 0, 725), (725, 1, 725)])
        self.assertEqual(protocol.phase_final_contexts()[1], (725, 1, 725))
        self.assertEqual(protocol.phase_final_contexts()[2], (2, 1, 50))
        self.assertEqual(protocol.phase_final_contexts()[3], (4, 1, 50))

    def test_formal_plan_has_84_dedicated_model_units(self) -> None:
        def context(path: Path, budget: int) -> SimpleNamespace:
            name = path.stem
            if "_full_" in name:
                building_draw_seed = None
                support_id = "full_725"
            else:
                building_draw_seed = int(name.split("_b", 1)[1].split("_", 1)[0])
                support_id = f"building_seed{building_draw_seed}"
            return SimpleNamespace(
                source_manifest_path=Path(f"/audit/source_{support_id}.json"),
                building_draw_seed=building_draw_seed,
                support_id=support_id,
            )

        with patch.object(protocol, "load_fixed_context", side_effect=context):
            units = launcher._build_units(
                Path("/audit"),
                Path("/formal"),
                mode="formal",
                model_seed=42,
                model_path=Path("/model.ckpt"),
                validation_context_rows=200,
                validation_holdout_rows=200,
            )
        self.assertEqual(len(units), 84)
        self.assertEqual([unit["identity"]["phase"] for unit in units[:4]], [1] * 4)
        self.assertEqual([unit["identity"]["phase"] for unit in units[4:52]], [2] * 48)
        self.assertEqual([unit["identity"]["phase"] for unit in units[52:]], [3] * 32)
        for tree, tabpfn in zip(units[0::2], units[1::2], strict=True):
            self.assertIn("run_m5_building_count_v5_tree_cell.py", tree["command"][1])
            self.assertIn(
                "run_m5_building_count_v5_tabpfn_cell.py", tabpfn["command"][1]
            )
            self.assertNotIn("--experiment-version", tree["command"])
            self.assertIn("--experiment-version", tabpfn["command"])

    def test_model_validation_is_two_bounded_extremes(self) -> None:
        self.assertEqual(
            launcher._selected_contexts("validation"),
            [(725, 0, 725), (0, 0, 50)],
        )


if __name__ == "__main__":
    unittest.main()
