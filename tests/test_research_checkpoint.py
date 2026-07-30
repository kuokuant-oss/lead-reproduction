from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts._research_checkpoint import (
    CheckpointError,
    ResearchCheckpointStore,
    canonical_unit_id,
)
from scripts.analyze_m5_meter_specific_learner_gap import (
    EXPECTED_VALIDATION_INTERRUPTION_EXIT,
    ExpectedValidationInterruption,
    PhaseHeartbeat,
    ValidationStopController,
    _execute_unit,
    _finish_phase,
)


class TestResearchCheckpoint(unittest.TestCase):
    def test_atomic_reuse_and_finalization_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = ResearchCheckpointStore(root, "bootstrap", {"seed": 7, "rows": "a"})
            with self.assertRaisesRegex(CheckpointError, "cannot finalize"):
                store.complete_phase(["meter/steam/draw/0"])
            store.write_unit("meter/steam/draw/0", {"estimate": 1.0})
            self.assertEqual(
                store.completed_units(["meter/steam/draw/0"]), {"meter/steam/draw/0"}
            )
            store.complete_phase(["meter/steam/draw/0"])
            self.assertTrue(
                (root / "checkpoints" / "bootstrap" / "COMPLETE.json").exists()
            )

    def test_temporary_corrupt_and_provenance_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = ResearchCheckpointStore(root, "loo", {"seed": 7})
            unit = "steam__building__12"
            store.unit_path(unit).parent.mkdir(parents=True)
            store.unit_path(unit).with_suffix(".json.tmp").write_text(
                "{}", encoding="utf-8"
            )
            self.assertEqual(store.completed_units([unit]), set())
            store.write_unit(unit, {"x": 1})
            payload = json.loads(store.unit_path(unit).read_text(encoding="utf-8"))
            payload["payload"]["x"] = 2
            store.unit_path(unit).write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CheckpointError, "corrupt"):
                store.read_unit(unit)
            with self.assertRaisesRegex(CheckpointError, "provenance mismatch"):
                ResearchCheckpointStore(root, "loo", {"seed": 8})

    def test_ids_and_heartbeat_are_safe_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ResearchCheckpointStore(Path(temporary), "segment", {"version": 1})
            self.assertEqual(canonical_unit_id("meter/steam", 1), "meter-steam__1")
            store.heartbeat(completed_units=1, total_units=2, current_unit="steam")
            status = json.loads(
                (
                    Path(temporary) / "checkpoints" / "segment" / "heartbeat.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(status["completed_units"], 1)

    def test_expected_interruption_preserves_unit_and_resume_reuses_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = ResearchCheckpointStore(
                root, "bootstrap", {"mode": "NON_SCIENTIFIC_VALIDATION"}
            )
            expected = ["steam__draw__0", "steam__draw__1"]
            heartbeat = PhaseHeartbeat(store, expected)
            controller = ValidationStopController(root, stop_after_units=1)
            with self.assertRaises(ExpectedValidationInterruption):
                _execute_unit(
                    store=store,
                    expected=expected,
                    unit_id=expected[0],
                    meter="steam",
                    rows=8,
                    compute=lambda: {"estimate": 1.0},
                    controller=controller,
                    heartbeat=heartbeat,
                )
            self.assertEqual(store.completed_units(expected), {expected[0]})
            self.assertFalse(
                (root / "checkpoints" / "bootstrap" / "COMPLETE.json").exists()
            )
            stop = json.loads(
                (root / "EXPECTED_VALIDATION_INTERRUPTION.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(stop["exit_code"], EXPECTED_VALIDATION_INTERRUPTION_EXIT)
            resumed = ValidationStopController(root, stop_after_units=None)
            resumed_heartbeat = PhaseHeartbeat(store, expected)
            calls = 0

            def should_not_compute() -> dict[str, float]:
                nonlocal calls
                calls += 1
                return {"estimate": 2.0}

            reused = _execute_unit(
                store=store,
                expected=expected,
                unit_id=expected[0],
                meter="steam",
                rows=8,
                compute=should_not_compute,
                controller=resumed,
                heartbeat=resumed_heartbeat,
            )
            computed = _execute_unit(
                store=store,
                expected=expected,
                unit_id=expected[1],
                meter="steam",
                rows=8,
                compute=lambda: {"estimate": 3.0},
                controller=resumed,
                heartbeat=resumed_heartbeat,
            )
            self.assertEqual(calls, 0)
            self.assertEqual(reused, {"estimate": 1.0})
            self.assertEqual(computed, {"estimate": 3.0})
            self.assertEqual(resumed.reused_units, 1)
            _finish_phase(store, expected, resumed_heartbeat)

    def test_interrupted_then_resumed_payloads_equal_uninterrupted_payloads(
        self,
    ) -> None:
        """Exercise the complete Run 1 -> Run 2 contract without artifacts."""
        expected = ["steam__draw__0", "steam__draw__1", "steam__draw__2"]

        def payload(unit: str) -> dict[str, int | str]:
            return {"unit": unit, "estimate": int(unit[-1])}

        with (
            tempfile.TemporaryDirectory() as interrupted_dir,
            tempfile.TemporaryDirectory() as reference_dir,
        ):
            interrupted_root, reference_root = (
                Path(interrupted_dir),
                Path(reference_dir),
            )
            interrupted = ResearchCheckpointStore(
                interrupted_root, "bootstrap", {"seed": 7}
            )
            interrupted_heartbeat = PhaseHeartbeat(interrupted, expected)
            stop = ValidationStopController(interrupted_root, stop_after_units=1)
            with self.assertRaises(ExpectedValidationInterruption):
                _execute_unit(
                    store=interrupted,
                    expected=expected,
                    unit_id=expected[0],
                    meter="steam",
                    rows=8,
                    compute=lambda: payload(expected[0]),
                    controller=stop,
                    heartbeat=interrupted_heartbeat,
                )
            resumed = ValidationStopController(interrupted_root, stop_after_units=None)
            resumed_heartbeat = PhaseHeartbeat(interrupted, expected)
            for unit in expected:
                _execute_unit(
                    store=interrupted,
                    expected=expected,
                    unit_id=unit,
                    meter="steam",
                    rows=8,
                    compute=lambda unit=unit: payload(unit),
                    controller=resumed,
                    heartbeat=resumed_heartbeat,
                )
            _finish_phase(interrupted, expected, resumed_heartbeat)
            reference = ResearchCheckpointStore(
                reference_root, "bootstrap", {"seed": 7}
            )
            reference_heartbeat = PhaseHeartbeat(reference, expected)
            uninterrupted = ValidationStopController(
                reference_root, stop_after_units=None
            )
            for unit in expected:
                _execute_unit(
                    store=reference,
                    expected=expected,
                    unit_id=unit,
                    meter="steam",
                    rows=8,
                    compute=lambda unit=unit: payload(unit),
                    controller=uninterrupted,
                    heartbeat=reference_heartbeat,
                )
            _finish_phase(reference, expected, reference_heartbeat)
            self.assertEqual(
                interrupted.assemble_json_records(expected),
                reference.assemble_json_records(expected),
            )
            self.assertEqual(resumed.reused_units, 1)
            self.assertEqual(resumed.computed_units, 2)

    def test_heartbeat_fresh_phase_progresses_to_completed_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = ResearchCheckpointStore(root, "segment", {"seed": 7})
            expected = ["steam__segment__0", "steam__segment__1"]
            heartbeat = PhaseHeartbeat(store, expected)
            initial = json.loads(
                (store.phase_root / "heartbeat.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                (
                    initial["status"],
                    initial["completed"],
                    initial["computed"],
                    initial["reused"],
                    initial["pending"],
                ),
                ("running", 0, 0, 0, 2),
            )
            controller = ValidationStopController(root, stop_after_units=None)
            for unit in expected:
                _execute_unit(
                    store=store,
                    expected=expected,
                    unit_id=unit,
                    meter="steam",
                    rows=8,
                    compute=lambda unit=unit: {"unit": unit},
                    controller=controller,
                    heartbeat=heartbeat,
                )
            _finish_phase(store, expected, heartbeat)
            final = json.loads(
                (store.phase_root / "heartbeat.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                (
                    final["status"],
                    final["completed"],
                    final["computed"],
                    final["reused"],
                    final["pending"],
                ),
                ("completed", 2, 2, 0, 0),
            )
            self.assertEqual(final["current_unit"], None)
            self.assertTrue(Path(final["phase_completion_marker"]).exists())
            self.assertFalse(list(store.phase_root.rglob("*.tmp")))

    def test_heartbeat_partial_resume_starts_from_valid_units_and_only_counts_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = ResearchCheckpointStore(root, "bootstrap", {"seed": 7})
            expected = ["steam__draw__0", "steam__draw__1"]
            store.write_unit(expected[0], {"unit": expected[0]})
            heartbeat = PhaseHeartbeat(store, expected)
            initial = json.loads(
                (store.phase_root / "heartbeat.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                (
                    initial["completed"],
                    initial["computed"],
                    initial["reused"],
                    initial["pending"],
                ),
                (1, 0, 1, 1),
            )
            controller = ValidationStopController(root, stop_after_units=None)
            _execute_unit(
                store=store,
                expected=expected,
                unit_id=expected[0],
                meter="steam",
                rows=8,
                compute=lambda: self.fail("valid checkpoint was recomputed"),
                controller=controller,
                heartbeat=heartbeat,
            )
            _execute_unit(
                store=store,
                expected=expected,
                unit_id=expected[1],
                meter="steam",
                rows=8,
                compute=lambda: {"unit": expected[1]},
                controller=controller,
                heartbeat=heartbeat,
            )
            _finish_phase(store, expected, heartbeat)
            final = json.loads(
                (store.phase_root / "heartbeat.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                (
                    final["status"],
                    final["completed"],
                    final["computed"],
                    final["reused"],
                    final["pending"],
                ),
                ("completed", 2, 1, 1, 0),
            )

    def test_heartbeat_rejects_counter_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ResearchCheckpointStore(Path(temporary), "identity", {"seed": 7})
            expected = ["tabpfn-5000"]
            store.write_unit(expected[0], {"unit": expected[0]})
            heartbeat = PhaseHeartbeat(store, expected)
            store.unit_path(expected[0]).unlink()
            with self.assertRaisesRegex(CheckpointError, "counter regressed"):
                heartbeat._write(
                    status="running",
                    current_unit=None,
                    current_meter=None,
                    completion_marker=None,
                )

    def test_heartbeat_full_reuse_completes_without_rewriting_units_or_marker(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = ResearchCheckpointStore(root, "leave_one_building", {"seed": 7})
            expected = ["steam__building__1", "steam__building__2"]
            for unit in expected:
                store.write_unit(unit, {"unit": unit})
            marker = store.complete_phase(expected)
            before_units = {
                unit: store.unit_path(unit).stat().st_mtime_ns for unit in expected
            }
            before_marker = marker.stat().st_mtime_ns
            heartbeat = PhaseHeartbeat(store, expected)
            controller = ValidationStopController(root, stop_after_units=None)
            for unit in expected:
                _execute_unit(
                    store=store,
                    expected=expected,
                    unit_id=unit,
                    meter="steam",
                    rows=8,
                    compute=lambda: self.fail("reused unit was recomputed"),
                    controller=controller,
                    heartbeat=heartbeat,
                )
            _finish_phase(store, expected, heartbeat)
            final = json.loads(
                (store.phase_root / "heartbeat.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                (
                    final["status"],
                    final["completed"],
                    final["computed"],
                    final["reused"],
                    final["pending"],
                ),
                ("completed", 2, 0, 2, 0),
            )
            self.assertEqual(
                before_units,
                {unit: store.unit_path(unit).stat().st_mtime_ns for unit in expected},
            )
            self.assertEqual(before_marker, marker.stat().st_mtime_ns)
            self.assertFalse(list(store.phase_root.rglob("*.tmp")))

    def test_expected_interruption_is_not_written_to_stderr(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "analyze_m5_meter_specific_learner_gap.py"
        ).read_text(encoding="utf-8")
        self.assertIn("EXPECTED_VALIDATION_INTERRUPTION", source)
        self.assertNotIn("print(str(error), file=sys.stderr", source)


if __name__ == "__main__":
    unittest.main()
