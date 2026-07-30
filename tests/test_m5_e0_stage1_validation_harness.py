from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestM5E0Stage1ValidationHarness(unittest.TestCase):
    def test_launcher_is_foreground_bounded_and_without_auto_kill(self) -> None:
        source = (ROOT / "scripts" / "run_m5_e0_stage1_validation.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '"-u", "scripts/analyze_m5_meter_specific_learner_gap.py"', source
        )
        self.assertIn('"--bootstrap-draws", "$BootstrapDraws"', source)
        self.assertIn('"--loo-buildings", "$LooBuildings"', source)
        self.assertIn('"--segment-draws", "$SegmentDraws"', source)
        self.assertNotIn("Start-Job", source)
        self.assertNotIn("Start-Process", source)
        self.assertNotIn("Wait-Process -Timeout", source)

    def test_evidence_suite_runs_interruption_resume_and_reuse_in_foreground(
        self,
    ) -> None:
        source = (ROOT / "scripts" / "run_m5_e0_stage1_validation.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("EvidenceSuite", source)
        self.assertIn("validation-stop-after-units", source)
        self.assertIn("EXPECTED_VALIDATION_INTERRUPTION", source)
        self.assertIn("checkpoint_census", source)
        self.assertIn('$ErrorActionPreference = "Continue"', source)
        self.assertIn("run2-resume", source)
        self.assertIn("run3-reuse", source)
        self.assertIn(
            "Run 2 changed or removed completed Run 1 checkpoint SHA256 values", source
        )
        self.assertIn("Run 2 did not compute only missing units", source)
        self.assertIn("$summary.reused_units -ne $afterRun3.Count", source)
        self.assertIn("Get-CheckpointModificationTimes", source)
        self.assertIn("Run 3 changed checkpoint modification times", source)
        self.assertIn("stderr log is not empty", source)

    def test_each_run_saves_an_atomic_provenance_checked_heartbeat_snapshot(
        self,
    ) -> None:
        source = (ROOT / "scripts" / "run_m5_e0_stage1_validation.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("Save-HeartbeatSnapshot", source)
        self.assertIn('"$runId.heartbeat.json"', source)
        self.assertIn("snapshot_timestamp_utc", source)
        self.assertIn("source_heartbeat_sha256", source)
        self.assertIn("source_provenance_sha256", source)
        self.assertIn("Heartbeat snapshot refused: no source heartbeat exists", source)
        self.assertIn("Heartbeat snapshot refused: invalid JSON", source)
        self.assertIn("Heartbeat snapshot refused: provenance mismatch", source)
        self.assertIn("Heartbeat snapshot already exists", source)
        self.assertIn("Run 1 heartbeat snapshot", source)
        self.assertIn("Run 2 heartbeat snapshot", source)
        self.assertIn("Run 3 heartbeat snapshot", source)
        self.assertIn("NewGuid", source)
        self.assertIn(
            "ForEach-Object { [int]$_.heartbeat.computed } | Measure-Object -Sum",
            source,
        )
        self.assertNotIn("Measure-Object -Property { $_.heartbeat", source)
        self.assertNotIn("Start-Job", source)
        self.assertNotIn("Start-Process", source)

    def test_probe_is_non_scientific_and_has_no_process_timeout(self) -> None:
        source = (
            ROOT / "scripts" / "probe_foreground_persistent_session.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--seconds", source)
        self.assertIn("heartbeat.json", source)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("data/processed", source)


if __name__ == "__main__":
    unittest.main()
