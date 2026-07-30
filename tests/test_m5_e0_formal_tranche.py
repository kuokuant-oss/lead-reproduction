from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts._research_checkpoint import ResearchCheckpointStore
from scripts.analyze_m5_meter_specific_learner_gap import (
    FORMAL_BOOTSTRAP_DRAWS_PER_METER,
    FORMAL_TRANCHE_DRAWS_PER_METER,
    METER_NAMES,
    formal_bootstrap_manifest,
    formal_tranche_units,
    _validate_formal_args,
)


ROOT = Path(__file__).resolve().parents[1]


class TestM5E0FormalTranche(unittest.TestCase):
    def test_manifest_is_exact_formal_universe_and_round_robin(self) -> None:
        manifest = formal_bootstrap_manifest()
        self.assertEqual(len(manifest), 4 * FORMAL_BOOTSTRAP_DRAWS_PER_METER)
        self.assertEqual(
            manifest[:8],
            [
                "electricity__draw__0",
                "chilledwater__draw__0",
                "steam__draw__0",
                "hotwater__draw__0",
                "electricity__draw__1",
                "chilledwater__draw__1",
                "steam__draw__1",
                "hotwater__draw__1",
            ],
        )

    def test_selects_smallest_missing_per_meter_without_counting_reuse(self) -> None:
        manifest = formal_bootstrap_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            store = ResearchCheckpointStore(
                Path(temporary), "bootstrap", {"mode": "test"}
            )
            for unit in (
                "electricity__draw__0",
                "electricity__draw__1",
                "steam__draw__0",
            ):
                store.write_unit(unit, {"ok": True})
            selected = formal_tranche_units(store, manifest, 2)
        self.assertEqual(
            selected,
            [
                "electricity__draw__2",
                "chilledwater__draw__0",
                "steam__draw__1",
                "hotwater__draw__0",
                "electricity__draw__3",
                "chilledwater__draw__1",
                "steam__draw__2",
                "hotwater__draw__1",
            ],
        )

    def test_formal_guards_and_scope_are_explicit(self) -> None:
        source = (
            ROOT / "scripts" / "analyze_m5_meter_specific_learner_gap.py"
        ).read_text(encoding="utf-8")
        self.assertIn('FORMAL_AUTHORIZATION_TOKEN = "AUTHORIZE_E0_FORMAL_RUN"', source)
        self.assertIn("--authorization-token", source)
        self.assertIn("--resume", source)
        self.assertIn("--max-new-draws-per-meter", source)
        self.assertIn("formal tranche 1 requires --max-new-draws-per-meter 42", source)
        self.assertIn(
            "formal tranche permits only identity, base_metrics, and bootstrap", source
        )
        self.assertIn("if len(completed) == len(manifest):", source)
        self.assertIn("PARTIAL_BOOTSTRAP_TRANCHE", source)
        self.assertIn("FORMAL_E0 roots must be isolated", source)
        self.assertNotIn("timeout=", source)
        self.assertNotIn("Start-Job", source)

    def test_formal_token_resume_and_limit_guards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = dict(
                authorization_token="AUTHORIZE_E0_FORMAL_RUN",
                resume=True,
                max_new_draws_per_meter=42,
                validation_mode=False,
                validation_stop_after_units=None,
                phase=None,
                output_root=root / "output",
                checkpoint_root=root / "checkpoints",
                log_root=root / "logs",
            )
            _validate_formal_args(SimpleNamespace(**valid))
            for key, value in (
                ("authorization_token", None),
                ("authorization_token", "bad"),
                ("resume", False),
                ("max_new_draws_per_meter", 41),
            ):
                invalid = {**valid, key: value}
                with self.assertRaises((PermissionError, ValueError)):
                    _validate_formal_args(SimpleNamespace(**invalid))

    def test_tracked_launcher_is_visible_safe_and_tranche_bounded(self) -> None:
        source = (ROOT / "scripts" / "run_m5_e0_formal_tranche.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("--formal-preflight", source)
        self.assertIn("--formal", source)
        self.assertIn("--authorization-token", source)
        self.assertIn("--resume", source)
        self.assertIn('"--max-new-draws-per-meter", "42"', source)
        self.assertIn("formal_checkpoints", source)
        self.assertIn("formal_logs", source)
        self.assertIn("Start-Transcript", source)
        self.assertIn(".venv\\Scripts\\python.exe", source)
        self.assertIn('$ErrorActionPreference = "Continue"', source)
        self.assertNotIn("Start-Job", source)
        self.assertNotIn("Start-Process", source)
        self.assertNotIn("-Timeout", source)

    def test_tranche_constant_is_exactly_authorized_limit(self) -> None:
        self.assertEqual(FORMAL_TRANCHE_DRAWS_PER_METER, 42)
        self.assertEqual(
            tuple(METER_NAMES.values()),
            ("electricity", "chilledwater", "steam", "hotwater"),
        )


if __name__ == "__main__":
    unittest.main()
