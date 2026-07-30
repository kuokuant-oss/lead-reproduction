from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import run_m5_tabpfn_canonical_factorial_sentinel as protocol


class TestM5TabPFNCanonicalFactorialProtocol(unittest.TestCase):
    def test_protocol_is_not_low_memory_and_retains_engineering_diagnostics(
        self,
    ) -> None:
        source = Path(protocol.__file__).read_text(encoding="utf-8")
        self.assertIn('"fit_mode": "fit_preprocessors"', source)
        self.assertIn('"memory_saving_mode": False', source)
        self.assertEqual(protocol.QUERY_BATCH_SIZE, 352)
        self.assertEqual(protocol.TOLERANCES["probability_mae_max"], 1e-4)
        self.assertEqual(protocol.TOLERANCES["probability_max_abs_max"], 0.005)
        self.assertEqual(protocol.TOLERANCES["spearman_min"], 0.99999)
        self.assertEqual(protocol.TOLERANCES["primary_estimand_abs_delta_max"], 0.002)

    def test_lifecycle_has_three_live_and_two_fresh_reload_predictions(self) -> None:
        self.assertEqual(
            protocol.PREDICTION_STAGES,
            (
                "R1_fit_gpu",
                "R2_live_gpu_repeat",
                "R3_live_gpu_repeat",
                "R5_same_process_gpu_reload",
                "R6_fresh_gpu_reload_1",
                "R7_fresh_gpu_reload_2",
            ),
        )

    def test_stage_artifacts_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, _ = protocol.stage_paths(root, "R1_fit_gpu")
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"already complete")
            with self.assertRaises(FileExistsError):
                protocol.require_new_stage(root, "R1_fit_gpu")


if __name__ == "__main__":
    unittest.main()
