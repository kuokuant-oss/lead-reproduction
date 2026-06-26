from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_m5_phaseC_tabpfn_spike.py"


class TestM5TabPFNSpike(unittest.TestCase):
    def test_runner_imports_frozen_lead_api_helpers(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "lead":
                imports.update(alias.name for alias in node.names)

        self.assertGreaterEqual(
            imports,
            {
                "BASELINE_FEATURE_COLS",
                "SHIFTS",
                "add_value_change_features",
                "assert_no_building_overlap",
                "classification_metrics",
                "downsample_indices",
                "load_m3_frame",
                "write_json_with_provenance",
            },
        )

    def test_runner_uses_m3_split_sample_and_row_offset_path(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('building_id"] % 5 == 4', source)
        self.assertIn("downsample_indices(y_train_full)", source)
        self.assertIn('VALUE_CHANGE_REGIME = "row_offset"', source)
        self.assertIn("list(SHIFTS)", source)

    def test_runner_supports_local_checkpoint_without_token(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--model-path", source)
        self.assertIn("TABPFN_MODEL_CACHE_DIR", source)
        self.assertIn("local_checkpoint_available", source)
        self.assertIn("not_run_missing_weights_and_token", source)
        self.assertIn(
            "model_path=model_path if local_checkpoint_available else None", source
        )


if __name__ == "__main__":
    unittest.main()
