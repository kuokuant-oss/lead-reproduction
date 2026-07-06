from __future__ import annotations

import json
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_JSON = ROOT / "data/processed/m6_phaseD_50_50_full_models_timestamp_merge.json"
REPORT = ROOT / "docs/reports/m5-foundation-vs-gbdt.md"


class TestM5TimestampMergeRegime(unittest.TestCase):
    def test_primary_json_uses_timestamp_merge(self) -> None:
        data = json.loads(PRIMARY_JSON.read_text(encoding="utf-8"))

        self.assertEqual(data["value_change_regime"], "timestamp_merge")
        for axis in ("in_domain", "label_scarcity", "minimal_fe", "site_transfer"):
            self.assertEqual(
                data["axes"][axis]["split"]["value_change_regime"],
                "timestamp_merge",
            )

    def test_report_primary_matrix_uses_timestamp_merge_json(self) -> None:
        report = REPORT.read_text(encoding="utf-8")

        self.assertIn("Feature basis**：`timestamp_merge`", report)
        self.assertIn("Feature regime | `timestamp_merge`", report)
        self.assertIn(
            "data/processed/m6_phaseD_50_50_full_models_timestamp_merge.json",
            report,
        )

    def test_confusion_matrix_figures_do_not_clip_outer_text(self) -> None:
        data = json.loads(PRIMARY_JSON.read_text(encoding="utf-8"))

        for figure_path in data["figures"].values():
            with self.subTest(figure_path=figure_path):
                path = ROOT / figure_path
                with Image.open(path).convert("RGB") as image:
                    width, height = image.size
                    right_edge_nonwhite = sum(
                        1
                        for x in range(width - 3, width)
                        for y in range(height)
                        if image.getpixel((x, y)) != (255, 255, 255)
                    )
                    top_edge_nonwhite = sum(
                        1
                        for x in range(width)
                        for y in range(3)
                        if image.getpixel((x, y)) != (255, 255, 255)
                    )

                self.assertEqual(right_edge_nonwhite, 0)
                self.assertEqual(top_edge_nonwhite, 0)


if __name__ == "__main__":
    unittest.main()
