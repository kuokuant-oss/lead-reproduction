from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden_metrics.json"
M3_50_50 = ROOT / "docs" / "metrics" / "m3-50-50-ensemble.json"
M3_REPORT = ROOT / "docs" / "reports" / "m3-report.md"
M4_REPORT = ROOT / "docs" / "reports" / "m4-evaluation-report.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def table_section(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def parse_m4_gate_auc_rows(section: str) -> dict[str, float]:
    rows: dict[str, float] = {}
    for line in section.splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 3 or cells[0] == "Result":
            continue
        if re.fullmatch(r"\d+\.\d+", cells[1]):
            rows[cells[0]] = float(cells[1])
    return rows


def parse_m3_table_2_1_auc_rows(section: str) -> dict[str, float]:
    rows: dict[str, float] = {}
    for line in section.splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 7 or cells[0] == "Split":
            continue
        if re.fullmatch(r"\d+\.\d+", cells[3]):
            rows[f"{cells[0]}|{cells[1]}"] = float(cells[3])
    return rows


class TestReportMetricConsistency(unittest.TestCase):
    def test_m4_regression_gate_auc_matches_golden_display_values(self) -> None:
        golden = json.loads(read(GOLDEN))["metrics"]
        section = table_section(read(M4_REPORT), "## 1.2 Regression Gate", "\n---\n")
        rows = parse_m4_gate_auc_rows(section)

        expected = {
            "M3.2 LightGBM 80/20 offline": golden["m3_2_lightgbm_80_20_offline_auc"][
                "display_value"
            ],
            "M3.4 4-model ensemble 80/20 offline": golden[
                "m3_4_ensemble_80_20_offline_auc"
            ]["display_value"],
            "50/50 offline ensemble": golden["m3_50_50_offline_ensemble_auc"][
                "display_value"
            ],
            "50/50 causal ensemble": golden["m3_50_50_causal_ensemble_auc"][
                "display_value"
            ],
            "Site-held-out ensemble": golden["m3_site_heldout_ensemble_auc"][
                "display_value"
            ],
            "Steam meter": golden["m3_steam_meter_auc"]["display_value"],
        }
        self.assertEqual(rows, expected)

    def test_m3_table_2_1_auc_matches_50_50_metrics_json(self) -> None:
        metrics = json.loads(read(M3_50_50))["metrics"]
        section = table_section(read(M3_REPORT), "| Split |", "表 2.1")
        rows = parse_m3_table_2_1_auc_rows(section)

        self.assertEqual(
            rows["50/50 mod2|Offline batch labeling"],
            metrics["offline"]["ensemble_auc"],
        )
        self.assertEqual(
            rows["50/50 mod2|Past-only"],
            metrics["causal"]["ensemble_auc"],
        )


if __name__ == "__main__":
    unittest.main()
