from __future__ import annotations

import ast
import subprocess
import unittest
from pathlib import Path

import pandas as pd

from lead import SHIFTS, add_value_change_features


ROOT = Path(__file__).resolve().parents[1]


def tracked_script_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "scripts/*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / path for path in result.stdout.splitlines()]


def is_value_change_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Name) and node.func.id == "add_value_change_features"
    )


def has_required_shifts_arg(node: ast.Call) -> bool:
    return len(node.args) >= 2 or any(
        keyword.arg == "shifts" for keyword in node.keywords
    )


class TestCallArity(unittest.TestCase):
    def test_value_change_calls_pass_shifts(self) -> None:
        failures = []
        for path in tracked_script_paths():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and is_value_change_call(node)
                    and not has_required_shifts_arg(node)
                ):
                    failures.append(f"{path.relative_to(ROOT)}:{node.lineno}")

        self.assertEqual(
            failures,
            [],
            "add_value_change_features calls must pass explicit shifts: "
            + ", ".join(failures),
        )

    def test_value_change_smoke_with_full_offline_shifts(self) -> None:
        df = pd.DataFrame(
            {
                "building_id": [1, 1, 1, 2, 2, 2],
                "timestamp": pd.date_range("2016-01-01", periods=3, freq="h").tolist()
                * 2,
                "meter_reading": [10.0, 12.0, 15.0, 20.0, 18.0, 21.0],
            }
        )

        out = add_value_change_features(df, list(SHIFTS))

        first_shift = SHIFTS[0]
        self.assertIn(f"lag_value_diff_{first_shift}", out.columns)
        self.assertIn(f"lag_value_ratio_{first_shift}", out.columns)
        self.assertEqual(len(out), len(df))


if __name__ == "__main__":
    unittest.main()
