from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PLOTTER = SCRIPTS / "plot_m6_seen_vs_unseen_curves.py"


def load_plotter():
    scripts_dir = str(SCRIPTS)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "plot_m6_seen_vs_unseen_curves_for_test", PLOTTER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {PLOTTER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def arm(index, site, y) -> dict[str, np.ndarray]:
    return {
        "index": np.asarray(index, dtype=np.int64),
        "site": np.asarray(site, dtype=np.int16),
        "y": np.asarray(y, dtype=np.int8),
    }


class TestSameRowGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_plotter()

    @staticmethod
    def full_arm() -> dict[str, np.ndarray]:
        """Two rows per site, one of each class, so every site is scoreable."""
        index = np.arange(32, dtype=np.int64)
        site = np.repeat(np.arange(16, dtype=np.int16), 2)
        y = np.tile(np.array([0, 1], dtype=np.int8), 16)
        return arm(index, site, y)

    def test_identical_arms_pass(self) -> None:
        self.m.assert_same_rows(self.full_arm(), self.full_arm())

    def test_different_rows_are_refused(self) -> None:
        shifted = self.full_arm()
        shifted["index"] = shifted["index"] + 1
        with self.assertRaisesRegex(SystemExit, "same evaluation rows"):
            self.m.assert_same_rows(self.full_arm(), shifted)

    def test_disagreeing_site_ids_are_refused(self) -> None:
        drifted = self.full_arm()
        drifted["site"] = drifted["site"][::-1].copy()
        with self.assertRaisesRegex(SystemExit, "disagree on site_id"):
            self.m.assert_same_rows(self.full_arm(), drifted)

    def test_disagreeing_labels_are_refused(self) -> None:
        drifted = self.full_arm()
        drifted["y"] = 1 - drifted["y"]
        with self.assertRaisesRegex(SystemExit, "disagree on labels"):
            self.m.assert_same_rows(self.full_arm(), drifted)

    def test_missing_site_is_refused(self) -> None:
        """A 4x4 figure with a blank panel would misread as a null result."""
        partial = arm([0, 1], [0, 0], [0, 1])
        with self.assertRaisesRegex(SystemExit, r"expected site_id 0\.\.15"):
            self.m.assert_same_rows(partial, partial)


class TestRestrictToSeenRows(unittest.TestCase):
    """The unseen folds cover every building; only one parity may be plotted."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_plotter()

    @staticmethod
    def wide_arm() -> dict[str, np.ndarray]:
        """Row ids are sparse, as real ones are, so a gap can be probed."""
        return {
            "index": np.arange(0, 20, 2, dtype=np.int64),
            "site": np.arange(10, dtype=np.int16),
            "score": np.arange(10, dtype=np.float32) / 10,
        }

    def test_kept_rows_carry_their_own_values(self) -> None:
        keep = np.array([2, 8, 14], dtype=np.int64)
        cut = self.m.restrict_to(self.wide_arm(), keep)
        np.testing.assert_array_equal(cut["index"], keep)
        np.testing.assert_array_equal(cut["site"], np.array([1, 4, 7]))
        np.testing.assert_allclose(cut["score"], np.array([0.1, 0.4, 0.7]), atol=1e-7)

    def test_row_absent_from_the_wide_arm_is_refused(self) -> None:
        with self.assertRaisesRegex(SystemExit, "do not share a row space"):
            self.m.restrict_to(self.wide_arm(), np.array([2, 5], dtype=np.int64))

    def test_row_past_the_end_is_refused(self) -> None:
        with self.assertRaisesRegex(SystemExit, "not all present"):
            self.m.restrict_to(self.wide_arm(), np.array([18, 20], dtype=np.int64))

    def test_unsorted_row_ids_are_refused(self) -> None:
        with self.assertRaisesRegex(SystemExit, "sorted and unique"):
            self.m.restrict_to(self.wide_arm(), np.array([8, 2], dtype=np.int64))


class TestCurveThinning(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_plotter()

    def test_short_curves_are_untouched(self) -> None:
        x = np.linspace(0, 1, 10)
        thinned_x, thinned_y = self.m._compressed(x, x, max_points=1_200)
        np.testing.assert_array_equal(thinned_x, x)
        np.testing.assert_array_equal(thinned_y, x)

    def test_long_curves_keep_both_endpoints(self) -> None:
        x = np.linspace(0, 1, 50_000)
        thinned_x, _ = self.m._compressed(x, x, max_points=1_200)
        self.assertLessEqual(len(thinned_x), 1_200)
        self.assertEqual(thinned_x[0], x[0])
        self.assertEqual(thinned_x[-1], x[-1])


if __name__ == "__main__":
    unittest.main()
