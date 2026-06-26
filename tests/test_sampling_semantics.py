from __future__ import annotations

import unittest

import pandas as pd

from lead import downsample_indices


class TestSamplingSemantics(unittest.TestCase):
    def test_downsample_indices_duplicates_positive_block_twice(self) -> None:
        y = pd.Series(
            [0, 0, 1, 0, 1, 0, 0, 0],
            index=["n0", "n1", "p0", "n2", "p1", "n3", "n4", "n5"],
        )

        ds_idx = downsample_indices(y)
        n_pos = int(y.sum())
        pos_idx = y.index[y == 1].to_numpy()

        self.assertEqual(len(ds_idx), 4 * n_pos)
        self.assertEqual(ds_idx[n_pos : 2 * n_pos].tolist(), pos_idx.tolist())
        self.assertEqual(ds_idx[3 * n_pos :].tolist(), pos_idx.tolist())
        self.assertEqual(y.loc[ds_idx].sum(), 2 * n_pos)
        self.assertEqual(float(y.loc[ds_idx].mean()), 0.5)


if __name__ == "__main__":
    unittest.main()
