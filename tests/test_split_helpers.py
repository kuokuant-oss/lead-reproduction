from __future__ import annotations

import unittest

import pandas as pd

from lead import leave_site_out_mask


class TestSplitHelpers(unittest.TestCase):
    def test_leave_site_out_mask_uses_explicit_site_ids(self) -> None:
        frame = pd.DataFrame({"site_id": [1, 2, 4, 9, "Swan"]})

        mask = leave_site_out_mask(frame, [4, 9, "Swan"])

        self.assertEqual(mask.tolist(), [False, False, True, True, True])


if __name__ == "__main__":
    unittest.main()
