from __future__ import annotations

import unittest

import lead


FROZEN_PUBLIC_API = [
    "ROOT",
    "M3",
    "PROC",
    "RANDOM_STATE",
    "DOWNSAMPLE_SEEDS",
    "MODEL_SEEDS",
    "SHUFFLE_SEEDS",
    "BASELINE_FEATURE_COLS",
    "BUILDING_META_FEATURE_COLS",
    "CYCLIC_FEATURE_COLS",
    "M3_3_EXTRA_FEATURE_COLS",
    "WEATHER_LAG_BASE_COLS",
    "WEATHER_WINDOWS",
    "SHIFTS",
    "PAST_SHIFTS",
    "FUTURE_SHIFTS",
    "load_m3_frame",
    "load_bdg2_frame",
    "add_value_change_features",
    "split_mask",
    "assert_no_building_overlap",
    "leave_site_out_mask",
    "downsample_indices",
    "classification_metrics",
    "write_json_with_provenance",
]


class TestPublicApi(unittest.TestCase):
    def test_src_lead_all_matches_m4_5_frozen_surface(self) -> None:
        self.assertEqual(lead.__all__, FROZEN_PUBLIC_API)
        for name in FROZEN_PUBLIC_API:
            self.assertTrue(hasattr(lead, name), f"lead.{name} is not exported")


if __name__ == "__main__":
    unittest.main()
