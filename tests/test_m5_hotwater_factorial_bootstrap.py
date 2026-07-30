import math
import unittest

import numpy as np

from scripts.analyze_m5_hotwater_label_role_factorial import factor_effect, pair_auc


class TestM5HotwaterFactorialBootstrap(unittest.TestCase):
    def test_pair_auc_is_tie_aware_and_rejects_empty_class(self) -> None:
        self.assertEqual(pair_auc(np.array([0.5]), np.array([0.5])), 0.5)
        self.assertTrue(math.isnan(pair_auc(np.array([]), np.array([0.5]))))

    def test_factorial_effects_keep_the_interaction_sign(self) -> None:
        effects = factor_effect(
            {
                (False, False): 0.0,
                (False, True): 1.0,
                (True, False): 2.0,
                (True, True): 6.0,
            }
        )
        self.assertEqual(effects["positive_support_main_effect"], 3.5)
        self.assertEqual(effects["negative_support_main_effect"], 2.5)
        self.assertEqual(effects["positive_x_negative_interaction"], 3.0)


if __name__ == "__main__":
    unittest.main()
