import math
import unittest

from scripts.train import foundation_selection_score


class FoundationSelectionTest(unittest.TestCase):
    def test_zero_denominator_raw_family_is_not_required(self):
        metrics = {
            "pf": {"V_wape_pct": 1.0, "Ibus_wape_pct": 6.0},
            "se_known": {"V_wape_pct": 1.5, "Ibus_wape_pct": 7.0},
            "param_one": {
                "Y_wape_pct": 0.8,
                "field_storage_Ystorage_r_tri_feat_scale_wape_pct": 0.1,
                "field_transformer_Yxfmr_r_tri_feat_scale_wape_pct": 2.5,
            },
            "injection": {
                "Icomp_wape_pct": 0.5,
                "field_load_Icomp_r_feat_scale_wape_pct": 1.2,
            },
        }
        self.assertEqual(foundation_selection_score(metrics), 7.0)

    def test_missing_required_task_fails_closed(self):
        self.assertTrue(math.isinf(foundation_selection_score({"pf": {}})))


if __name__ == "__main__":
    unittest.main()
