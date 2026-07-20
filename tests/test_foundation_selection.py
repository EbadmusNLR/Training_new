import math
import unittest

from scripts.train import evaluate_task_lenses, foundation_selection_score


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

    def test_task_lenses_mutate_and_restore_the_live_dataset(self):
        class Dataset:
            mask_cfg = {"mixture": {"random_safe": 1.0}, "p_voltage": 0.3}

        dataset = Dataset()
        original = dataset.mask_cfg
        metrics = evaluate_task_lenses(
            dataset,
            {"pf": ["V"], "param_one": ["Y"], "random": ["V", "Y"]},
            lambda: (
                next(iter(dataset.mask_cfg["mixture"])),
                dataset.mask_cfg.get("p_admittance", 0.0),
            ),
        )
        self.assertEqual(
            metrics,
            {
                "pf": ("pf", 0.0),
                "param_one": ("param_one", 0.0),
                "random": ("random", 0.10),
            },
        )
        self.assertIs(dataset.mask_cfg, original)


if __name__ == "__main__":
    unittest.main()
