import unittest
from pathlib import Path

import yaml


class ContractSourceTest(unittest.TestCase):
    def test_model_has_no_target_derived_nominal_or_solver(self):
        text = (Path(__file__).parents[1] / "gridfm" / "model.py").read_text().lower()
        self.assertNotIn("v_nominal", text)
        self.assertNotIn("linalg.solve", text)
        self.assertNotIn("opendss", text)

    def test_config_uses_real_topology_holdout(self):
        cfg = yaml.safe_load(
            (Path(__file__).parents[1] / "configs" / "pf_fraction.yaml").read_text()
        )
        self.assertLess(0, cfg["data"]["train_frac"])
        self.assertLess(cfg["data"]["train_frac"], 1)
        self.assertGreater(cfg["data"]["val_frac"], 0)
        self.assertFalse(cfg["data"]["cast_float32"])


if __name__ == "__main__":
    unittest.main()
