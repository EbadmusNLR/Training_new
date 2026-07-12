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

    def test_tree_current_decoder_is_not_a_voltage_solver(self):
        text = (Path(__file__).parents[1] / "gridfm" / "tree_current.py").read_text().lower()
        self.assertNotIn("linalg", text)
        self.assertNotIn("v_init", text)
        self.assertNotIn(".dv", text)
        self.assertNotIn("opendss", text)
        self.assertIn("subtree", text)

    def test_hybrid_current_decoder_is_local_and_solver_free(self):
        text = (Path(__file__).parents[1] / "gridfm" / "hybrid_current.py").read_text().lower()
        self.assertNotIn("linalg", text)
        self.assertNotIn("opendss", text)
        self.assertIn("decode_currents", text)
        self.assertNotIn('"line"', text.split("safe_physics_stores", 1)[1].split(")", 1)[0])


if __name__ == "__main__":
    unittest.main()
