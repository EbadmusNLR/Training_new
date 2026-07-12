import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "select_champion.py"
SPEC = importlib.util.spec_from_file_location("select_champion", SCRIPT)
SELECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECT)


class SelectChampionTest(unittest.TestCase):
    def report(self, root, name, checkpoint, voltage, current, split="unseen"):
        path = root / name
        path.write_text(json.dumps({
            "split": split,
            "tree_line": True,
            "kcl_vsource": True,
            "checkpoint": str(checkpoint),
            "V_wape_pct": voltage,
            "Ibus_wape_pct": current,
        }))
        return path

    def test_selects_only_by_declared_validation_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt_a, ckpt_b = root / "a.pt", root / "b.pt"
            ckpt_a.write_bytes(b"a")
            ckpt_b.write_bytes(b"b")
            a = self.report(root, "a.json", ckpt_a, 2.0, 7.0)
            b = self.report(root, "b.json", ckpt_b, 1.0, 10.0)
            output = root / "selection.json"
            argv = sys.argv
            try:
                sys.argv = [str(SCRIPT), "--report", str(a), "--report", str(b),
                            "--output", str(output)]
                self.assertEqual(SELECT.main(), 0)
            finally:
                sys.argv = argv
            selected = json.loads(output.read_text())
            self.assertEqual(selected["selected"]["checkpoint"], str(ckpt_b))
            self.assertFalse(selected["test_metrics_read"])

    def test_rejects_test_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "a.pt"
            checkpoint.write_bytes(b"a")
            report = self.report(root, "test.json", checkpoint, 1.0, 1.0, split="test")
            with self.assertRaises(SystemExit):
                SELECT.load_report(report)


if __name__ == "__main__":
    unittest.main()
