"""gridfm/core must never import the migrated modules as top-level names.

`data`, `masking` and `physics` were migrated out of the old DG_FM_Training tree
into gridfm/core. An absolute `from masking import ...` still resolves whenever
that tree happens to be on sys.path, so the severance can rot silently and only
break on a machine where it is absent.

This is a static check on purpose. The import that survived the migration lived
inside ScenarioDataset.__getitem__, so it fired only when a sample was actually
drawn -- no import-time smoke test and no unit test that stops short of reading a
batch would ever reach it. Walking the AST catches imports at any nesting depth.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "gridfm" / "core"
MIGRATED = {"data", "masking", "physics"}


class TestCoreImports(unittest.TestCase):
    def test_no_absolute_imports_of_migrated_modules(self) -> None:
        offenders = []
        for path in sorted(CORE.glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                # `from masking import x` with level 0 is absolute; level > 0 is
                # the relative form we want.
                if isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module in MIGRATED:
                        offenders.append(f"{path.name}:{node.lineno}: from {node.module} import ...")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in MIGRATED:
                            offenders.append(f"{path.name}:{node.lineno}: import {alias.name}")
        self.assertEqual(offenders, [], "absolute imports of migrated modules:\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
