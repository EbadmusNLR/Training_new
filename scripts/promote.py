#!/usr/bin/env python3
"""Verify and copy the selected final artifact into one self-contained directory."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from select_champion import sha256_file


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    receipt = read(args.selection)
    if receipt.get("test_metrics_read") is not False:
        raise SystemExit("invalid selection receipt")
    selected = receipt["selected"]
    checkpoint = Path(selected["checkpoint"])
    if sha256_file(checkpoint) != selected["checkpoint_sha256"]:
        raise SystemExit("selected checkpoint hash no longer matches receipt")
    run_dir = checkpoint.parent
    sources = {
        "checkpoint.pt": checkpoint,
        "config_used.yaml": run_dir / "config_used.yaml",
        "split_manifest.json": run_dir / "split_manifest.json",
        "unseen_validation.json": Path(selected["report"]),
        "test_direct.json": run_dir / "test_direct.json",
        "test_tree.json": run_dir / "test_tree.json",
        "seen_tree.json": run_dir / "seen_tree_final.json",
        "current_diagnostics_test.json": run_dir / "current_diagnostics_test.json",
        "selection.json": args.selection,
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing final artifacts: {missing}")
    if read(sources["unseen_validation.json"]).get("split") != "unseen":
        raise SystemExit("selected report is not unseen validation")
    if read(sources["test_tree.json"]).get("split") != "test":
        raise SystemExit("final structural report is not test")

    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        shutil.copy2(source, args.output / name)

    reports = {name: read(path) for name, path in sources.items() if name.endswith(".json")}
    manifest = {
        "model": "EdgeStateGridFM H384",
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "inference": {"tree_line": True, "kcl_vsource": True, "power_flow_solver": False},
        "seen": reports["seen_tree.json"],
        "unseen_validation": reports["unseen_validation.json"],
        "test": reports["test_tree.json"],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
