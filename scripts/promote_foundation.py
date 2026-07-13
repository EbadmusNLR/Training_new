#!/usr/bin/env python3
"""Package a hash-pinned foundation checkpoint and its multi-task reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_reports(source: Path, target: Path, split: str) -> None:
    required = (
        "pf.json", "pf_tree.json", "se_known.json", "param_one.json",
        "injection.json", "random_safe.json", "random.json",
    )
    target.mkdir(parents=True, exist_ok=True)
    for name in required:
        path = source / name
        if not path.is_file():
            raise SystemExit(f"missing report: {path}")
        row = json.loads(path.read_text())
        if row.get("split") != split:
            raise SystemExit(f"{path}: expected split={split}")
        shutil.copy2(path, target / name)
    scorecard = source / "scorecard.json"
    if split == "unseen" and scorecard.is_file():
        shutil.copy2(scorecard, target / scorecard.name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--seen-reports", type=Path, required=True)
    ap.add_argument("--test-reports", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    receipt = json.loads(args.selection.read_text())
    if receipt.get("test_metrics_read") is not False:
        raise SystemExit("invalid selection receipt")
    selected = receipt["selected"]
    checkpoint = Path(selected["checkpoint"])
    if digest(checkpoint) != selected["checkpoint_sha256"]:
        raise SystemExit("checkpoint hash differs from selection receipt")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, args.output / "checkpoint.pt")
    for name in ("config_used.yaml", "split_manifest.json"):
        path = checkpoint.parent / name
        if not path.is_file():
            raise SystemExit(f"missing checkpoint companion: {path}")
        shutil.copy2(path, args.output / name)
    unseen = Path(selected["scorecard"]).parent
    copy_reports(unseen, args.output / "unseen", "unseen")
    copy_reports(args.seen_reports, args.output / "seen", "seen")
    copy_reports(args.test_reports, args.output / "test", "test")
    shutil.copy2(args.selection, args.output / "selection.json")
    manifest = {
        "model": "EdgeStateGridFM H384 role-balanced foundation",
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "selection_contract": receipt["selection_contract"],
        "unseen_checks_pct": selected["checks_pct"],
        "physics_contract": "Ibus + Icomp = YV; KCL sums Ibus",
        "inference": {
            "external_power_flow_solver": False,
            "slack_voltage_clamped": True,
            "tree_line_current_available": True,
            "stable_Yh_shunt_available": True,
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
