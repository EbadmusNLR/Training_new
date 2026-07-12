#!/usr/bin/env python3
"""Select one checkpoint using unseen-topology structural-current reports only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_report(path: Path) -> dict:
    row = json.loads(path.read_text())
    if row.get("split") != "unseen":
        raise SystemExit(f"{path}: expected split=unseen")
    if row.get("tree_line") is not True or row.get("kcl_vsource") is not True:
        raise SystemExit(f"{path}: expected tree_line=true and kcl_vsource=true")
    if not row.get("checkpoint"):
        raise SystemExit(f"{path}: missing checkpoint")
    for key in ("V_wape_pct", "Ibus_wape_pct"):
        if key not in row:
            raise SystemExit(f"{path}: missing {key}")
    return row


def score(row: dict) -> float:
    # Current is the binding target; voltage receives a strong guardrail.
    return float(row["Ibus_wape_pct"]) + 5.0 * float(row["V_wape_pct"])


def checkpoint_path(report_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"{report_path}: checkpoint does not exist: {path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if len(args.report) < 2:
        ap.error("provide at least two validation reports")

    candidates = []
    for report_path in args.report:
        row = load_report(report_path)
        checkpoint = checkpoint_path(report_path, row["checkpoint"])
        candidates.append({
            "report": str(report_path.resolve()),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "V_wape_pct": float(row["V_wape_pct"]),
            "Ibus_wape_pct": float(row["Ibus_wape_pct"]),
            "score": score(row),
        })
    candidates.sort(key=lambda row: row["score"])
    payload = {
        "selection_contract": "unseen tree-line current; score=Ibus_WAPE+5*V_WAPE",
        "test_metrics_read": False,
        "selected": candidates[0],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
