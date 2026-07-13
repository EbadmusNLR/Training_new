#!/usr/bin/env python3
"""Select a broad checkpoint from unseen-topology foundation scorecards only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASK_REPORTS = (
    "pf", "pf_tree", "se_known", "param_one", "injection",
    "random_safe", "random",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate(scorecard_path: Path) -> dict:
    root = scorecard_path.parent
    scorecard = json.loads(scorecard_path.read_text())
    reports = {}
    checkpoint = None
    for task in TASK_REPORTS:
        path = root / f"{task}.json"
        if not path.is_file():
            raise SystemExit(f"{scorecard_path}: missing {path.name}")
        report = json.loads(path.read_text())
        if report.get("split") != "unseen":
            raise SystemExit(f"{path}: selection requires split=unseen")
        current = report.get("checkpoint")
        if checkpoint is None:
            checkpoint = current
        elif current != checkpoint:
            raise SystemExit(f"{path}: checkpoint differs inside report set")
        reports[task] = str(path.resolve())
    ckpt = Path(checkpoint).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint: {ckpt}")
    checks = {key: float(value) for key, value in scorecard["checks_pct"].items()}
    # Arbitrary simultaneous deletion is retained as a stress diagnostic but
    # is underdetermined from one snapshot. Identifiable random_safe is part of
    # selection together with every core forward/inverse task.
    core = {
        key: value for key, value in checks.items()
        if not key.startswith("random_") or key.startswith("random_safe_")
    }
    maximum = max(core.values())
    mean = sum(core.values()) / len(core)
    return {
        "scorecard": str(scorecard_path.resolve()),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256(ckpt),
        "reports": reports,
        "checks_pct": checks,
        "core_max_pct": maximum,
        "core_mean_pct": mean,
        "selection_score": maximum + mean,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scorecard", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if len(args.scorecard) < 2:
        ap.error("provide at least two unseen scorecards")
    rows = sorted((candidate(path) for path in args.scorecard),
                  key=lambda row: row["selection_score"])
    payload = {
        "selection_contract": (
            "unseen only; minimize core_max_pct + core_mean_pct; "
            "random_safe included, underdetermined random stress excluded"
        ),
        "test_metrics_read": False,
        "selected": rows[0],
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
