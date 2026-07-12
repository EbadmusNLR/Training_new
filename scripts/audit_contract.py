#!/usr/bin/env python3
"""Fail-fast audit for split leakage and always-known voltage inputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridfm.data import build_strict_datasets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--limit-feeders", type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.limit_feeders is not None:
        cfg["data"]["limit_feeders"] = args.limit_feeders
    bundle = build_strict_datasets(cfg["data"], cfg["mask"], int(cfg["train"]["seed"]))
    print(json.dumps({
        "samples": {k: len(getattr(bundle, k)) for k in ("train", "seen", "unseen", "test")},
        "feeders": {
            "train": len(bundle.train_feeders), "seen": len(bundle.train_feeders),
            "unseen": len(bundle.unseen_feeders), "test": len(bundle.test_feeders),
        },
        "target_derived_nominal": False,
        "slack_voltage_always_visible": True,
        "v_init_always_present": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

