#!/usr/bin/env python3
"""Create one inference checkpoint by arithmetic model-weight averaging."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if len(args.ckpt) < 2:
        ap.error("provide at least two checkpoints")

    checkpoints = [torch.load(path, map_location="cpu", weights_only=False)
                   for path in args.ckpt]
    key_sets = [set(row["model"]) for row in checkpoints]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise SystemExit("model state dictionaries do not have identical keys")

    averaged = {}
    for key in checkpoints[0]["model"]:
        values = [row["model"][key] for row in checkpoints]
        first = values[0]
        if first.is_floating_point() or first.is_complex():
            value = first.clone().div_(len(values))
            for other in values[1:]:
                value.add_(other, alpha=1.0 / len(values))
            averaged[key] = value
        else:
            if any(not torch.equal(first, other) for other in values[1:]):
                raise SystemExit(f"non-floating state differs: {key}")
            averaged[key] = first.clone()

    output = dict(checkpoints[-1])
    output["model"] = averaged
    output.pop("optimizer", None)
    output.pop("scheduler", None)
    output["averaged_checkpoints"] = [str(path.resolve()) for path in args.ckpt]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
