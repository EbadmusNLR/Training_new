#!/usr/bin/env python3
"""Precompute + cache the FeederCache positional encoding for every feeder.

build_strict_datasets constructs a FeederCache per feeder in the main process
(serial); for large SMART-DS feeders the PE dominates (~2s each -> ~35 min for
1000). The PE is per-topology and cached to `pe_cache_v1.pt`, so precomputing it
in parallel here makes training startup a fast cache load. Idempotent: skips
feeders whose cache already exists.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _one(args):
    feeder_dir, scaler = args
    from gridfm import legacy  # noqa: F401 -- adds DG_FM_Training to sys.path
    import data as D
    cache = Path(feeder_dir) / "pe_cache_v1.pt"
    if cache.is_file():
        return feeder_dir, "cached"
    try:
        D.FeederCache(feeder_dir, scaler, None)  # builds + writes pe cache
        return feeder_dir, "ok"
    except Exception as e:  # noqa
        return feeder_dir, f"FAIL: {type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()
    root = Path(args.root)
    scaler = json.loads((root / "feature_scaler.json").read_text())
    feeders = sorted(os.path.dirname(p) for p in glob.glob(str(root / "*" / "static.pt")))
    print(f"precomputing PE for {len(feeders)} feeders on {args.workers} workers", flush=True)
    ok = fail = cached = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_one, (f, scaler)) for f in feeders]
        for i, fut in enumerate(as_completed(futs), 1):
            _, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "cached":
                cached += 1
            else:
                fail += 1
                print(status, flush=True)
            if i % 100 == 0:
                print(f"  {i}/{len(feeders)}  ok={ok} cached={cached} fail={fail}", flush=True)
    print(f"done: ok={ok} cached={cached} fail={fail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
