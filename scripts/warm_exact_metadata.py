#!/usr/bin/env python3
"""Precompute the exact-metadata disk cache for a feature store, in parallel.

Why this exists: attach_exact_metadata decodes with a ThreadPoolExecutor, and the
decode is CPU-bound enough that the GIL pins it to roughly one core. Measured on the
definition-carrying corpus, a training job spent 3.5 hours in startup and had decoded
83 of ~2550 train feeders (CPULoad 2.30 on a 128-core node) -- about four days to
finish. Three arms launched together each recomputed the same shared cache.

The decode already has a per-feeder disk cache keyed by feeder path, written
atomically (tempfile + os.replace). So the fix does not need to touch the delicate
decode path -- whose threads were chosen deliberately, since an earlier
multiprocessing attempt deadlocked passing large per-variant arrays through pipes.
Instead we warm that cache here with real processes, where each worker writes its
result to disk and returns only a boolean. Nothing large crosses a pipe.

Afterwards every training, evaluation and validation job over the same store gets
cache hits and starts in seconds.

Usage:
    python scripts/warm_exact_metadata.py --root <feature-store> --workers 104
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT), str(PROJECT)]

import torch  # noqa: E402

from gridfm.exact_metadata import _decode_cache_index  # noqa: E402
from gridfm.legacy import data as legacy_data  # noqa: E402

# The v8 contract decodes all nine families. The cache key is
# "-".join(families), so this order must match attach_exact_metadata's
# `requested` order exactly or the warmed files are never found.
FAMILIES = (
    "line", "transformer", "generator", "capacitor", "reactor",
    "load", "pvsystem", "vsource", "storage",
)

_STATE: dict = {}


def _init(root_str: str, cast_float32: bool) -> None:
    root = Path(root_str)
    _STATE["scaler"] = json.loads((root / "feature_scaler.json").read_text())
    _STATE["dtype"] = torch.float32 if cast_float32 else None


def _warm_one(feeder_str: str, cache_dir_str: str):
    """Decode one feeder into the shared disk cache. Returns only small values."""
    feeder = Path(feeder_str)
    try:
        cache = legacy_data.FeederCache(
            feeder, _STATE["scaler"], None, dtype=_STATE["dtype"]
        )
        _, _, hit = _decode_cache_index(0, cache, FAMILIES, Path(cache_dir_str))
        return feeder.name, bool(hit), None
    except Exception as exc:  # keep one bad feeder from killing the sweep
        return feeder.name, False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", 0)) or os.cpu_count() or 1)
    ap.add_argument("--cast-float32", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = args.root.resolve()
    cache_dir = root / ".exact_metadata_cache_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    feeders = sorted(
        p.parent for p in root.rglob("static.pt") if (p.parent / "dynamic.npy").is_file()
    )
    if args.limit:
        feeders = feeders[: args.limit]
    print(f"root={root}", flush=True)
    print(f"feeders={len(feeders)} workers={args.workers} cache={cache_dir}", flush=True)

    started = time.perf_counter()
    hits = misses = failures = 0
    done = 0
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init,
        initargs=(str(root), bool(args.cast_float32)),
    ) as pool:
        futures = [pool.submit(_warm_one, str(f), str(cache_dir)) for f in feeders]
        for future in as_completed(futures):
            name, hit, error = future.result()
            done += 1
            if error is not None:
                failures += 1
                print(f"  FAIL {name}: {error}", flush=True)
            elif hit:
                hits += 1
            else:
                misses += 1
            if done % 100 == 0 or done == len(feeders):
                rate = done / max(time.perf_counter() - started, 1e-9)
                print(
                    f"warm-progress {done}/{len(feeders)} hits={hits} decoded={misses} "
                    f"failed={failures} ({rate:.1f} feeders/s)", flush=True,
                )
    elapsed = time.perf_counter() - started
    print(
        f"WARM_DONE feeders={len(feeders)} hits={hits} decoded={misses} "
        f"failed={failures} elapsed={elapsed:.1f}s", flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
