#!/usr/bin/env python3
"""Fit one leakage-safe global feature scaler from pu scenario stores.

The fitter uses deterministic bounded sketches per feeder so SMART-DS-sized
corpora do not materialize every matrix entry in RAM. Line and triplex-line rows
share one physical Line coordinate, removing the old baseline-JSON dependency.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT), str(PROJECT)]

from datakit.core.hetero_graph import COMPONENTS, FIXED_CONDUCTORS  # noqa: E402
from datakit.core.scenario_store import FeederScenarios  # noqa: E402
from gridfm.featurizing import (  # noqa: E402
    FEAT_EPS, SCALE_FLOOR, SCENARIO_Y_FIELDS, SPECS,
    save_scaler_metadata,
)


def _training_group(group: str, train_frac: float, seed: int) -> bool:
    value = int(hashlib.md5(f"{seed}{group}".encode()).hexdigest(), 16) / float(1 << 128)
    return value < train_frac


def _topology_group(path: Path, root: Path, groups: dict[str, str]) -> str:
    key = path.relative_to(root).as_posix()
    if key not in groups and key.startswith("minimal_component_v4/"):
        legacy = "minimal_component/" + key.split("/", 1)[1]
        if legacy in groups:
            key = legacy
    try:
        return groups[key]
    except KeyError as exc:
        raise RuntimeError(
            f"topology manifest has no entry for {path.relative_to(root)}"
        ) from exc


def _bounded(values: list[np.ndarray], cap: int) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=np.float64)
    merged = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values])
    if merged.size <= cap:
        return merged
    index = np.linspace(0, merged.size - 1, cap, dtype=np.int64)
    return merged[index]


def _worker(args: tuple[str, int, int]) -> dict[str, np.ndarray]:
    feeder_s, variants_limit, cap = args
    scenarios = FeederScenarios(feeder_s)
    samples: dict[str, list[np.ndarray]] = {}
    n_variants = len(scenarios) if variants_limit <= 0 else min(variants_limit, len(scenarios))
    for variant in range(n_variants):
        data = scenarios[variant]
        for store, (dim, y_fields) in SCENARIO_Y_FIELDS.items():
            if store not in data.node_types:
                continue
            first = getattr(data[store], y_fields[0][0], None)
            if first is None or first.shape[0] == 0:
                continue
            family = "Line" if store == "line" else SPECS[store]["json_key"]
            rows, cols = np.tril_indices(dim)
            diagonal = rows == cols
            for pu_field, _feat_field, part in y_fields:
                full = getattr(data[store], pu_field).double().numpy().reshape(-1, dim, dim)
                packed = np.abs(full[:, rows, cols])
                for band, select in (("diag", diagonal), ("offdiag", ~diagonal)):
                    samples.setdefault(f"Y|{family}|{part}|{band}", []).append(
                        packed[:, select]
                    )

            spec = COMPONENTS[store]
            icomp_slots = int(spec.get("icomp_slots", 0))
            ic_r = getattr(data[store], "Icomp_r_pu", None) if icomp_slots else None
            ic_i = getattr(data[store], "Icomp_i_pu", None) if icomp_slots else None
            if ic_r is not None:
                samples.setdefault(f"I|{family}", []).append(
                    np.hypot(ic_r.double().numpy(), ic_i.double().numpy())
                )
            for term in range(1, int(spec["terminals"]) + 1):
                real = getattr(data[store], f"I_r_bus{term}_pu").double().numpy()
                imag = getattr(data[store], f"I_i_bus{term}_pu").double().numpy()
                if ic_r is not None:
                    start = (term - 1) * FIXED_CONDUCTORS
                    real = real + ic_r[:, start:start + FIXED_CONDUCTORS].double().numpy()
                    imag = imag + ic_i[:, start:start + FIXED_CONDUCTORS].double().numpy()
                samples.setdefault(f"I|{family}", []).append(np.hypot(real, imag))
    return {key: _bounded(values, cap) for key, values in samples.items()}


def _p95(chunks: list[np.ndarray]) -> float:
    values = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    return max(float(np.quantile(values, 0.95)) if values.size else SCALE_FLOOR, SCALE_FLOOR)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--limit-feeders", type=int)
    parser.add_argument(
        "--limit-per-corpus", type=int,
        help="select this many hash-shuffled training feeders per top-level corpus",
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--variants", type=int, default=0, help="0 uses every variant")
    parser.add_argument("--sketch-per-feeder", type=int, default=2048)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--topology-manifest", type=Path,
        default=ROOT / "gridfm" / "topology_fingerprints.json",
    )
    parser.add_argument("--alpha-current", type=float, default=1e-3)
    args = parser.parse_args()
    root = args.root.resolve()
    groups = json.loads(args.topology_manifest.read_text()).get("feeders", {})
    if not groups:
        raise RuntimeError(f"empty topology manifest: {args.topology_manifest}")
    feeders = sorted(
        path.parent for path in root.rglob("static.pt")
        if (path.parent / "dynamic.npy").is_file()
        and _training_group(
            _topology_group(path.parent, root, groups),
            args.train_frac, args.split_seed,
        )
    )
    if args.limit_feeders is not None and args.limit_per_corpus is not None:
        raise ValueError("use only one of --limit-feeders and --limit-per-corpus")
    if args.limit_per_corpus is not None:
        grouped: dict[str, list[Path]] = {}
        for feeder in feeders:
            corpus = feeder.relative_to(root).parts[0]
            grouped.setdefault(corpus, []).append(feeder)
        feeders = []
        for corpus in sorted(grouped):
            rows = sorted(
                grouped[corpus],
                key=lambda path: hashlib.sha256(
                    f"{args.selection_seed}|{path.relative_to(root)}".encode()
                ).digest(),
            )
            feeders.extend(rows[:args.limit_per_corpus])
    if args.limit_feeders is not None:
        feeders = feeders[:args.limit_feeders]
    if not feeders:
        raise RuntimeError("no training-split scenario stores found")
    jobs = [(str(feeder), int(args.variants), int(args.sketch_per_feeder)) for feeder in feeders]
    chunks: dict[str, list[np.ndarray]] = {}
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)), mp_context=mp.get_context("spawn")
    ) as pool:
        for done, result in enumerate(pool.map(_worker, jobs, chunksize=1), 1):
            for key, values in result.items():
                chunks.setdefault(key, []).append(values)
            if done % 100 == 0 or done == len(jobs):
                print(f"scaler sketch: feeders={done}/{len(jobs)}", flush=True)

    scaler = {"epsilon": FEAT_EPS, "current": {}, "admittance": {}}
    for key, values in chunks.items():
        parts = key.split("|")
        if parts[0] == "I":
            scaler["current"][parts[1]] = {"I_scale": _p95(values), "transform": "asinh"}
        else:
            _, family, part, band = parts
            cfg = scaler["admittance"].setdefault(family, {"transform": "asinh"})
            cfg[f"Y_{part}_{band}_scale"] = _p95(values)
    healthy = [
        float(cfg["I_scale"]) for cfg in scaler["current"].values()
        if float(cfg["I_scale"]) > SCALE_FLOOR
    ]
    current_floor = (
        float(args.alpha_current) * statistics.median(healthy) if healthy else SCALE_FLOOR
    )
    for cfg in scaler["current"].values():
        cfg["I_scale"] = max(float(cfg["I_scale"]), current_floor)
    scaler["training_split"] = {
        "rule": "topology-grouped md5(seed + WL fingerprint)",
        "train_frac": float(args.train_frac), "seed": int(args.split_seed),
        "topology_manifest": str(args.topology_manifest.resolve()),
        "feeders": len(feeders), "variants_limit": int(args.variants),
    }
    scaler["fit_source"] = {
        "root": str(root), "sketch_per_feeder": int(args.sketch_per_feeder),
        "unified_line_scale": True, "alpha_current": float(args.alpha_current),
        "current_floor": current_floor,
    }
    out = (args.out or (root / "feature_scaler.json")).resolve()
    save_scaler_metadata(scaler, out)
    print(json.dumps({"out": str(out), "current": scaler["current"],
                      "admittance": scaler["admittance"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
