#!/usr/bin/env python3
"""End-to-end feature-store, definition-carry, and exact-Y integration gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT), str(PROJECT), str(PROJECT / "DG_FM_Training")]

from gridfm.data import build_strict_datasets  # noqa: E402
from datakit.core.scenario_store import FeederScenarios  # noqa: E402
from gridfm.featurizing import (  # noqa: E402
    _scenario_build_feat_sample, _scenario_unified_families,
)


def _assert_exact_conversion(source: Path, target: Path, scaler: dict) -> None:
    raw_scenarios = FeederScenarios(source)
    feat_scenarios = FeederScenarios(target)
    if raw_scenarios.variant_ids != feat_scenarios.variant_ids:
        raise AssertionError(f"variant IDs changed: {target}")
    for variant in range(len(raw_scenarios)):
        raw = raw_scenarios[variant]
        expected = _scenario_build_feat_sample(
            raw, scaler, _scenario_unified_families(raw)
        )
        actual = feat_scenarios[variant]
        if set(expected.node_types) != set(actual.node_types):
            raise AssertionError(f"node types changed: {target} variant={variant}")
        for store in expected.node_types:
            for field, value in expected[store].items():
                if not torch.is_tensor(value):
                    continue
                got = getattr(actual[store], field, None)
                if not torch.is_tensor(got) or not torch.equal(value, got):
                    raise AssertionError(
                        f"conversion mismatch: {target} variant={variant} {store}.{field}"
                    )
        if set(expected.edge_types) != set(actual.edge_types):
            raise AssertionError(f"edge types changed: {target} variant={variant}")
        for edge_type in expected.edge_types:
            for field, value in expected[edge_type].items():
                if torch.is_tensor(value) and not torch.equal(
                    value, getattr(actual[edge_type], field)
                ):
                    raise AssertionError(
                        f"edge mismatch: {target} variant={variant} {edge_type}.{field}"
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--source-root", type=Path,
        help="pu stores to compare exactly against the feature conversion",
    )
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument(
        "--limit-per-corpus", type=int,
        help="check this many stores from each top-level corpus",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    scaler = json.loads((root / "feature_scaler.json").read_text())
    feeders = sorted(
        path.parent for path in root.rglob("static.pt")
        if (path.parent / "dynamic.npy").is_file()
    )
    if args.limit_per_corpus is not None:
        grouped: dict[str, list[Path]] = {}
        for feeder in feeders:
            grouped.setdefault(feeder.relative_to(root).parts[0], []).append(feeder)
        feeders = [
            feeder
            for corpus in sorted(grouped)
            for feeder in grouped[corpus][:args.limit_per_corpus]
        ]
    else:
        feeders = feeders[:args.limit]
    if not feeders:
        raise AssertionError("no feature stores")
    if args.source_root is not None:
        source_root = args.source_root.resolve()
        for feeder in feeders:
            _assert_exact_conversion(
                source_root / feeder.relative_to(root), feeder, scaler
            )
    line_rows = transformer_rows = 0
    for feeder in feeders:
        meta = torch.load(feeder / "static.pt", map_location="cpu", weights_only=False)
        if meta.get("basis") != "feat":
            raise AssertionError(f"not feature basis: {feeder}")
        for entry in meta["layout"]:
            if entry["store"] == "line" and entry["field"] == "Ys_r_tri_feat":
                line_rows += int(entry["shape"][0])
            elif entry["store"] == "transformer" and entry["field"] == "Yxfmr_r_tri_feat":
                transformer_rows += int(entry["shape"][0])
    cfg = {
        "root": str(root), "cache_dir": str(root / ".contract_cache"),
        "cast_float32": True, "train_frac": 0.8, "val_frac": 0.1,
        "train_variants": 2, "eval_variants": 1,
        "limit_feeders": len(feeders),
        "exact_line_metadata": True, "exact_transformer_metadata": True,
    }
    mask = {
        "mixture": {"pf": 1.0}, "p_voltage": 0.3, "p_current": 0.15,
        "p_icomp": 0.0, "p_admittance": 0.0, "p_terminal": 0.0,
        "p_component": 0.0,
    }
    bundle = build_strict_datasets(cfg, mask, seed=0)
    if not bundle.train.caches:
        raise AssertionError("self-contained dataset build returned no caches")
    for cache in bundle.train.caches:
        for variant in {0, cache.n_variants - 1}:
            sample = cache.sample(variant)
            for family in ("line", "transformer"):
                if sample[family].num_nodes and not hasattr(
                    sample[family], "metadata_y_feat"
                ):
                    raise AssertionError(
                        f"missing exact feature: {cache.name} variant={variant} {family}"
                    )
    print(
        f"FEATURE_STORE_CONTRACT_OK feeders={len(bundle.train.caches)} "
        f"line={line_rows} transformer={transformer_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
