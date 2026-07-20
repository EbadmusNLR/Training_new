"""Strict dataset construction and leakage assertions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch

from .legacy import build_datasets
from .exact_metadata import attach_exact_metadata


@dataclass(frozen=True)
class DatasetBundle:
    train: object
    seen: object
    unseen: object
    test: object
    train_feeders: tuple[str, ...]
    unseen_feeders: tuple[str, ...]
    test_feeders: tuple[str, ...]


def _names(dataset) -> set[str]:
    return {
        getattr(dataset.caches[fi], "split_group", dataset.caches[fi].name)
        for fi, _ in dataset.items
    }


def build_strict_datasets(data_cfg: dict, mask_cfg: dict, seed: int) -> DatasetBundle:
    """Build feeder-disjoint splits and reject target-derived nominal fields."""
    data_cfg = dict(data_cfg)
    manifest = Path(__file__).with_name("topology_fingerprints.json")
    if manifest.is_file():
        data_cfg.setdefault("topology_manifest", str(manifest))
        data_cfg.setdefault("split_seed", 42)
        data_cfg.setdefault("require_topology_manifest_coverage", True)
    limit = data_cfg.get("limit_feeders")
    train, seen, unseen, test = build_datasets(data_cfg, mask_cfg, seed, limit=limit)
    exact_started = time.perf_counter()
    exact_line = bool(data_cfg.get("exact_line_metadata", False))
    exact_transformer = bool(data_cfg.get("exact_transformer_metadata", False))
    exact_generator = bool(data_cfg.get("exact_generator_metadata", False))
    exact_workers = int(data_cfg.get("exact_metadata_workers", 0))
    attach_exact_metadata(
        train.caches,
        exact_line,
        exact_transformer,
        exact_workers,
        generator=exact_generator,
        # Anchor the derived cache to the immutable feature corpus so training,
        # evaluation and validation configs all reuse one decode.
        disk_cache_dir=Path(data_cfg["root"]) / ".exact_metadata_cache_v1",
    )
    if exact_line or exact_transformer or exact_generator:
        print(
            f"exact metadata prepared in {time.perf_counter() - exact_started:.1f}s "
            f"with workers={exact_workers}",
            flush=True,
        )
    train_names, seen_names = _names(train), _names(seen)
    unseen_names, test_names = _names(unseen), _names(test)
    if train_names != seen_names:
        raise AssertionError("seen split must contain held variants of exactly the train feeders")
    if train_names & unseen_names or train_names & test_names or unseen_names & test_names:
        raise AssertionError("feeder leakage across topology splits")
    if not train.items:
        raise ValueError("empty training split")

    probe = train[0]
    nd = probe["node"]
    if hasattr(nd, "v_nominal") or hasattr(nd, "v_nominal_raw"):
        raise AssertionError("target-derived per-topology voltage nominal is forbidden")
    if int(nd.slack.sum()) != 3 or not bool(nd.vis_v[nd.slack].all()):
        raise AssertionError("three solved slack-phase voltages must always be visible")
    if nd.v_init.shape != nd.dv.shape or nd.v_init.shape[1] != 2:
        raise AssertionError("V_init and dV must preserve separate real/imaginary channels")

    return DatasetBundle(
        train=train,
        seen=seen,
        unseen=unseen,
        test=test,
        train_feeders=tuple(sorted(train_names)),
        unseen_feeders=tuple(sorted(unseen_names)),
        test_feeders=tuple(sorted(test_names)),
    )


def fit_feature_stats(dataset, min_std: float = 1e-8) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Fit per-column feature mean/std on training feeders and variants only."""
    from .legacy import SPECS, store_width

    acc = {
        s: {
            "sum": torch.zeros(store_width(s), dtype=torch.float64),
            "sq": torch.zeros(store_width(s), dtype=torch.float64),
            "count": torch.zeros(store_width(s), dtype=torch.float64),
        }
        for s in SPECS
    }
    for idx in range(len(dataset)):
        sample = dataset[idx]
        for store in SPECS:
            st = sample[store]
            if st.num_nodes == 0:
                continue
            x, active = st.x_true.double(), st.act
            acc[store]["sum"] += (x * active).sum(0)
            acc[store]["sq"] += (x.square() * active).sum(0)
            acc[store]["count"] += active.sum(0)
    out = {}
    for store, row in acc.items():
        count = row["count"].clamp_min(1)
        mean = row["sum"] / count
        var = (row["sq"] / count - mean.square()).clamp_min(0)
        std = var.sqrt()
        # Constant/unseen columns keep identity scaling; subtracting their
        # fitted mean is still useful when they are visible model inputs.
        std = torch.where(std >= min_std, std, torch.ones_like(std))
        out[store] = (mean, std)
    return out
