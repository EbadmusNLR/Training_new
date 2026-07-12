"""Strict dataset construction and leakage assertions."""
from __future__ import annotations

from dataclasses import dataclass

from .legacy import build_datasets


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
    return {dataset.caches[fi].name for fi, _ in dataset.items}


def build_strict_datasets(data_cfg: dict, mask_cfg: dict, seed: int) -> DatasetBundle:
    """Build feeder-disjoint splits and reject target-derived nominal fields."""
    limit = data_cfg.get("limit_feeders")
    train, seen, unseen, test = build_datasets(data_cfg, mask_cfg, seed, limit=limit)
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

