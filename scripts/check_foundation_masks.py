#!/usr/bin/env python3
"""Compute-node gate for the identifiable foundation masking contract."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridfm.config import load_config
from gridfm.data import build_strict_datasets
from gridfm.legacy import SPECS, i_offset, y_width


def sample_for(bundle, task: str):
    bundle.train.mask_cfg = {**bundle.train.mask_cfg, "mixture": {task: 1.0}}
    bundle.train.set_epoch(0)
    return bundle.train[0]


def assert_boundary(data) -> None:
    nd = data["node"]
    assert int(nd.slack.sum()) == 3
    assert bool(nd.vis_v[nd.slack].all())
    assert not bool(nd.msk_v[nd.slack].any())
    assert nd.v_init.shape == nd.dv.shape and nd.v_init.shape[1] == 2


def role_counts(data) -> dict[str, int]:
    counts = {"V": int(data["node"].msk_v.sum()), "Y": 0, "Icomp": 0, "Ibus": 0}
    for store in SPECS:
        st = data[store]
        ny, ni = y_width(store), i_offset(store)
        counts["Y"] += int(st.msk[:, :ny].sum())
        counts["Icomp"] += int(st.msk[:, ny:ni].sum())
        counts["Ibus"] += int(st.msk[:, ni:].sum())
        assert not bool((st.msk & ~st.act).any())
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/e19_foundation_identifiable.yaml"))
    ap.add_argument("--limit-feeders", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["data"]["limit_feeders"] = args.limit_feeders
    bundle = build_strict_datasets(cfg["data"], cfg["mask"], int(cfg["train"]["seed"]))

    rows = {}
    for task in ("pf", "se_known", "param_one", "injection"):
        data = sample_for(bundle, task)
        assert_boundary(data)
        rows[task] = role_counts(data)

    pf = rows["pf"]
    assert pf["V"] > 0 and pf["Ibus"] > 0 and pf["Y"] == pf["Icomp"] == 0
    se = rows["se_known"]
    assert se["V"] > 0 and se["Ibus"] > 0 and se["Y"] == se["Icomp"] == 0
    param = rows["param_one"]
    assert param["Y"] > 0 and param["V"] == param["Icomp"] == param["Ibus"] == 0
    inj = rows["injection"]
    assert inj["Icomp"] > 0 and inj["V"] == inj["Y"] == inj["Ibus"] == 0

    # param_one must select exactly one active triangular position in every
    # non-empty component row and pair all stored real/imaginary Y fields.
    pdata = sample_for(bundle, "param_one")
    for store, spec in SPECS.items():
        st = pdata[store]
        if st.num_nodes == 0:
            continue
        ny = y_width(store)
        tri = ny // len(spec.yfields)
        target = st.msk[:, :ny].reshape(st.num_nodes, len(spec.yfields), tri)
        active = st.act[:, :ny].reshape_as(target)
        for row in range(st.num_nodes):
            if active[row].any():
                selected = target[row].any(0)
                assert int(selected.sum()) == 1
                col = int(selected.nonzero()[0])
                assert bool((target[row, :, col] == active[row, :, col]).all())

    for task, counts in rows.items():
        print(task, counts)
    print("foundation masking gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
