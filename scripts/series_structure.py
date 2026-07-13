#!/usr/bin/env python3
"""Model-free structural check of series-element terminal currents (truth only).

For the extended-KCL current sweep to reconstruct transformer/reactor series
currents (instead of leaving them as stiff direct-head injections that pollute
line reconstruction), we need to know how their terminal currents relate:

  reactor (2 terminals):     is I1 ~= -I2 per conductor (paired series + shunt)?
  transformer (3 terminals): is I1 ~= -(I2 + I3) per conductor (pu-ideal), or is
                             a turns ratio / power balance needed?

Uses ground-truth Ibus decoded from the corpus. Reports, per family, the WAPE of
the KCL-implied terminal current vs the stored truth, and how big the residual
(I1+I2[+I3]) is relative to the individual terminal magnitude. Also reports how
often each family appears per feeder so we know the leverage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridfm.data import build_strict_datasets
from gridfm.config import load_config
from gridfm.legacy import physics, FC, i_offset, SPECS


def truth_terminal_currents(batch, store):
    """Decoded truth Ibus as complex [n, terms, FC]."""
    st = batch[store]
    ni = i_offset(store)
    pu = physics.decode_truth(st.x_true[:, ni:], st.scale[:, ni:])
    terms = SPECS[store].terms
    # column layout per terminal t: real block [t*2FC : t*2FC+FC], imag [+FC:+2FC]
    out = torch.zeros(st.num_nodes, terms, FC, dtype=torch.cfloat)
    for t in range(terms):
        r = pu[:, t * 2 * FC: t * 2 * FC + FC]
        i = pu[:, t * 2 * FC + FC: t * 2 * FC + 2 * FC]
        out[:, t] = torch.complex(r.float(), i.float())
    return out, st.act[:, ni:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", default="unseen")
    ap.add_argument("--limit-batches", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["mask"]["mixture"] = {"pf": 1.0}
    bundle = build_strict_datasets(cfg["data"], cfg["mask"], int(cfg["train"]["seed"]))
    dataset = getattr(bundle, args.split)
    batches = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)

    stat = {s: {"kcl_num": 0.0, "term_den": 0.0, "count": 0, "graphs": 0}
            for s in ("reactor", "transformer", "line", "vsource")}
    n_graphs = 0
    with torch.no_grad():
        for bi, batch in enumerate(batches):
            if bi >= args.limit_batches:
                break
            nb = getattr(batch["node"], "batch", None)
            n_graphs += int(nb.max().item()) + 1 if nb is not None else 1
            for store in stat:
                st = batch[store]
                if st.num_nodes == 0:
                    continue
                cur, act = truth_terminal_currents(batch, store)
                terms = SPECS[store].terms
                # per-conductor sum of terminal currents (KCL residual if paired)
                resid = cur.sum(dim=1)                    # [n, FC]
                term_mag = cur.abs().sum(dim=1)           # [n, FC] sum over terminals
                # only conductors active on terminal 0
                a0 = act[:, :FC].bool()
                m = a0
                stat[store]["kcl_num"] += resid.abs()[m].sum().item()
                stat[store]["term_den"] += term_mag[m].sum().item()
                stat[store]["count"] += st.num_nodes
                stat[store]["graphs"] += st.num_nodes

    print(f"graphs scanned: {n_graphs}")
    print(f"{'family':14s} {'sum|I1+..+In|/sum|Ii| %':>24s} {'#elems':>10s} {'per-graph':>10s}")
    for s, d in stat.items():
        if d["term_den"] > 0:
            pct = 100.0 * d["kcl_num"] / d["term_den"]
            print(f"{s:14s} {pct:24.4f} {d['count']:10d} {d['count']/max(1,n_graphs):10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
