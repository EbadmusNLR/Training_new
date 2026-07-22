"""Does caching the TOPOLOGY half of build_recon_ctx make the decoder port affordable?

build_recon_ctx is per-VARIANT because transformer null-space maps depend on Yxfmr and
variants move taps. But its topology half (tree, KVL rows, injection indices, series
classification) is driven by edge_index, which IS static across a feeder's variants.
`build_recon_ctx(data, topo=<ctx from a sibling variant>)` reuses that half.

Measured full-build cost was 2.29s on a 9710-node / 517-transformer feeder, which at
batch 32 / 16 workers would make the dataloader the bottleneck. This asks how much of
that is topology (cacheable once per feeder, like build_tree_plan) versus the Y-dependent
transformer maps (unavoidably per-variant).

Also verifies reuse is CORRECT, not just fast: a topo-reused ctx must reconstruct to the
same WAPE as a freshly built one on a variant with different taps.
"""
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")

from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, store_size, stored_currents, element_currents, node_count
from gridfm.dk_tree import (build_recon_ctx, reconstruct_full, SHUNT_STORES, AMBIG_STORES)

REPS = 5


def load(fdir, k):
    d = FeederScenarios(fdir)[k]
    d["node"].num_nodes = node_count(d)
    return d


def prep(d):
    vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
    present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
    truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}

    def fresh():
        cur = {}
        for s in present:
            if s in SHUNT_STORES or s in AMBIG_STORES:
                cur[s] = tuple(x.clone() for x in element_currents(d, s, vr, vi))
            else:
                z = torch.zeros_like(truth[s][0]); cur[s] = (z, z.clone())
        return cur
    return vr, vi, present, truth, fresh


def wape(rec, truth, present):
    num = den = 0.0
    for s in present:
        if s not in rec:
            continue
        num += float((rec[s][0] - truth[s][0]).abs().sum() + (rec[s][1] - truth[s][1]).abs().sum())
        den += float(truth[s][0].abs().sum() + truth[s][1].abs().sum())
    return num / (den + 1e-30)


w = load(sys.argv[1], 0)
vr, vi, present, truth, fresh = prep(w)
_t = build_recon_ctx(w); reconstruct_full(w, fresh(), vr, vi, ctx=_t)
print(f"(warmed up)\n")

print(f"{'feeder':38s} {'nodes':>6s} {'xfmr':>5s} {'full':>9s} {'topo-reuse':>11s} "
      f"{'speedup':>8s} {'WAPE_reuse':>11s} {'WAPE_full':>10s}")
for fdir in sys.argv[1:]:
    name = os.path.basename(fdir)[-36:]
    d0 = load(fdir, 0)
    topo = build_recon_ctx(d0)                       # cache once per feeder
    n = node_count(d0)
    nx = store_size(d0, "transformer") if "transformer" in d0.node_types else 0
    full_t, reuse_t = [], []
    w_reuse = w_full = 0.0
    for k in range(1, REPS + 1):                     # variants with DIFFERENT taps
        d = load(fdir, k)
        vr, vi, present, truth, fresh = prep(d)
        t = time.perf_counter(); c_full = build_recon_ctx(d); full_t.append(time.perf_counter() - t)
        t = time.perf_counter(); c_re = build_recon_ctx(d, topo=topo); reuse_t.append(time.perf_counter() - t)
        w_full = wape(reconstruct_full(d, fresh(), vr, vi, ctx=c_full), truth, present)
        w_reuse = wape(reconstruct_full(d, fresh(), vr, vi, ctx=c_re), truth, present)
    mf, mr = statistics.median(full_t), statistics.median(reuse_t)
    print(f"{name:38s} {n:6d} {nx:5d} {mf:8.4f}s {mr:10.4f}s {mf/max(mr,1e-9):7.1f}x "
          f"{w_reuse:11.2e} {w_full:10.2e}")
