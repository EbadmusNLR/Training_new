"""Gating measurement for the decoder port: what does the exact path COST per sample?

dk_model calls reconstruct_vectorized (measured ~9e-01 WAPE on SMART-DS -- useless).
reconstruct_full is the exact path (~7e-08) but needs build_recon_ctx, which is
per-VARIANT, not per-feeder: its transformer null-space maps come from Yxfmr, and variants
change transformer TAPS. build_tree_plan is cached once per feeder because topology is
static; this cannot be. So the port hinges on build_recon_ctx being cheap enough to run
per sample in the dataloader workers.

Timings are WARMED UP and taken over several variants: the first call in a process pays
numpy/torch sparse first-call overhead that made a 23-node feeder look like 1.1s.

Run on a compute node with feeder dirs on argv.
"""
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")

from core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, store_size, stored_currents, element_currents, node_count
from gridfm.dk_tree import (build_recon_ctx, reconstruct_full, build_tree_plan,
                            reconstruct_vectorized, SHUNT_STORES, AMBIG_STORES)

REPS = 5


def load(fdir, k):
    d = FeederScenarios(fdir)[k]
    d["node"].num_nodes = node_count(d)
    return d


def wape(rec, truth, present):
    num = den = 0.0
    for s in present:
        if s not in rec:
            continue
        num += float((rec[s][0] - truth[s][0]).abs().sum() + (rec[s][1] - truth[s][1]).abs().sum())
        den += float(truth[s][0].abs().sum() + truth[s][1].abs().sum())
    return num / (den + 1e-30)


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


# ---- warm every code path once so first-call overhead is not attributed to a feeder
w = load(sys.argv[1], 0)
vr, vi, present, truth, fresh = prep(w)
_c = build_recon_ctx(w); reconstruct_full(w, fresh(), vr, vi, ctx=_c)
reconstruct_vectorized(build_tree_plan(w), fresh())
print(f"(warmed up on {os.path.basename(sys.argv[1])[-30:]})\n")

print(f"{'feeder':40s} {'nodes':>6s} {'xfmr':>5s} {'build_ctx':>10s} {'recon_full':>11s} "
      f"{'WAPE_full':>10s} {'recon_vec':>10s} {'WAPE_vec':>9s}")
for fdir in sys.argv[1:]:
    name = os.path.basename(fdir)[-38:]
    ctxs, fulls, vecs = [], [], []
    wf = wv = 0.0
    n = nx = 0
    for k in range(REPS):
        d = load(fdir, k)
        n = node_count(d); nx = store_size(d, "transformer") if "transformer" in d.node_types else 0
        vr, vi, present, truth, fresh = prep(d)
        t = time.perf_counter(); ctx = build_recon_ctx(d); ctxs.append(time.perf_counter() - t)
        t = time.perf_counter(); rec = reconstruct_full(d, fresh(), vr, vi, ctx=ctx); fulls.append(time.perf_counter() - t)
        wf = wape(rec, truth, present)
        plan = build_tree_plan(d)
        t = time.perf_counter(); rv = reconstruct_vectorized(plan, fresh()); vecs.append(time.perf_counter() - t)
        wv = wape(rv, truth, present)
    print(f"{name:40s} {n:6d} {nx:5d} {statistics.median(ctxs):9.4f}s {statistics.median(fulls):10.4f}s "
          f"{wf:10.2e} {statistics.median(vecs):9.4f}s {wv:9.2e}")
