"""What does the MODEL's decoder actually deliver on the TRAINING corpus?

dk_model._completed_currents calls reconstruct_vectorized (the old plan-based sweep).
reconstruct_full -- the one validated to 6.05e-08 on SMART-DS -- is NOT in the model
path. So every decoder fix (joint transformer system, bridges, mesh/KVL loops, line
charging, per-element ambiguous split) may be sitting outside training entirely.

Both are fed the SAME truth V and the SAME physics-decoded shunts, so any difference is
purely the reconstruction. Also runs the old path in fp32 (as the model does) vs fp64,
since the model trains fp32.
"""
import glob, os, sys
import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, store_size, stored_currents, element_currents
from gridfm.dk_tree import (reconstruct_full, reconstruct_vectorized, build_recon_ctx,
                            build_tree_plan, SHUNT_STORES, AMBIG_STORES, SERIES_STORES)

CORPUS = os.environ.get("CORPUS", "SMART-DS_1000")
NF = int(os.environ.get("NF", "6"))
NV = int(os.environ.get("NV", "2"))
TD = f"/kfs2/projects/gogpt/Ebadmus/training_data/{CORPUS}"
paths = sorted(glob.glob(os.path.join(TD, "*", "static.pt")))
step = max(1, len(paths) // NF)
paths = paths[::step][:NF]

print(f"corpus {CORPUS}: {len(paths)} feeders x {NV} variants")
print(f"{'feeder':34s} {'recon_full(fp64)':>17s} {'model path fp64':>16s} {'model path fp32':>16s}")
tot = {"full": [0.0, 0.0], "vec64": [0.0, 0.0], "vec32": [0.0, 0.0]}
for p in paths:
    name = os.path.basename(os.path.dirname(p))[:32]
    fs = FeederScenarios(os.path.dirname(p))
    ctx = None
    acc = {"full": [0.0, 0.0], "vec64": [0.0, 0.0], "vec32": [0.0, 0.0]}
    for v in range(min(NV, len(fs))):
        d = fs[v]
        vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
        present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
        truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
        cur = {}
        for s in present:
            if s in SHUNT_STORES or s in AMBIG_STORES:
                cur[s] = element_currents(d, s, vr, vi)
            else:
                z = torch.zeros_like(truth[s][0]); cur[s] = (z, z.clone())
        ctx = build_recon_ctx(d, topo=ctx)
        recs = {"full": reconstruct_full(d, cur, vr, vi, ctx=ctx)}
        d["node"].slack = torch.zeros(d["node"].V_r_pu.shape[0], dtype=torch.bool)
        rel = ("vsource", "bus1", "node")
        if rel in d.edge_types and d[rel].edge_index.numel():
            d["node"].slack[d[rel].edge_index[1]] = True
        plan = build_tree_plan(d)
        recs["vec64"] = reconstruct_vectorized(plan, cur)
        cur32 = {s: (a.float(), b.float()) for s, (a, b) in cur.items()}
        recs["vec32"] = reconstruct_vectorized(plan, cur32)
        for k, rec in recs.items():
            for s in present:
                R = rec.get(s, cur[s]); T = truth[s]
                acc[k][0] += float((R[0].double() - T[0]).abs().sum()
                                   + (R[1].double() - T[1]).abs().sum())
                acc[k][1] += float(T[0].abs().sum() + T[1].abs().sum())
    for k in acc:
        tot[k][0] += acc[k][0]; tot[k][1] += acc[k][1]
    w = {k: acc[k][0] / (acc[k][1] + 1e-30) for k in acc}
    print(f"{name:34s} {w['full']:17.3e} {w['vec64']:16.3e} {w['vec32']:16.3e}")
print("-" * 88)
w = {k: tot[k][0] / (tot[k][1] + 1e-30) for k in tot}
print(f"{'AGGREGATE':34s} {w['full']:17.3e} {w['vec64']:16.3e} {w['vec32']:16.3e}")
print("\nrecon_full = validated decoder (NOT in the model). model path = what training sees.")
