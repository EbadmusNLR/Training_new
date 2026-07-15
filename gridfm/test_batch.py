"""Full solver-free decoder, zero-init series, stored shunts: aggregate WAPE per
series family (line / transformer / vsource) across a broad feeder set."""
import glob, os, sys
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
from core.scenario_store import FeederScenarios
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from gridfm.dk_physics import STORES, store_size, stored_currents
from gridfm.dk_tree import reconstruct_full

ROOT = "/kfs2/projects/gogpt/Ebadmus/training_data/SMART-DS_1000"
feeders = sorted(glob.glob(os.path.join(ROOT, "*", "static.pt")), key=os.path.getsize)
probe = feeders[:5] + feeders[len(feeders)//2: len(feeders)//2+5] + feeders[-3:]

agg = {}
for p in probe:
    d = FeederScenarios(os.path.dirname(p))[0]
    cur = {s: stored_currents(d, s, dtype=torch.float64) for s in STORES
           if s in d.node_types and store_size(d, s) > 0}
    rec = reconstruct_full(d, cur, d["node"].V_r_pu, d["node"].V_i_pu)
    for s in ("line", "transformer", "vsource"):
        if s not in rec or s not in cur:
            continue
        Rr, Ri = rec[s]; Tr, Ti = cur[s]
        a = agg.setdefault(s, [0.0, 0.0])
        a[0] += float((Rr - Tr).abs().sum() + (Ri - Ti).abs().sum())
        a[1] += float(Tr.abs().sum() + Ti.abs().sum() + 1e-12)
print(f"=== reconstruct_full aggregate, zero-init, {len(probe)} feeders ===")
for s, (nu, de) in agg.items():
    print(f"  {s:12s} WAPE = {nu/de:.3e}")
