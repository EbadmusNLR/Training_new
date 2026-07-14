import glob, os, sys
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
from core.scenario_store import FeederScenarios
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from gridfm.dk_physics import STORES, store_size, stored_currents
from gridfm.dk_tree import build_tree_plan, reconstruct_vectorized, SERIES_STORES

ROOT = "/kfs2/projects/gogpt/Ebadmus/training_data/SMART-DS_1000"
feeders = sorted(glob.glob(os.path.join(ROOT, "*", "static.pt")), key=os.path.getsize)
sel = feeders[len(feeders) // 2: len(feeders) // 2 + 3]
datas = [FeederScenarios(os.path.dirname(p))[0] for p in sel]
plans = [build_tree_plan(d) for d in datas]
curs = [{s: stored_currents(d, s, dtype=torch.float64) for s in STORES
         if s in d.node_types and store_size(d, s) > 0} for d in datas]


def zero_series(cur):
    c = dict(cur)
    for s in SERIES_STORES:
        if s in c:
            z = torch.zeros_like(c[s][0]); c[s] = (z, z.clone())
    return c


print("--- WAPE vs stored, zero-init + KCL closure (per-feeder) ---")
agg = {}
for i, (p, c) in enumerate(zip(plans, curs)):
    rec = reconstruct_vectorized(p, zero_series(c))
    for s in ("line", "transformer", "vsource"):
        if s not in rec:
            continue
        Rr, Ri = rec[s]; Tr, Ti = c[s]
        a = agg.setdefault(s, [0.0, 0.0])
        a[0] += float((Rr - Tr).abs().sum() + (Ri - Ti).abs().sum())
        a[1] += float(Tr.abs().sum() + Ti.abs().sum() + 1e-12)
for s, (nu, de) in agg.items():
    print(f"  {s:12s} WAPE = {nu/de:.3e}")
