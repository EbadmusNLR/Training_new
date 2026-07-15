"""TRUE end-to-end decoder test, ALL components, from V alone.

  V -> physics-decode SHUNTS (I=Y@V-Icomp) -> reconstruct_full -> ALL series.

Shunts are NOT taken from stored here: they are decoded from V, exactly as the
model does. Series start at zero. Reports WAPE for every component family.
At truth V this is the decoder's ceiling; the model's V error then propagates.
"""
import glob, os, sys
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
from core.scenario_store import FeederScenarios
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from gridfm.dk_physics import STORES, store_size, stored_currents, element_currents
from gridfm.dk_tree import reconstruct_full, SHUNT_STORES, SERIES_STORES

ROOT = "/kfs2/projects/gogpt/Ebadmus/training_data/SMART-DS_1000"
feeders = sorted(glob.glob(os.path.join(ROOT, "*", "static.pt")), key=os.path.getsize)
probe = feeders[:5] + feeders[len(feeders)//2: len(feeders)//2+5] + feeders[-3:]

agg = {}
for p in probe:
    d = FeederScenarios(os.path.dirname(p))[0]
    vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
    present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
    truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
    # shunts: DECODED from V (what the model does). series: zero placeholders.
    cur = {}
    for s in present:
        if s in SHUNT_STORES:
            cur[s] = element_currents(d, s, vr, vi)
        else:
            z = torch.zeros_like(truth[s][0]); cur[s] = (z, z.clone())
    rec = reconstruct_full(d, cur, vr, vi)
    for s in present:
        R = rec.get(s, cur[s]); T = truth[s]
        a = agg.setdefault(s, [0.0, 0.0])
        a[0] += float((R[0]-T[0]).abs().sum() + (R[1]-T[1]).abs().sum())
        a[1] += float(T[0].abs().sum() + T[1].abs().sum() + 1e-12)

print(f"=== FULL decoder from V alone, ALL components, {len(probe)} feeders ===")
tot_n = tot_d = 0.0
for s in list(SHUNT_STORES) + list(SERIES_STORES):
    if s not in agg:
        continue
    nu, de = agg[s]
    kind = "shunt (Y@V)" if s in SHUNT_STORES else "series (KCL)"
    print(f"  {s:12s} {kind:13s} WAPE = {nu/de:.3e}")
    tot_n += nu; tot_d += de
print(f"  {'AGGREGATE':12s} {'':13s} WAPE = {tot_n/tot_d:.3e}")
