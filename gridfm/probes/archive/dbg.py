import glob, os, sys
from collections import deque, defaultdict
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, FC, store_size, stored_currents, element_currents
from gridfm.dk_tree import (reconstruct_full, build_recon_ctx, _series_edges,
                            _slot_node_map, _slack_xfmrsec_roots, SHUNT_STORES, AMBIG_STORES)
TD=os.path.join("/kfs2/projects/gogpt/Ebadmus/training_data",
                os.environ.get("CORPUS","minimal_component"))
import re
TGT = os.environ.get("TGT","t1_0208_k10_storage_de")
p = [x for x in sorted(glob.glob(os.path.join(TD, "*", "static.pt"))) if TGT in x][0]
fs = FeederScenarios(os.path.dirname(p))
d = fs[int(os.environ.get("VAR","0"))]
print("feeder:", os.path.basename(os.path.dirname(p)))
print("nodes:", d["node"].V_r_pu.shape[0])
for s in d.node_types:
    if s!="node": print(f"  {s:12s} n={store_size(d,s)}")
slack, xsec = _slack_xfmrsec_roots(d)
print("slack nodes:", sorted(slack), " xfmr-sec nodes:", sorted(xsec))
for s in ("line","transformer","vsource","reactor","capacitor"):
    if s not in d.node_types or store_size(d,s)==0: continue
    for t in (1,2,3):
        m = _slot_node_map(d, s, t)
        if m: print(f"  {s}.bus{t}: {sorted(m.items())[:6]}")
E = _series_edges(d, ("line","reactor","capacitor"))
print("series edges (store,comp,n1,n2,cola,colb):", E[:8])
vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
present=[s for s in STORES if s in d.node_types and store_size(d,s)>0]
truth={s: stored_currents(d,s,dtype=torch.float64) for s in present}
cur={}
for s in present:
    if s in SHUNT_STORES or s in AMBIG_STORES: cur[s]=element_currents(d,s,vr,vi)
    else:
        z=torch.zeros_like(truth[s][0]); cur[s]=(z,z.clone())
rec = reconstruct_full(d, cur, vr, vi)
print("\nper-store:")
for s in present:
    R=rec.get(s,cur[s]); T=truth[s]
    num=float((R[0]-T[0]).abs().sum()+(R[1]-T[1]).abs().sum()); den=float(T[0].abs().sum()+T[1].abs().sum())
    print(f"  {s:12s} WAPE={num/(den+1e-30):.3e}  |I|={den:.4e}")
    if s in ("line","transformer","vsource") and den>0:
        print(f"      stored[0]={[round(float(x),6) for x in T[0][0]]}")
        print(f"      recon [0]={[round(float(x),6) for x in R[0][0]]}")
