"""Per-VARIANT scan of one feeder (variant 0 is never representative)."""
import glob, os, sys
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, store_size, stored_currents, element_currents
from gridfm.dk_tree import reconstruct_full, build_recon_ctx, SHUNT_STORES, AMBIG_STORES
TD = os.path.join("/kfs2/projects/gogpt/Ebadmus/training_data", os.environ.get("CORPUS","dss_data"))
TGT = os.environ.get("TGT", "case3_balanced_battery")
p = [x for x in sorted(glob.glob(os.path.join(TD, "*", "static.pt")))
     if TGT in os.path.basename(os.path.dirname(x))][0]
fs = FeederScenarios(os.path.dirname(p))
print("feeder:", os.path.basename(os.path.dirname(p))[:50], " variants:", len(fs))
ctx = None; rows = []
for v in range(len(fs)):
    d = fs[v]
    ctx = build_recon_ctx(d, topo=ctx)
    vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
    present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
    truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
    cur = {}
    for s in present:
        if s in SHUNT_STORES or s in AMBIG_STORES: cur[s] = element_currents(d, s, vr, vi)
        else:
            z = torch.zeros_like(truth[s][0]); cur[s] = (z, z.clone())
    rec = reconstruct_full(d, cur, vr, vi, ctx=ctx)
    per = {}
    tn = td = 0.0
    for s in present:
        R = rec.get(s, cur[s]); T = truth[s]
        num = float((R[0]-T[0]).abs().sum()+(R[1]-T[1]).abs().sum())
        den = float(T[0].abs().sum()+T[1].abs().sum())
        per[s] = (num/(den+1e-30), den); tn += num; td += den
    rows.append((tn/(td+1e-30), v, td, per))
rows.sort(reverse=True)
print(f"{'WAPE':>10s} {'var':>4s} {'|I|_total':>11s}   worst families")
for w, v, td, per in rows[:8]:
    bad = sorted(((x, s) for s, (x, d_) in per.items() if x > 1e-8), reverse=True)[:4]
    print(f"{w:10.3e} {v:4d} {td:11.4e}   " + " ".join(f"{s}={x:.2e}" for x, s in bad))
print(f"variants > 1e-6: {sum(1 for r in rows if r[0] > 1e-6)} / {len(rows)}")
