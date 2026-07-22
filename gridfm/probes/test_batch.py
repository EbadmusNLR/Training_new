"""Decoder validation on a RANDOM feeder sample (not cherry-picked), from V alone.

Also answers: is the residual fp32, or unreconstructed ground-touching conductors?
  - shunts hit ~4e-8 on the SAME fp32 V/Y -> fp32 supports 1e-8, so a 1e-3 residual
    is NOT precision.
  - reports the |I| share sitting in line conductors with a GROUND endpoint, which
    the tree excludes (node 0) and therefore never writes.
"""
import glob, os, sys, random
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
from Datakit.core.scenario_store import FeederScenarios
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from gridfm.dk_physics import STORES, store_size, stored_currents, element_currents
from gridfm.dk_tree import reconstruct_full, _series_edges, SHUNT_STORES, SERIES_STORES

ROOT = "/kfs2/projects/gogpt/Ebadmus/training_data/SMART-DS_1000"
feeders = sorted(glob.glob(os.path.join(ROOT, "*", "static.pt")))
random.seed(0)
probe = random.sample(feeders, 50)

USE_CHARGING = os.environ.get("NO_CHARGE", "0") != "1"
agg = {}
gnd_num = gnd_den = 0.0
nf = 0
for p in probe:
    try:
        d = FeederScenarios(os.path.dirname(p))[0]
    except Exception as e:
        print(f"skip {os.path.basename(os.path.dirname(p))[:20]}: {type(e).__name__}"); continue
    nf += 1
    vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
    present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
    truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
    cur = {}
    for s in present:
        if s in SHUNT_STORES:
            cur[s] = element_currents(d, s, vr, vi)
        else:
            z = torch.zeros_like(truth[s][0]); cur[s] = (z, z.clone())
    rec = reconstruct_full(d, cur, vr, vi) if USE_CHARGING else reconstruct_full(d, cur)
    for s in present:
        R = rec.get(s, cur[s]); T = truth[s]
        a = agg.setdefault(s, [0.0, 0.0])
        a[0] += float((R[0]-T[0]).abs().sum() + (R[1]-T[1]).abs().sum())
        a[1] += float(T[0].abs().sum() + T[1].abs().sum() + 1e-12)
    # |I| carried by GROUND-touching line conductors (tree never writes these)
    if "line" in truth:
        Tr, Ti = truth["line"]
        gnd_den += float(Tr.abs().sum() + Ti.abs().sum())
        for (s_, c, n1, n2, ca, cb) in _series_edges(d, ("line",)):
            if n1 == 0 or n2 == 0:
                gnd_num += float(Tr[c, ca].abs()+Ti[c, ca].abs()+Tr[c, cb].abs()+Ti[c, cb].abs())

print(f"=== decoder from V alone, {nf} RANDOM feeders, charging={USE_CHARGING} ===")
tot_n = tot_d = 0.0
for s in list(SHUNT_STORES) + list(SERIES_STORES):
    if s not in agg: continue
    nu, de = agg[s]
    kind = "shunt (Y@V)" if s in SHUNT_STORES else "series (KCL)"
    print(f"  {s:12s} {kind:13s} WAPE = {nu/de:.3e}")
    tot_n += nu; tot_d += de
print(f"  {'AGGREGATE':12s} {'':13s} WAPE = {tot_n/tot_d:.3e}")
print(f"\n  |I| in GROUND-touching line conductors (never written) = {gnd_num/(gnd_den+1e-12):.3e}")
