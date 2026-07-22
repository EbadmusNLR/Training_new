"""Single-feeder decode probe: per-family WAPE + why the xfmr system is short.

IEEE 30 Bus is the only feeder the decoder refuses. It is MESHED through
transformers, so it is the test case for the loop path (loop_dof + mesh_correct),
not a curiosity -- more data is coming and a silent refusal is a landmine.

  TGT="IEEE 30 Bus" python gridfm/dbg_30.py
"""
import glob, os, sys
import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, store_size, stored_currents, element_currents
from gridfm.dk_tree import (reconstruct_full, build_recon_ctx, SHUNT_STORES,
                            AMBIG_STORES, UnsupportedNetwork)

CORPUS = os.environ.get("CORPUS", "dss_data")
TD = f"/kfs2/projects/gogpt/Ebadmus/training_data/{CORPUS}"
TGT = os.environ.get("TGT", "IEEE 30 Bus")
NV = int(os.environ.get("NV", "3"))

paths = [x for x in sorted(glob.glob(os.path.join(TD, "*", "static.pt")))
         if TGT in os.path.basename(os.path.dirname(x))]
if not paths:
    raise SystemExit(f"no feeder matching {TGT!r} in {TD}")
fs = FeederScenarios(os.path.dirname(paths[0]))
print(f"feeder: {os.path.basename(os.path.dirname(paths[0]))} | variants: {len(fs)}")

ctx = None
for v in range(min(NV, len(fs))):
    d = fs[v]
    try:
        ctx = build_recon_ctx(d, topo=ctx)
    except UnsupportedNetwork as e:
        print(f"--- variant {v}: REFUSED: {str(e)[:200]}")
        continue
    lt = ctx["ltree"]
    print(f"--- variant {v}: mchords={len(lt.get('mchords', []))} "
          f"chords={len(lt.get('chords', []))} bridges={len(ctx.get('bridges', []))} "
          f"groups={len(ctx['xmaps'])}")
    vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
    present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
    truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
    cur = {}
    for s in present:
        # AMBIG too (shunt reactor/capacitor): recon keeps what it is handed
        if s in SHUNT_STORES or s in AMBIG_STORES:
            cur[s] = element_currents(d, s, vr, vi)
        else:
            z = torch.zeros_like(truth[s][0]); cur[s] = (z, z.clone())
    rec = reconstruct_full(d, cur, vr, vi, ctx=ctx)
    tn = td = 0.0
    for s in present:
        R = rec.get(s, cur[s]); T = truth[s]
        num = float((R[0] - T[0]).abs().sum() + (R[1] - T[1]).abs().sum())
        den = float(T[0].abs().sum() + T[1].abs().sum())
        tn += num; td += den
        print(f"      {s:12s} WAPE {num/(den+1e-30):.3e}   (|I| {den:.3e})")
    print(f"      {'TOTAL':12s} WAPE {tn/(td+1e-30):.3e}")
