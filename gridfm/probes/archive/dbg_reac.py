"""Series reactors: the reactor residual (3.967e-04) is IDENTICAL across three runs
that changed the transformer path, so it has its own cause.

Shunt reactors physics-decode exactly; SERIES ones (both terminals live) are routed
through the tree. Per element, compare:
   tree reconstruction   vs   the exact physics decode I = Y@V - Icomp at truth V
and scan variants (variant 0 alone is NOT representative -- taps vary).
"""
import glob, os, sys
from collections import Counter
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, FC, store_size, stored_currents, element_currents
from gridfm.dk_tree import (reconstruct_full, build_recon_ctx, classify_series,
                            _slot_node_map, SHUNT_STORES, AMBIG_STORES)

TD = "/kfs2/projects/gogpt/Ebadmus/training_data/minimal_component"
NF = int(os.environ.get("NF", "40"))
NV = int(os.environ.get("NV", "8"))

tot = Counter()
agg = {"series": [0.0, 0.0], "shunt": [0.0, 0.0], "phys": [0.0, 0.0]}
worst = []
for p in sorted(glob.glob(os.path.join(TD, "*", "static.pt")))[:NF]:
    fs = FeederScenarios(os.path.dirname(p))
    d0 = fs[0]
    if "reactor" not in d0.node_types or store_size(d0, "reactor") == 0:
        continue
    ser = classify_series(d0, "reactor")
    n = store_size(d0, "reactor")
    tot["reactors"] += n; tot["series"] += len(ser); tot["shunt"] += n - len(ser)
    ctx = None
    for v in range(min(NV, len(fs))):
        d = fs[v]
        ctx = build_recon_ctx(d, topo=ctx)
        vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
        present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
        truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
        cur = {}
        for s in present:
            if s in SHUNT_STORES or s in AMBIG_STORES:
                cur[s] = element_currents(d, s, vr, vi)
            else:
                z = torch.zeros_like(truth[s][0]); cur[s] = (z, z.clone())
        rec = reconstruct_full(d, cur, vr, vi, ctx=ctx)
        R, T = rec["reactor"], truth["reactor"]
        P = cur["reactor"]          # the pure physics decode I = Y@V - Icomp
        for c in range(n):
            grp = "series" if c in ser else "shunt"
            num = float((R[0][c]-T[0][c]).abs().sum() + (R[1][c]-T[1][c]).abs().sum())
            den = float(T[0][c].abs().sum() + T[1][c].abs().sum())
            agg[grp][0] += num; agg[grp][1] += den
            pn = float((P[0][c]-T[0][c]).abs().sum() + (P[1][c]-T[1][c]).abs().sum())
            agg["phys"][0] += pn; agg["phys"][1] += den
            if grp == "series" and den > 0 and num/den > 1e-6:
                worst.append((num/den, os.path.basename(os.path.dirname(p))[:30], v, c,
                              pn/(den+1e-30)))

print(f"feeders scanned with reactors: {tot['reactors']} reactors "
      f"({tot['series']} series / {tot['shunt']} shunt)")
for k in ("shunt", "series", "phys"):
    num, den = agg[k]
    print(f"  {k:8s} WAPE = {num/(den+1e-30):.3e}   (|I| = {den:.4e})")
worst.sort(reverse=True)
print("\n  worst SERIES reactors (tree WAPE | same element via pure physics decode):")
for w, nm, v, c, pw in worst[:10]:
    print(f"    {nm:32s} var{v:<3d} comp{c:<3d} tree={w:.3e}  phys={pw:.3e}")
