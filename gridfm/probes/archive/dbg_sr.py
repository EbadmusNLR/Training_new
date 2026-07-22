"""Why is a SERIES reactor carrying REAL current reconstructed as exactly 0?

WAPE=1.000 with |I| >> noise means the conductor is never written. Check whether
its edge reaches the tree at all: in E? a tree edge? or a CHORD (mesh_correct
builds its loop impedance Z from LINE blocks only -- a non-line chord would be
dropped, and dropped == silently zero).
"""
import glob, os, sys
from collections import Counter
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, FC, store_size, stored_currents, element_currents
from gridfm.dk_tree import (reconstruct_full, build_recon_ctx, classify_series,
                            _series_edges, _slot_node_map, _slack_xfmrsec_roots,
                            TREE_STORES, SHUNT_STORES, AMBIG_STORES)

TD = "/kfs2/projects/gogpt/Ebadmus/training_data/minimal_component"
TGT = os.environ.get("TGT", "t1_2000_k71_load_wye_neutral_3")
p = [x for x in sorted(glob.glob(os.path.join(TD, "*", "static.pt")))
     if TGT in os.path.basename(os.path.dirname(x))][0]
fs = FeederScenarios(os.path.dirname(p))
d = fs[int(os.environ.get("VAR", "99"))]
print("feeder:", os.path.basename(os.path.dirname(p))[:60])

ser = classify_series(d, "reactor")
print(f"reactors: {store_size(d,'reactor')}  series: {sorted(ser)}")
m1 = _slot_node_map(d, "reactor", 1); m2 = _slot_node_map(d, "reactor", 2)
slack, xsec = _slack_xfmrsec_roots(d)
print(f"slack={sorted(slack)} xsec={sorted(xsec)[:8]}")

E = _series_edges(d, TREE_STORES)
ctx = build_recon_ctx(d)
tr = ctx["ltree"]
from gridfm.dk_tree import _SID
tree_set = {(int(s), int(c)) for s, c in zip(tr["sid"].tolist(), tr["comp"].tolist())}
E_reac = [e for e in E if e[0] == "reactor"]
print(f"\nseries edges total={len(E)}  reactor edges in E={len(E_reac)}")
print(f"tree edges={len(tr['sid'])}  chords={len(ctx['ltree'].get('chords', []))}")

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

sid_re = _SID.get("reactor")
print(f"\n{'comp':>5s} {'n1':>4s} {'n2':>4s} {'|I_true|':>11s} {'|I_rec|':>11s} "
      f"{'|I_phys|':>11s} {'in E':>5s} {'in tree':>8s}")
for c in sorted(ser):
    n1 = [v for (cc, sl), v in m1.items() if cc == c]
    n2 = [v for (cc, sl), v in m2.items() if cc == c]
    it = float(truth["reactor"][0][c].abs().sum() + truth["reactor"][1][c].abs().sum())
    ir = float(rec["reactor"][0][c].abs().sum() + rec["reactor"][1][c].abs().sum())
    ip = float(cur["reactor"][0][c].abs().sum() + cur["reactor"][1][c].abs().sum())
    inE = any(e[1] == c for e in E_reac)
    inT = (sid_re, c) in tree_set
    print(f"{c:5d} {str(n1[:2]):>4s} {str(n2[:2]):>4s} {it:11.4e} {ir:11.4e} "
          f"{ip:11.4e} {str(inE):>5s} {str(inT):>8s}")
