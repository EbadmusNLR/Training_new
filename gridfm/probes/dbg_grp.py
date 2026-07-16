"""Per-transformer-group detail on one feeder: unknowns, KCL rows, chosen
constraint stiffness, and stored-vs-recon per component."""
import glob, os, sys
import numpy as np
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, FC, store_size, stored_currents, element_currents
from gridfm.dk_tree import (reconstruct_full, build_recon_ctx, build_xfmr_system,
                            _slack_xfmrsec_roots, _series_edges, _tree_from_edges,
                            TREE_STORES, SHUNT_STORES, AMBIG_STORES)

TD = os.path.join("/kfs2/projects/gogpt/Ebadmus/training_data",
                  os.environ.get("CORPUS", "dss_data"))
TGT = os.environ.get("TGT", "37Bus")
p = [x for x in sorted(glob.glob(os.path.join(TD, "*", "static.pt")))
     if TGT in os.path.basename(os.path.dirname(x))][0]
d = FeederScenarios(os.path.dirname(p))[int(os.environ.get("VAR", "0"))]
print("feeder:", os.path.basename(os.path.dirname(p))[:50])

uns = []
_E = _series_edges(d, TREE_STORES)
_sl, _xs = _slack_xfmrsec_roots(d)
_tr = _tree_from_edges(_E, _sl | _xs)
_br = [_E[i] for i in _tr["bridges"]]
print(f"bridges={len(_br)} chords={len(_tr['chords'])}")
groups = build_xfmr_system(d, unsolved=uns, bridges=_br)
print(f"groups={len(groups)}  unsolved={uns}")
for g in groups:
    nx = sum(len(v[0]) for v in g['scatter'].values())
    print(f"  comps={g['comps']}  unknowns={nx}  kcl_rows={g['nkcl']} "
          f"bridge_rows={g['nbridge']}  dir_rows={len(g['dirs'])}")
    print(f"      selected direction stiffness (sv/smax) = "
          f"{[f'{x:.2e}' for x in g['svs']]}")
    print(f"      cond(system) = {g['cond']:.3e}")

vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
cur = {}
for s in present:
    if s in SHUNT_STORES or s in AMBIG_STORES:
        cur[s] = element_currents(d, s, vr, vi)
    else:
        z = torch.zeros_like(truth[s][0]); cur[s] = (z, z.clone())
rec = reconstruct_full(d, cur, vr, vi)

# the EXACT physics reference: I = Y@V for a transformer (linear, no Icomp)
st = d["transformer"]
Y = (st["Yxfmr_r_pu"].reshape(-1, 12, 12).double() + 1j*st["Yxfmr_i_pu"].reshape(-1, 12, 12).double())
sn = {}
for t in (1, 2, 3):
    rel = ("transformer", f"bus{t}", "node")
    if rel in d.edge_types and d[rel].edge_index.numel():
        from gridfm.dk_physics import terminal_slot
        ei = d[rel].edge_index; k = terminal_slot(ei[0])
        for c, kk, nd in zip(ei[0].tolist(), k.tolist(), ei[1].tolist()):
            sn[(int(c), (t-1)*FC + int(kk))] = int(nd)
V = (vr + 1j*vi).numpy()

print(f"\n{'comp':>4s} {'|I_true|':>11s} {'|I_rec|':>11s} {'WAPE':>10s} {'phys(Y@V)':>11s}")
for c in range(store_size(d, "transformer")):
    T = truth["transformer"][0][c] + 1j*truth["transformer"][1][c]
    R = rec["transformer"][0][c] + 1j*rec["transformer"][1][c]
    Vv = np.array([V[sn.get((c, s), 0)] if sn.get((c, s), 0) != 0 else 0j for s in range(12)])
    Iph = (Y[c].numpy() @ Vv)
    den = float(T.abs().sum()) + 1e-30
    print(f"{c:4d} {den:11.4e} {float(R.abs().sum()):11.4e} "
          f"{float((R-T).abs().sum())/den:10.3e} {np.abs(Iph-T.numpy()).sum()/den:11.3e}")
    if float((R-T).abs().sum())/den > 1e-6:
        print(f"      true={np.round(T.numpy(), 6).tolist()}")
        print(f"      rec ={np.round(R.numpy(), 6).tolist()}")
