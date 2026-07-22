"""Which EQUATION is wrong? Substitute the TRUE currents into the group's own rows.

The system is rank-full and well-conditioned and the data is exact, so if x_true does
not satisfy R @ x = b, the defect is in a ROW or its RHS -- not in the solve. Checks
the KCL rows and the (Yn)^T V rows separately, using truth everywhere.
"""
import glob, os, sys
import numpy as np
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, FC, store_size, stored_currents, element_currents
from gridfm.dk_tree import (build_xfmr_system, _full_residual, _inj_index,
                            SHUNT_STORES, AMBIG_STORES)

TD = os.path.join("/kfs2/projects/gogpt/Ebadmus/training_data",
                  os.environ.get("CORPUS", "dss_data"))
TGT = os.environ.get("TGT", "37Bus")
p = [x for x in sorted(glob.glob(os.path.join(TD, "*", "static.pt")))
     if TGT in os.path.basename(os.path.dirname(x))][0]
d = FeederScenarios(os.path.dirname(p))[0]
print("feeder:", os.path.basename(os.path.dirname(p))[:50])

vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
n = d["node"].V_r_pu.shape[0]
present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}

# TRUE state everywhere: every store at its stored current
out_true = {s: (truth[s][0].clone(), truth[s][1].clone()) for s in present}
# ... and the same with transformers zeroed, which is what _apply_xfmr_system uses
out_zero = {s: (truth[s][0].clone(), truth[s][1].clone()) for s in present}
out_zero["transformer"] = (torch.zeros_like(truth["transformer"][0]),
                           torch.zeros_like(truth["transformer"][1]))
r_all = _full_residual(d, out_true, n)
r0 = _full_residual(d, out_zero, n)
print(f"KCL residual at ALL nodes with TRUE currents (should be ~0): "
      f"max={float(r_all.abs().max()):.3e}")

groups = build_xfmr_system(d)
V = (vr + 1j * vi).numpy()
Tr, Ti = truth["transformer"]
for g in groups:
    xt = np.array([complex(Tr[c, s], Ti[c, s])
                   for c, s in zip(g["ci"].tolist(), g["si"].tolist())])
    nk = g["nkcl"]
    b = np.zeros(nk + len(g["dirs"]), dtype=np.complex128)
    kn = g["knodes"].tolist()
    b[:nk] = -(r0[kn, 0].numpy() + 1j * r0[kn, 1].numpy())
    for j, (nd, Ynr, Yni) in enumerate(g["dirs"]):
        Yn = Ynr.numpy() + 1j * Yni.numpy()
        b[nk + j] = Yn @ V[nd.tolist()]
    # rebuild R exactly as the solver did: pinv(R) was stored, so recover R rows
    # by re-deriving them from the same definitions
    Nx = len(xt)
    idx = {(c, s): i for i, (c, s) in enumerate(zip(g["ci"].tolist(), g["si"].tolist()))}
    R = np.zeros((nk + len(g["dirs"]), Nx), dtype=np.complex128)
    for i, nd in enumerate(kn):
        for (c, s), k in idx.items():
            # conductor (c,s) sits on node nd?
            pass
    print(f"\n  group comps={g['comps']}  unknowns={Nx}  kcl={nk}")
    print(f"    KCL rows: rhs(-r0 at knodes) = {np.round(b[:nk], 8).tolist()}")
    # what the TRUE transformer currents actually sum to at each knode
    tsum = {}
    comp, col, node = _inj_index(d, "transformer")
    for c_, col_, nd_ in zip(comp.tolist(), col.tolist(), node.tolist()):
        if nd_ in kn:
            tsum[nd_] = tsum.get(nd_, 0j) + complex(Tr[c_, col_], Ti[c_, col_])
    print(f"    TRUE sum of transformer conductors at knodes = "
          f"{[np.round(tsum.get(k_, 0j), 8) for k_ in kn]}")
    resid = [abs(tsum.get(k_, 0j) - b[i]) for i, k_ in enumerate(kn)]
    print(f"    |KCL row violation by TRUTH| = {[f'{x:.2e}' for x in resid]}")
    dv = []
    for j, (nd, Ynr, Yni) in enumerate(g["dirs"]):
        # n^T I_t over that transformer's act slots
        pass
    print(f"    (Yn)^T V rows: rhs = {[f'{abs(x):.3e}' for x in b[nk:]]}")
