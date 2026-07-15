"""Is IEEE 30 Bus BAD DATA, or just a network my decoder cannot handle?

Those are different claims and I have only tested the second. The decoder refuses it at
ctx-build, so its physics was never validated. Check the data on its own terms:
  1. KCL at the STORED currents      -> should be ~0 if the solve is self-consistent
  2. I = Y@V - Icomp at truth V      -> should be ~1e-14 like every other feeder
  3. Ybus @ V = sum(Icomp)           -> should recover the hidden V (well-posedness)
If all three pass, the data is sound and the refusal is MY gap, not a corpus defect.
"""
import glob, os, sys
import numpy as np
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios
from gridfm.dk_physics import (STORES, FC, store_size, stored_currents,
                               element_currents, node_count)
from gridfm.dk_tree import _full_residual, SHUNT_STORES, AMBIG_STORES
from gridfm.test_ladder import build_ybus

TD = "/kfs2/projects/gogpt/Ebadmus/training_data/dss_data"
TGT = os.environ.get("TGT", "IEEE 30 Bus")
p = [x for x in sorted(glob.glob(os.path.join(TD, "*", "static.pt")))
     if TGT in os.path.basename(os.path.dirname(x))][0]
fs = FeederScenarios(os.path.dirname(p))
print("feeder:", os.path.basename(os.path.dirname(p)), "| variants:", len(fs))

for v in (0, 50, 99):
    if v >= len(fs):
        continue
    d = fs[v]
    n = node_count(d)
    present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
    truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
    vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()

    # 1. KCL at the stored currents
    out = {s: (truth[s][0].clone(), truth[s][1].clone()) for s in present}
    r = _full_residual(d, out, n)
    tot = sum(float(truth[s][0].abs().sum() + truth[s][1].abs().sum()) for s in present)
    kcl = float(r.abs().max())

    # 2. per-element physics I = Y@V - Icomp at truth V
    worst = []
    for s in present:
        if s in SHUNT_STORES or s in AMBIG_STORES:
            pr, pi = element_currents(d, s, vr, vi)
            num = float((pr - truth[s][0]).abs().sum() + (pi - truth[s][1]).abs().sum())
            den = float(truth[s][0].abs().sum() + truth[s][1].abs().sum()) + 1e-30
            worst.append((num / den, s))
    worst.sort(reverse=True)

    # 3. well-posedness: Ybus @ V = sum(Icomp)
    Ybus, rhs = build_ybus(d, n)
    Vt = vr.numpy() + 1j * vi.numpy()
    vis = np.zeros(n, dtype=bool); vis[0] = True
    rel = ("vsource", "bus1", "node")
    if rel in d.edge_types and d[rel].edge_index.numel():
        vis[d[rel].edge_index[1].numpy()] = True
    free = np.where(~vis)[0]; fix = np.where(vis)[0]
    A = Ybus[np.ix_(free, free)]
    b = rhs[free] - Ybus[np.ix_(free, fix)] @ Vt[fix]
    Vf = np.linalg.solve(A, b)
    verr = np.abs(Vf - Vt[free]).sum() / (np.abs(Vt[free]).sum() + 1e-30)
    vmag = np.abs(Vt[1:])

    print(f"\n--- variant {v}")
    print(f"  1. KCL at STORED currents : max|residual| = {kcl:.3e}   (|I| total {tot:.3e})")
    print(f"  2. shunt physics I=Y@V-Ic : worst = {worst[0][0]:.3e} ({worst[0][1]})"
          if worst else "  2. (no shunt families)")
    print(f"  3. Ybus@V = sum(Icomp)    : V recovered to {verr:.3e}")
    print(f"  V pu: min={vmag.min():.4f} max={vmag.max():.4f} "
          f"dead(<1e-6)={100*float((vmag<1e-6).mean()):.1f}%")
