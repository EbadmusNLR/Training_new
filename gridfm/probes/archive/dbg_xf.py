"""Isolate the transformer null-space map on ONE feeder.

Splits the three candidate causes apart:
  (a) DATA/PHYSICS : does I = Y@V - Icomp at TRUTH V reproduce the stored current?
  (b) MAP          : does I_U = A@V_act + B@I_K reproduce I_U given TRUTH I_K?
  (c) I_K          : is the KCL-derived I_K itself right?
If (a) is exact and (b) fails with truth I_K, the map is at fault, not the data.
"""
import glob, os, sys
import numpy as np
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, FC, store_size, stored_currents, element_currents
from gridfm.dk_tree import build_xfmr_maps, _slot_node_map

TD = "/kfs2/projects/gogpt/Ebadmus/training_data/minimal_component"
TGT = os.environ.get("TGT", "t1_0208_k10_storage_de")
p = [x for x in sorted(glob.glob(os.path.join(TD, "*", "static.pt"))) if TGT in x][0]
d = FeederScenarios(os.path.dirname(p))[0]
print("feeder:", os.path.basename(os.path.dirname(p)))

vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
V = (vr + 1j * vi).numpy()
st = d["transformer"]
Yr = st["Yxfmr_r_pu"].reshape(-1, 12, 12).double().numpy()
Yi = st["Yxfmr_i_pu"].reshape(-1, 12, 12).double().numpy()
Tr, Ti = stored_currents(d, "transformer", dtype=torch.float64)
Ic_r = st["Icomp_r_pu"].reshape(-1, 12).double().numpy() if "Icomp_r_pu" in st else np.zeros((Yr.shape[0], 12))
Ic_i = st["Icomp_i_pu"].reshape(-1, 12).double().numpy() if "Icomp_i_pu" in st else np.zeros((Yr.shape[0], 12))

slot_node = {}
for t in (1, 2, 3):
    rel = ("transformer", f"bus{t}", "node")
    if rel not in d.edge_types or not d[rel].edge_index.numel():
        continue
    ei = d[rel].edge_index
    from gridfm.dk_physics import terminal_slot
    k = terminal_slot(ei[0])
    for c, kk, nd in zip(ei[0].tolist(), k.tolist(), ei[1].tolist()):
        slot_node[(int(c), (t - 1) * FC + int(kk))] = int(nd)

maps = {m["comp"]: m for m in build_xfmr_maps(d)}
print(f"transformers: {Yr.shape[0]}   maps built: {sorted(maps)}")

for row in range(Yr.shape[0]):
    Y = Yr[row].astype(np.complex128) + 1j * Yi[row]
    Ist = Tr[row].numpy() + 1j * Ti[row].numpy()
    Icomp = Ic_r[row] + 1j * Ic_i[row]
    diag = np.abs(np.diag(Y))
    act = [int(i) for i in np.where(diag > 1e-9 * diag.max())[0]]
    nodes = {s: slot_node.get((row, s), 0) for s in act}
    Vs = np.array([V[nodes[s]] if nodes[s] != 0 else 0j for s in act])
    print(f"\n--- transformer {row} ---")
    print(f"  act slots {act}")
    print(f"  slot->node {nodes}")
    print(f"  |I_stored| per slot: {np.round(np.abs(Ist), 8).tolist()}")
    # (a) DATA/PHYSICS: full 12x12 Y@V - Icomp at truth V
    Vfull = np.array([V[slot_node.get((row, s), 0)] if slot_node.get((row, s), 0) != 0 else 0j
                      for s in range(12)])
    Iphys = Y @ Vfull - Icomp
    den = np.abs(Ist).sum() + 1e-30
    print(f"  (a) DATA  |Y@V-Icomp - I_stored| / |I| = {np.abs(Iphys-Ist).sum()/den:.3e}")
    # null-space geometry: the map needs rank(N_U^T) == |U| to be determined
    Ya = Y[np.ix_(act, act)]
    _, S, Vh = np.linalg.svd(Ya)
    ratios = S / S.max()
    K_ = [s for s in act if s >= FC and nodes[s] != 0
          and sum(1 for q in act if nodes[q] == nodes[s]) == 1]
    U_ = [s for s in act if s not in K_]
    print(f"  sv ratios = {np.array2string(ratios, precision=3, formatter={'float_kind':lambda v: f'{v:.2e}'})}")
    for t in (1e-4, 1e-3, 1e-2, 1e-1):
        N = Vh[ratios < t].conj().T
        if N.shape[1] == 0:
            print(f"    thr={t:.0e}: null dim 0"); continue
        Ul_ = [act.index(s) for s in U_]
        r = np.linalg.matrix_rank(N[Ul_].T, tol=1e-10)
        print(f"    thr={t:.0e}: null dim {N.shape[1]:2d}  rank(N_U^T)={r}  |U|={len(U_)}"
              f"  {'DETERMINED' if r >= len(U_) else '*** UNDERDETERMINED ***'}")
    if row not in maps:
        print("  *** NO MAP BUILT for this transformer (null space empty or K/U empty)")
        # why?
        Ya = Y[np.ix_(act, act)]
        _, S, _ = np.linalg.svd(Ya)
        K = [s for s in act if s >= FC and nodes[s] != 0
             and sum(1 for q in act if nodes[q] == nodes[s]) == 1]
        U = [s for s in act if s not in K]
        print(f"      K={K}  U={U}  sv ratios={np.round(S/S.max(), 12).tolist()}")
        continue
    m = maps[row]
    K = m["K"].tolist(); U = m["U"].tolist()
    A = m["Ar"].numpy() + 1j * m["Ai"].numpy()
    B = m["Br"].numpy() + 1j * m["Bi"].numpy()
    print(f"  K(known from KCL)={K}  U(solved by map)={U}")
    # (b) MAP with TRUTH I_K
    IK = np.array([Ist[s] for s in K])
    IU_pred = A @ Vs + B @ IK
    IU_true = np.array([Ist[s] for s in U])
    dn = np.abs(IU_true).sum() + 1e-30
    print(f"  (b) MAP@truth I_K: |I_U_pred - I_U_true| / |I_U| = {np.abs(IU_pred-IU_true).sum()/dn:.3e}")
    print(f"      I_U true = {np.round(IU_true, 8).tolist()}")
    print(f"      I_U pred = {np.round(IU_pred, 8).tolist()}")
