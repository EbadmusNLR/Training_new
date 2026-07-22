"""Type-agnostic transformer current map via the YPrim NULL SPACE.
Since I=Y@V, every valid current satisfies n^T I = 0 for n in null(YPrim) (the
amp-turn constraints - encode turns/polarity/connection automatically). Given the
SECONDARY currents (cheap from KCL), solve n^T I = 0 for the PRIMARY. Verify vs
stored primary. No stiff Y@V, no per-type branching."""
import glob, os, sys
import numpy as np
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
from Datakit.core.scenario_store import FeederScenarios
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from gridfm.dk_physics import store_size, stored_currents
import torch

ROOT = "/kfs2/projects/gogpt/Ebadmus/training_data/SMART-DS_1000"
feeders = sorted(glob.glob(os.path.join(ROOT, "*", "static.pt")), key=os.path.getsize)
probe = feeders[len(feeders)//2:len(feeders)//2+4]

num = den = 0.0; neg_gap = []
for p in probe:
    d = FeederScenarios(os.path.dirname(p))[0]
    if "transformer" not in d.node_types or store_size(d, "transformer") == 0:
        continue
    st = d["transformer"]
    Yr = st["Yxfmr_r_pu"].reshape(-1,12,12).numpy().astype(np.complex128)
    Yi = st["Yxfmr_i_pu"].reshape(-1,12,12).numpy()
    Y = Yr + 1j*Yi
    Ir, Ii = stored_currents(d, "transformer", dtype=torch.float64)
    I = (Ir.numpy() + 1j*Ii.numpy())
    for row in range(Y.shape[0]):
        Ym = Y[row]
        diag = np.abs(np.diag(Ym))
        act = np.where(diag > 1e-9*diag.max())[0]        # active conductors
        if len(act) == 0: continue
        Ya = Ym[np.ix_(act, act)]
        Ivec = I[row, act]
        prim = np.array([k for k,i in enumerate(act) if i < 4])
        sec  = np.array([k for k,i in enumerate(act) if i >= 4])
        if len(prim)==0 or len(sec)==0: continue
        # null space of Ya (Ya n = 0): small singular vectors
        U,S,Vh = np.linalg.svd(Ya)
        gap = S/ S.max()
        null_mask = gap < 1e-4
        if row==0 and len(neg_gap)<3:
            neg_gap.append((act.tolist(), np.round(gap,6).tolist()))
        N = Vh[null_mask].conj().T                        # columns: Ya@N~0
        if N.shape[1]==0: continue
        # constraints N^T I = 0 -> N[prim]^T I_prim = -N[sec]^T I_sec
        A = N[prim].T; b = -N[sec].T @ Ivec[sec]
        Iprim, *_ = np.linalg.lstsq(A, b, rcond=None)
        num += float(np.abs(Iprim - Ivec[prim]).sum())
        den += float(np.abs(Ivec[prim]).sum() + 1e-12)
print("singular-value gaps (active, S/Smax):")
for a,g in neg_gap: print(f"  active={a}\n    gaps={g}")
print(f"\nprimary-from-secondary (YPrim null-space) WAPE = {num/den:.3e}")
