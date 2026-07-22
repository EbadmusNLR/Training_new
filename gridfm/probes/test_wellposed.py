"""Is the `pf` task well-posed -- do the VISIBLE entries determine the target V?

mask_pf hides only non-slack V; every Y and Icomp is observed. Since I = Y@V - Icomp
for every element and KCL is sum(I)=0 at each node, the visible data implies

    Ybus @ V = sum(Icomp)          (LINEAR in V)

so V is recoverable by a direct solve. Verify that numerically before blaming the
architecture for the 4% V error: if this does NOT reproduce V, no model can, and the
task/mask is the bug (it was once: hiding Icomp made pf unsolvable at 10% V).

Also reports the CONDITION NUMBER -- what a learned iterative solver is up against.
"""
import argparse, glob, os, sys
import numpy as np
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, FC, store_size, _y_full, terminal_slot, node_count

TD = "/kfs2/projects/gogpt/Ebadmus/training_data"


def slot_nodes(d, store, nterm):
    sn = {}
    for t in range(1, nterm + 1):
        rel = (store, f"bus{t}", "node")
        if rel not in d.edge_types or not d[rel].edge_index.numel():
            continue
        ei = d[rel].edge_index
        k = terminal_slot(ei[0])
        for c, kk, nd in zip(ei[0].tolist(), k.tolist(), ei[1].tolist()):
            sn[(int(c), (t - 1) * FC + int(kk))] = int(nd)
    return sn


def one(path, nvar=3):
    fs = FeederScenarios(os.path.dirname(path))
    name = os.path.basename(os.path.dirname(path))
    out = []
    for v in range(min(nvar, len(fs))):
        d = fs[v]
        n = node_count(d)
        Ybus = np.zeros((n, n), dtype=np.complex128)
        rhs = np.zeros(n, dtype=np.complex128)
        for s in STORES:
            if s not in d.node_types or store_size(d, s) == 0:
                continue
            prefix, nterm, _ = STORES[s]
            dim = nterm * FC
            Yr, Yi = _y_full(d[s], prefix, dim, torch.float64, store=s)
            Y = Yr.numpy() + 1j * Yi.numpy()
            sn = slot_nodes(d, s, nterm)
            st = d[s]
            ic = None
            if f"Icomp_r_pu" in st:
                ic = (st["Icomp_r_pu"].reshape(-1, dim).double().numpy()
                      + 1j * st["Icomp_i_pu"].reshape(-1, dim).double().numpy())
            for c in range(Y.shape[0]):
                for a in range(dim):
                    na = sn.get((c, a))
                    if na is None:
                        continue
                    if ic is not None:
                        rhs[na] += ic[c, a]
                    for b in range(dim):
                        nb = sn.get((c, b))
                        if nb is None:
                            continue
                        Ybus[na, nb] += Y[c, a, b]
        Vt = (d["node"].V_r_pu.double().numpy() + 1j * d["node"].V_i_pu.double().numpy())
        vis = np.zeros(n, dtype=bool)
        vis[0] = True                                     # ground
        rel = ("vsource", "bus1", "node")                 # slack
        if rel in d.edge_types and d[rel].edge_index.numel():
            vis[d[rel].edge_index[1].numpy()] = True
        free = np.where(~vis)[0]
        fix = np.where(vis)[0]
        A = Ybus[np.ix_(free, free)]
        b = rhs[free] - Ybus[np.ix_(free, fix)] @ Vt[fix]
        Vf = np.linalg.solve(A, b)
        err = np.abs(Vf - Vt[free]).sum() / (np.abs(Vt[free]).sum() + 1e-30)
        out.append((err, np.linalg.cond(A), len(free)))
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="SMART-DS_1000")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(TD, a.corpus, "*", "static.pt")))
    step = max(1, len(fs) // a.n)
    print(f"=== {a.corpus}: does Ybus@V = sum(Icomp) recover the hidden V? ===")
    print(f"{'feeder':40s} {'V rel err':>11s} {'cond(Ybus)':>11s} {'n_free':>7s}")
    for p in fs[::step][:a.n]:
        name, rows = one(p)
        for err, cond, nf in rows[:1]:
            print(f"{name[:38]:40s} {err:11.3e} {cond:11.3e} {nf:7d}")


if __name__ == "__main__":
    raise SystemExit(main())
