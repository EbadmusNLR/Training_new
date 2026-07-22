"""Does a BACKWARD-FORWARD SWEEP converge on this corpus, and how fast?

cond(Ybus) reaches 1e18, so no iterative scheme that looks like "invert Ybus" can
work in a handful of message-passing steps. The classical answer in distribution
power flow is NOT to invert Ybus: it is the ladder / backward-forward sweep

    backward:  shunt currents from V (physics decode)  ->  branch currents by tree KCL
    forward :  V_child = V_parent - Z_branch @ I_branch  (accumulate from the slack)

Both halves are O(1)-conditioned TREE accumulations, so the sweep's convergence is
governed by the shunt<->series coupling, not by cond(Ybus). We already have the
backward half exact (dk_tree). This measures the forward half + the fixed point:
how many sweeps to reach 1e-6 / 1e-10 in V, starting from a flat start.

If this converges in ~10 sweeps, a 12-step network CAN solve pf -- provided its
step is a sweep and not a generic Ybus relaxation. That is an architecture claim
worth testing before building it.
"""
import argparse, glob, os, sys
import numpy as np
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, FC, store_size, _y_full, terminal_slot, node_count

TD = "/kfs2/projects/gogpt/Ebadmus/training_data"


SHUNT = {"load", "capacitor", "pvsystem", "storage", "generator"}


def build_ybus(d, n, only=None):
    """Ybus and the Icomp rhs, assembled from every element's own Y.
    `only` = restrict to a subset of stores (to split series vs shunt)."""
    Ybus = np.zeros((n, n), dtype=np.complex128)
    rhs = np.zeros(n, dtype=np.complex128)
    for s in (STORES if only is None else only):
        if s not in d.node_types or store_size(d, s) == 0:
            continue
        prefix, nterm, _ = STORES[s]
        dim = nterm * FC
        Yr, Yi = _y_full(d[s], prefix, dim, torch.float64, store=s)
        Y = Yr.numpy() + 1j * Yi.numpy()
        st = d[s]
        ic = None
        if "Icomp_r_pu" in st:
            ic = (st["Icomp_r_pu"].reshape(-1, dim).double().numpy()
                  + 1j * st["Icomp_i_pu"].reshape(-1, dim).double().numpy())
        sn = {}
        for t in range(1, nterm + 1):
            rel = (s, f"bus{t}", "node")
            if rel not in d.edge_types or not d[rel].edge_index.numel():
                continue
            ei = d[rel].edge_index
            k = terminal_slot(ei[0])
            for c, kk, nd in zip(ei[0].tolist(), k.tolist(), ei[1].tolist()):
                sn[(int(c), (t - 1) * FC + int(kk))] = int(nd)
        for c in range(Y.shape[0]):
            for a in range(dim):
                na = sn.get((c, a))
                if na is None:
                    continue
                if ic is not None:
                    rhs[na] += ic[c, a]
                for b in range(dim):
                    nb = sn.get((c, b))
                    if nb is not None:
                        Ybus[na, nb] += Y[c, a, b]
    return Ybus, rhs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="SMART-DS_1000")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--sweeps", type=int, default=60)
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(TD, a.corpus, "*", "static.pt")))
    step = max(1, len(fs) // a.n)
    print(f"=== {a.corpus}: local relaxation vs the LADDER (series-solve) splitting ===")
    print("GJ/GS = Gauss-Jacobi/Seidel on Ybus = what a message-passing step can do.")
    print("LAD   = backward-forward sweep in matrix form: Y_series V^k+1 = Icomp - Y_shunt V^k")
    print("        (the series solve is a TREE accumulation: O(1)-conditioned, one pass)")
    print(f"{'feeder':30s} {'cond':>9s} {'GJ@60':>9s} {'GS@60':>9s} "
          f"{'LAD@3':>9s} {'LAD@10':>9s} {'rho_lad':>9s}")
    for p in fs[::step][:a.n]:
        d = FeederScenarios(os.path.dirname(p))[0]
        n = node_count(d)
        name = os.path.basename(os.path.dirname(p))
        Ybus, rhs = build_ybus(d, n)
        series = [s for s in STORES if s not in SHUNT]
        Yser, _ = build_ybus(d, n, only=series)
        Ysh, _ = build_ybus(d, n, only=[s for s in STORES if s in SHUNT])
        Vt = d["node"].V_r_pu.double().numpy() + 1j * d["node"].V_i_pu.double().numpy()
        Vi = (d["node"].V_r_init_pu.double().numpy()
              + 1j * d["node"].V_i_init_pu.double().numpy())
        vis = np.zeros(n, dtype=bool); vis[0] = True
        rel = ("vsource", "bus1", "node")
        if rel in d.edge_types and d[rel].edge_index.numel():
            vis[d[rel].edge_index[1].numpy()] = True
        free = np.where(~vis)[0]; fix = np.where(vis)[0]
        A = Ybus[np.ix_(free, free)]
        b = rhs[free] - Ybus[np.ix_(free, fix)] @ Vt[fix]
        den = np.abs(Vt[free]).sum() + 1e-30
        cond = np.linalg.cond(A)
        D = np.diag(np.diag(A))
        L = np.tril(A);
        out = {}
        for tag, M in (("GJ", D), ("GS", L)):
            V = Vi[free].copy(); errs = []
            try:
                for k in range(1, a.sweeps + 1):
                    r = b - A @ V
                    V = V + np.linalg.solve(M, r)
                    if k in (10, a.sweeps):
                        errs.append(np.abs(V - Vt[free]).sum() / den)
                    if not np.isfinite(V).all():
                        errs = [np.inf, np.inf]; break
            except Exception:
                errs = [np.inf, np.inf]
            out[tag] = (errs + [np.inf, np.inf])[:2]
        # LADDER: Y_series V^{k+1} = Icomp - Y_shunt V^k, slack fixed.
        # The series solve is what a tree sweep does in one pass; convergence is set
        # by rho(Yser^-1 Ysh) -- the shunt/series ratio (loads ~1e-2 vs lines ~1e6),
        # NOT by cond(Ybus). This is the claim the architecture rests on.
        As = Yser[np.ix_(free, free)]
        Ash = Ysh[np.ix_(free, free)]
        lad = [np.inf, np.inf]; rho = np.inf
        try:
            rho = np.abs(np.linalg.eigvals(np.linalg.solve(As, Ash))).max()
            V = Vi[free].copy(); errs = []
            bs = rhs[free] - Ybus[np.ix_(free, fix)] @ Vt[fix]
            for k in range(1, 11):
                V = np.linalg.solve(As, bs - Ash @ V)
                if k in (3, 10):
                    errs.append(np.abs(V - Vt[free]).sum() / den)
            lad = errs
        except Exception:
            pass
        print(f"{name[:28]:30s} {cond:9.2e} {out['GJ'][1]:9.2e} {out['GS'][1]:9.2e} "
              f"{lad[0]:9.2e} {lad[1]:9.2e} {rho:9.2e}")


if __name__ == "__main__":
    raise SystemExit(main())
