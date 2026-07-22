"""Fast LADDER check: does the backward-forward sweep converge, and how fast?

    Y_series V^{k+1} = sum(Icomp) - Y_shunt V^k        (slack fixed)

This is the backward-forward (ladder) sweep in matrix form: the series solve is
exactly what a TREE sweep does in one pass (O(1)-conditioned forward/backward
substitution), and the shunt term is the physics decode we already do exactly.
Its convergence is rho(Yser^-1 Ysh) -- the shunt/series admittance ratio (loads
~1e-2 vs lines ~1e6) -- NOT cond(Ybus). If rho << 1, a handful of sweeps solves
pf on an operator where Gauss-Jacobi diverges and Gauss-Seidel stalls.

Small feeders only: rho is a physical ratio, not a size effect, and the 60-sweep
GS reference costs O(n^3) per iteration.
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import node_count, STORES
from gridfm.test_ladder import build_ybus, SHUNT

corpus = sys.argv[1] if len(sys.argv) > 1 else "SMART-DS_1000"
TD = "/kfs2/projects/gogpt/Ebadmus/training_data"
fs = sorted(glob.glob(os.path.join(TD, corpus, "*", "static.pt")))
sizes = []
for p in fs[::max(1, len(fs) // 60)][:60]:
    try:
        sizes.append((node_count(FeederScenarios(os.path.dirname(p))[0]), p))
    except Exception:
        pass
sizes.sort()
print(f"=== {corpus}: LADDER (series-solve) vs local relaxation ===")
print(f"{'feeder':24s} {'n':>5s} {'cond':>9s} {'rho_lad':>9s} {'LAD@1':>9s} "
      f"{'LAD@3':>9s} {'LAD@10':>9s} {'GS@60':>9s}")
for _n, p in sizes[:5]:
    d = FeederScenarios(os.path.dirname(p))[0]
    n = node_count(d)
    name = os.path.basename(os.path.dirname(p))[:22]
    Ybus, rhs = build_ybus(d, n)
    Yser, _ = build_ybus(d, n, only=[s for s in STORES if s not in SHUNT])
    Ysh, _ = build_ybus(d, n, only=[s for s in STORES if s in SHUNT])
    Vt = d["node"].V_r_pu.double().numpy() + 1j * d["node"].V_i_pu.double().numpy()
    Vi = d["node"].V_r_init_pu.double().numpy() + 1j * d["node"].V_i_init_pu.double().numpy()
    vis = np.zeros(n, dtype=bool); vis[0] = True
    rel = ("vsource", "bus1", "node")
    if rel in d.edge_types and d[rel].edge_index.numel():
        vis[d[rel].edge_index[1].numpy()] = True
    free = np.where(~vis)[0]; fix = np.where(vis)[0]
    A = Ybus[np.ix_(free, free)]; As = Yser[np.ix_(free, free)]; Ash = Ysh[np.ix_(free, free)]
    b = rhs[free] - Ybus[np.ix_(free, fix)] @ Vt[fix]
    den = np.abs(Vt[free]).sum() + 1e-30
    cond = np.linalg.cond(A)
    try:
        rho = np.abs(np.linalg.eigvals(np.linalg.solve(As, Ash))).max()
    except Exception:
        rho = np.inf
    errs = {}
    V = Vi[free].copy()
    try:
        for k in range(1, 11):
            V = np.linalg.solve(As, b - Ash @ V)
            if k in (1, 3, 10):
                errs[k] = np.abs(V - Vt[free]).sum() / den
    except Exception:
        errs = {1: np.inf, 3: np.inf, 10: np.inf}
    L = np.tril(A); V = Vi[free].copy(); gs = np.inf
    try:
        for _k in range(60):
            V = V + np.linalg.solve(L, b - A @ V)
        gs = np.abs(V - Vt[free]).sum() / den
    except Exception:
        pass
    print(f"{name:24s} {len(free):5d} {cond:9.2e} {rho:9.2e} {errs.get(1, np.inf):9.2e} "
          f"{errs.get(3, np.inf):9.2e} {errs.get(10, np.inf):9.2e} {gs:9.2e}")
