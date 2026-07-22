"""Root-cause the two ladder anomalies (probe v2, T15):

  1. SMART-DS P10U: Icomp->V gain ~450 even with shunt-only perturbation.
     Which store/node carries the amplifying |rhs_sh| entry?
  2. minimal_component t1_1201_k44_transformer_*: ladder DIVERGES (5.7e+86).
     Is rho(Yser^-1 Ysh) > 1, and which store pushes it there? Candidate: a
     transformer magnetizing SHUNT classified into the series matrix (or vice
     versa) breaking the loads-are-small assumption.

Fix ideas to test in place: move offending small-shunt stores into Yser (any
store whose Y is node-diagonal can sit on either side; putting it in Yser keeps
the solve exact and shrinks Ysh -> rho down).
"""
import glob, os, sys
import numpy as np

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, node_count
from gridfm.tests.test_ladder import build_ybus, SHUNT

TD = "/kfs2/projects/gogpt/Ebadmus/training_data"
TARGETS = [
    ("SMART-DS_1000", "v1.0__peak__SFO__P10U*"),
    ("minimal_component", "t1_1201_k44_transformer*"),
]


def diag(fdir):
    d = FeederScenarios(fdir)[0]
    n = node_count(d)
    Ybus, rhs = build_ybus(d, n)
    series = [s for s in STORES if s not in SHUNT]
    Yser, _ = build_ybus(d, n, only=series)
    Ysh, _ = build_ybus(d, n, only=[s for s in STORES if s in SHUNT])
    Vt = d["node"].V_r_pu.double().numpy() + 1j * d["node"].V_i_pu.double().numpy()
    vis = np.zeros(n, dtype=bool); vis[0] = True
    rel = ("vsource", "bus1", "node")
    if rel in d.edge_types and d[rel].edge_index.numel():
        vis[d[rel].edge_index[1].numpy()] = True
    free = np.where(~vis)[0]; fix = np.where(vis)[0]
    As = Yser[np.ix_(free, free)]; Ash = Ysh[np.ix_(free, free)]
    print(f"\n=== {fdir}  n={n} free={free.size}")
    print("stores present:", {s: int((d[s].get("Y_r_pu", d[s].get("yr", None)) is not None)
          if hasattr(d[s], 'get') else 1) for s in STORES if s in d.node_types})
    # rho of the iteration matrix
    T = np.linalg.solve(As, Ash)
    ev = np.abs(np.linalg.eigvals(T))
    rho = ev.max()
    print(f"rho(Yser^-1 Ysh) = {rho:.3e}   (>1 = divergent)")
    # which nodes/stores dominate Ysh? per-store shunt magnitude at the worst rows
    if rho > 1 or True:
        k = min(5, free.size)
        # rows of T with largest row-sum = where the iteration amplifies
        rs = np.abs(T).sum(1)
        worst = free[np.argsort(-rs)[:k]]
        print(f"worst iteration rows (node ids): {worst.tolist()}  rowsum={np.sort(rs)[-k:][::-1]}")
        for s in sorted(SHUNT):
            Ys_only, _ = build_ybus(d, n, only=[s])
            m = np.abs(Ys_only[np.ix_(free, free)]).max() if free.size else 0.0
            if m > 0:
                print(f"  shunt store {s:10s}: max|Y| = {m:.3e}")
        for s in series:
            if s not in d.node_types:
                continue
            Ys_only, _ = build_ybus(d, n, only=[s])
            sub = Ys_only[np.ix_(free, free)]
            if np.abs(sub).max() > 0:
                dmin = np.abs(np.diag(sub))[np.abs(np.diag(sub)) > 0]
                print(f"  series store {s:10s}: max|Y| = {np.abs(sub).max():.3e}"
                      f"  min nonzero diag = {dmin.min() if dmin.size else float('nan'):.3e}")
    # gain localization: perturb ONE store at a time
    lu = np.linalg.inv(As)
    Vi = (d["node"].V_r_init_pu.double().numpy() + 1j * d["node"].V_i_init_pu.double().numpy())
    b0 = rhs[free] - Ybus[np.ix_(free, fix)] @ Vt[fix]
    den = np.abs(Vt[free]).sum() + 1e-30

    def ladder(bb):
        V = Vi[free].copy()
        for _ in range(15):
            V = lu @ (bb - Ash @ V)
        return V

    base = np.abs(ladder(b0) - Vt[free]).sum() / den
    print(f"ladder base err = {base:.2e}")
    rng = np.random.default_rng(0)
    for s in sorted(SHUNT):
        _, rhs_s = build_ybus(d, n, only=[s])
        if np.abs(rhs_s[free]).max() == 0:
            continue
        z = rng.standard_normal(free.size) + 1j * rng.standard_normal(free.size)
        pert = 1e-2 * np.abs(rhs_s[free]) * z / np.maximum(np.abs(z), 1e-30)
        err = np.abs(ladder(b0 + pert) - Vt[free]).sum() / den
        print(f"  gain from {s:10s} @1e-2: {err / 1e-2:10.2f}   max|rhs_s| = {np.abs(rhs_s[free]).max():.3e}")


def main():
    for corpus, pat in TARGETS:
        hits = sorted(glob.glob(os.path.join(TD, corpus, pat)))
        if not hits:
            print(f"NO MATCH {corpus}/{pat}")
            continue
        diag(hits[0])


if __name__ == "__main__":
    raise SystemExit(main())
