"""Is the LADDER the machine-precision architecture, corpus-wide?

test_ladder.py showed the matrix-form ladder splitting

    Yser V^{k+1} = Icomp - Ysh V^k        (slack pinned)

reaches ~1e-9 in 10 iterations on SMART-DS (rho 0.03-0.18) where Jacobi/Seidel on
Ybus diverge. If that holds on ALL FOUR corpora, the model design flips:

    model predicts HIDDEN Icomp  ->  ladder solves V  (no learned V head at all)

Two claims to measure before wiring it into DKSolver:
  1. convergence: ladder V error at truth Icomp, per corpus (want ~1e-9)
  2. conditioning: perturb Icomp by relative eps -> resulting relative V error.
     Memory says shunt-I errors amplify ~1x (unlike V->I at 2e7x). If gain ~ O(1),
     a 1% Icomp estimator gives ~1% V -- and a 1e-6 estimator gives ~1e-6 V.
     THAT is the path to very-very-low error: precision limited only by the
     estimator, not the operator.
"""
import argparse, glob, os, sys, time
import numpy as np

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, node_count
from gridfm.tests.test_ladder import build_ybus, SHUNT

TD = "/kfs2/projects/gogpt/Ebadmus/training_data"
CORPORA = ["SMART-DS_1000", "new_dss_data", "dss_data", "minimal_component"]


def run_feeder(fdir, sweeps=15, eps_list=(1e-2, 1e-6)):
    d = FeederScenarios(fdir)[0]
    n = node_count(d)
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
    As = Yser[np.ix_(free, free)]
    Ash = Ysh[np.ix_(free, free)]
    den = np.abs(Vt[free]).sum() + 1e-30
    t0 = time.time()
    lu = np.linalg.inv(As)  # probe-only; production = sparse LU, factor once per variant

    def ladder(rhs_free):
        V = Vi[free].copy()
        for _ in range(sweeps):
            V = lu @ (rhs_free - Ash @ V)
        return V

    b0 = rhs[free] - Ybus[np.ix_(free, fix)] @ Vt[fix]
    V0 = ladder(b0)
    err0 = np.abs(V0 - Vt[free]).sum() / den
    # DIRECT solve of the full Ybus: the diag (T15) showed the wye-delta divergence is
    # As singular in the delta zero-seq mode, grounded only by the SHUNT load Y --
    # folding shunts into the solved matrix is exactly the direct solve. cond on this
    # corpus is ~4e9 (not the old 1e18), so fp64 direct may already be at 1e-7..1e-9.
    Vd = np.linalg.solve(Ybus[np.ix_(free, free)], b0)
    errd = np.abs(Vd - Vt[free]).sum() / den
    err0 = min(err0, np.inf)
    t_solve = time.time() - t0
    # conditioning: relative perturbation of the ESTIMAND -> relative V error.
    # The model only ever estimates SHUNT Icomp (loads/pv/storage/gen/caps); the
    # vsource head current is a known boundary, so perturbing it (first probe
    # version) measured a sensitivity the model never exposes (P10U gain 497).
    _, rhs_sh = build_ybus(d, n, only=[s for s in STORES if s in SHUNT])
    gains = []
    rng = np.random.default_rng(0)
    Aff = Ybus[np.ix_(free, free)]
    for eps in eps_list:
        z = rng.standard_normal(free.size) + 1j * rng.standard_normal(free.size)
        pert = eps * np.abs(rhs_sh[free]) * z / np.maximum(np.abs(z), 1e-30)
        Vp = np.linalg.solve(Aff, b0 + pert)   # gain measured through the DIRECT solve
        rel_v = np.abs(Vp - Vt[free]).sum() / den
        gains.append((eps, rel_v, rel_v / eps))
    return n, err0, errd, gains, t_solve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-corpus", type=int, default=5)
    ap.add_argument("--sweeps", type=int, default=15)
    a = ap.parse_args()
    print(f"{'corpus/feeder':44s} {'n':>6s} {'ladder_err':>11s} {'direct_err':>11s} "
          f"{'gain@1e-2':>10s} {'gain@1e-6':>10s} {'t(s)':>6s}")
    worst_lad, worst_dir, worst_gain = 0.0, 0.0, 0.0
    for c in CORPORA:
        fs = sorted(glob.glob(os.path.join(TD, c, "*", "static.pt")))
        step = max(1, len(fs) // a.per_corpus)
        for p in fs[::step][:a.per_corpus]:
            fdir = os.path.dirname(p)
            name = f"{c}/{os.path.basename(fdir)}"
            try:
                n, err0, errd, gains, ts = run_feeder(fdir, sweeps=a.sweeps)
            except Exception as e:
                print(f"{name[:44]:44s} FAIL {type(e).__name__}: {e}")
                continue
            g = {f"{eps:.0e}": gain for eps, _, gain in gains}
            print(f"{name[:44]:44s} {n:6d} {err0:11.2e} {errd:11.2e} "
                  f"{g.get('1e-02', float('nan')):10.2f} "
                  f"{g.get('1e-06', float('nan')):10.2f} {ts:6.1f}")
            worst_lad = max(worst_lad, err0)
            worst_dir = max(worst_dir, errd)
            worst_gain = max(worst_gain, max(gain for _, _, gain in gains))
    print(f"\nWORST ladder err = {worst_lad:.2e}   WORST direct err = {worst_dir:.2e}")
    print(f"WORST Icomp->V gain (direct) = {worst_gain:.2f}   (V->I was 2e7)")
    if worst_dir < 1e-6:
        print("VERDICT: DIRECT fp64 solve of full Ybus is machine-precision corpus-wide"
              " -- the solver layer is a single sparse solve; no splitting needed")
    elif worst_lad < 1e-6:
        print("VERDICT: ladder splitting needed (direct loses digits); fold shunt Y into As")
    else:
        print("VERDICT: neither holds -- keep digging")


if __name__ == "__main__":
    raise SystemExit(main())
