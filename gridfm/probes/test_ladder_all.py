"""Does the LADDER sweep solve pf across the WHOLE corpus, or only on the handful
of feeders I happened to look at?

Twice this session I generalised from n=1 and was wrong. The ladder result (3 sweeps
-> 1e-7) came from 8 feeders, so measure it on a real sample before anyone builds an
architecture on it. Sparse throughout, so big feeders are included rather than
quietly skipped -- a size-filtered "everything works" would be exactly the same trap.

Reports the distribution of sweeps-to-1e-6, rho_lad, and any feeder where the ladder
FAILS to converge (that is the interesting tail, not the mean).
"""
import argparse, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
TD = "/kfs2/projects/gogpt/Ebadmus/training_data"
SHUNT = {"load", "capacitor", "pvsystem", "storage", "generator"}


def one(args):
    path, nvar = args
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    import torch
    from core.scenario_store import FeederScenarios
    from gridfm.dk_physics import STORES, FC, store_size, _y_full, terminal_slot, node_count
    name = os.path.basename(os.path.dirname(path))

    def assemble(d, n, only):
        rows, cols, vals = [], [], []
        rhs = np.zeros(n, dtype=np.complex128)
        for s in only:
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
                        if nb is not None and Y[c, a, b] != 0:
                            rows.append(na); cols.append(nb); vals.append(Y[c, a, b])
        M = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.complex128).tocsr()
        return M, rhs

    try:
        fs = FeederScenarios(os.path.dirname(path))
        out = []
        for v in range(min(nvar, len(fs))):
            d = fs[v]
            n = node_count(d)
            allst = list(STORES)
            Yb, rhs = assemble(d, n, allst)
            Ys, _ = assemble(d, n, [s for s in allst if s not in SHUNT])
            Ysh, _ = assemble(d, n, [s for s in allst if s in SHUNT])
            Vt = (d["node"].V_r_pu.double().numpy() + 1j * d["node"].V_i_pu.double().numpy())
            Vi = (d["node"].V_r_init_pu.double().numpy()
                  + 1j * d["node"].V_i_init_pu.double().numpy())
            vis = np.zeros(n, dtype=bool); vis[0] = True
            rel = ("vsource", "bus1", "node")
            if rel in d.edge_types and d[rel].edge_index.numel():
                vis[d[rel].edge_index[1].numpy()] = True
            free = np.where(~vis)[0]; fix = np.where(vis)[0]
            if len(free) == 0:
                continue
            As = Ys[free][:, free].tocsc(); Ash = Ysh[free][:, free].tocsc()
            b = rhs[free] - (Yb[free][:, fix] @ Vt[fix])
            den = np.abs(Vt[free]).sum() + 1e-30
            if den < 1e-12:
                continue

            def sweep(M, N):
                """V <- M^-1 (b - N V). M is CONSTANT per sample (no V in Y), so its
                factorisation is precomputable; N carries the coupling."""
                lu = spla.splu(M.tocsc())
                V = Vi[free].copy()
                h6 = h9 = -1; e = np.inf
                for k in range(1, 31):
                    V = lu.solve(b - N @ V)
                    if not np.isfinite(V).all():
                        return np.inf, -1, -1
                    e = np.abs(V - Vt[free]).sum() / den
                    if h6 < 0 and e < 1e-6:
                        h6 = k
                    if h9 < 0 and e < 1e-9:
                        h9 = k; break
                return float(e), h6, h9

            # (1) plain ladder: series solve, all shunt on the rhs
            e30, hit6, hit9 = sweep(As, Ash)
            # (2) ladder + the shunt's OWN DIAGONAL folded into the solve matrix.
            # A grounded shunt (reactor/cap) is purely diagonal, so N becomes ~0 and
            # the stiff tail that made (1) DIVERGE should converge at once. A diagonal
            # addition does not break the tree structure, so this is still a sweep.
            dsh = sp.diags(Ash.diagonal())
            e30d, hit6d, hit9d = sweep((As + dsh).tocsc(), (Ash - dsh).tocsc())
            out.append({"n_free": int(len(free)), "err30": float(e30),
                        "sweeps_1e6": hit6, "sweeps_1e9": hit9,
                        "err30d": float(e30d), "sweeps_1e6d": hit6d,
                        "base": float(np.abs(Vi[free] - Vt[free]).sum() / den)})
        return {"name": name, "rows": out, "err": None}
    except Exception as e:
        return {"name": name, "rows": [], "err": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="SMART-DS_1000")
    ap.add_argument("--feeders", type=int, default=120)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(TD, a.corpus, "*", "static.pt")))
    step = max(1, len(fs) // a.feeders)
    sel = fs[::step][:a.feeders]
    res = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
        for r in ex.map(one, [(p, a.variants) for p in sel], chunksize=1):
            res.append(r)
    rows = [x for r in res for x in r["rows"]]
    fails = [r for r in res if r["err"]]
    if not rows:
        print(f"=== {a.corpus}: no rows ({len(fails)} failed)")
        for f in fails[:5]:
            print("   ", f["name"][:40], f["err"])
        return 0
    e30 = np.array([r["err30"] for r in rows])
    s6 = np.array([r["sweeps_1e6"] for r in rows])
    s9 = np.array([r["sweeps_1e9"] for r in rows])
    base = np.array([r["base"] for r in rows])
    print(f"=== {a.corpus}: LADDER on {len(rows)} samples / {len(res)} feeders "
          f"({len(fails)} build-failed) ===")
    print(f"  start (flat)      : {base.mean():.3e}   <- the dv=0 baseline the model sits at")
    print(f"  err after 30 sweeps: median={np.median(e30):.3e}  p95={np.percentile(e30,95):.3e}  "
          f"max={e30.max():.3e}")
    ok6 = s6 > 0; ok9 = s9 > 0
    print(f"  reached 1e-6      : {ok6.sum()}/{len(rows)}  median sweeps={np.median(s6[ok6]) if ok6.any() else -1:.0f}")
    print(f"  reached 1e-9      : {ok9.sum()}/{len(rows)}  median sweeps={np.median(s9[ok9]) if ok9.any() else -1:.0f}")
    bad = [r for r in rows if not (r["err30"] < 1e-6)]
    print(f"  DID NOT converge to 1e-6: {len(bad)} / {len(rows)}")
    for r in sorted(bad, key=lambda x: -x["err30"])[:5]:
        print(f"     n_free={r['n_free']:6d} err30={r['err30']:.3e} base={r['base']:.3e}"
              f"  -> with diag-shunt: {r.get('err30d', float('nan')):.3e}")
    # ladder + shunt diagonal folded into the solve matrix
    e30d = np.array([r["err30d"] for r in rows])
    s6d = np.array([r["sweeps_1e6d"] for r in rows])
    okd = s6d > 0
    badd = [r for r in rows if not (r["err30d"] < 1e-6)]
    print(f"  --- LADDER + diag(Y_shunt) folded into the series solve ---")
    print(f"  err after 30 sweeps: median={np.median(e30d):.3e}  p95={np.percentile(e30d,95):.3e}"
          f"  max={e30d.max():.3e}")
    print(f"  reached 1e-6      : {okd.sum()}/{len(rows)}  median sweeps="
          f"{np.median(s6d[okd]) if okd.any() else -1:.0f}")
    print(f"  DID NOT converge  : {len(badd)} / {len(rows)}   (plain ladder: {len(bad)})")
    for f in fails[:4]:
        print(f"  BUILD-FAIL {f['name'][:40]}: {f['err'][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
