"""Parameter estimation corner of the masking universe: recover a MISSING Y.

Scenario (user's): full powerflow state known (V everywhere, all Icomp), but the
Y matrix of one line and one transformer is unknown. Claim: this is LINEAR in the
missing Y entries -- the Y*V bilinearity needs BOTH factors masked -- so stacking
K operating points (Y is constant across variants) closes it exactly:

  1. every known component's terminal current = Y_loc V_loc - Icomp  (V known)
  2. KCL at the unknown element's nodes -> its terminal currents, exactly
  3. I_elem(k) = Y_elem V_elem(k), k=1..K  -> least squares for Y_elem rows

Report per feeder/element: Y relative error vs K, and the downstream V error
when the recovered Y is used in the full direct solve of a HELD-OUT variant.
"""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, FC, node_count, terminal_slot
from gridfm.tests.test_ladder import build_ybus, _y_full

TD = "/kfs2/projects/gogpt/Ebadmus/training_data"
CORPORA = ["SMART-DS_1000", "new_dss_data", "dss_data", "minimal_component"]


def elem_edges(d, s, c):
    """[(col, node)] for component c of store s across its terminals."""
    _, nterm, _ = STORES[s]
    out = []
    for t in range(1, nterm + 1):
        rel = (s, f"bus{t}", "node")
        if rel not in d.edge_types or not d[rel].edge_index.numel():
            continue
        ei = d[rel].edge_index
        kk = terminal_slot(ei[0])
        for cc, k, nd_i in zip(ei[0].tolist(), kk.tolist(), ei[1].tolist()):
            if cc == c:
                out.append(((t - 1) * FC + int(k), int(nd_i)))
    return out


def terminal_currents_all(d, n):
    """rr[node] accumulators are not needed; return per-(store,comp,col) current
    scatter as a dict node -> complex sum over ALL components' terminals."""
    tot = np.zeros(n, dtype=np.complex128)
    per = {}
    V = d["node"].V_r_pu.double().numpy() + 1j * d["node"].V_i_pu.double().numpy()
    for s in STORES:
        if s not in d.node_types:
            continue
        prefix, nterm, _ = STORES[s]
        st = d[s]
        if f"{prefix}_r_pu" not in st:
            continue
        dim = nterm * FC
        ncomp = st[f"{prefix}_r_pu"].shape[0]
        if not ncomp:
            continue
        Yr, Yi = _y_full(st, prefix, dim, __import__("torch").float64, store=s)
        Y = Yr.numpy() + 1j * Yi.numpy()
        ic = None
        if "Icomp_r_pu" in st:
            ic = (st["Icomp_r_pu"].reshape(ncomp, -1).double().numpy()
                  + 1j * st["Icomp_i_pu"].reshape(ncomp, -1).double().numpy())
        # local voltage vector per comp
        sn = {}
        for t in range(1, nterm + 1):
            rel = (s, f"bus{t}", "node")
            if rel not in d.edge_types or not d[rel].edge_index.numel():
                continue
            ei = d[rel].edge_index
            kk = terminal_slot(ei[0])
            for c, k, nd_i in zip(ei[0].tolist(), kk.tolist(), ei[1].tolist()):
                sn[(int(c), (t - 1) * FC + int(k))] = int(nd_i)
        Vloc = np.zeros((ncomp, dim), dtype=np.complex128)
        conn = np.zeros((ncomp, dim), dtype=bool)
        for (c, a), nd_i in sn.items():
            Vloc[c, a] = V[nd_i]; conn[c, a] = True
        I = np.einsum("cab,cb->ca", Y, Vloc)
        if ic is not None:
            w = min(ic.shape[1], dim)
            I[:, :w] -= ic[:, :w]
        per[s] = (I, sn, Vloc, conn, Y)
        for (c, a), nd_i in sn.items():
            tot[nd_i] += I[c, a]
    return per, tot, V


def recover(fdir, s_target, K_list=(1, 2, 4, 8), holdout=90):
    scen = FeederScenarios(fdir)
    d0 = scen[0]
    n = node_count(d0)
    if s_target not in d0.node_types:
        return None
    prefix, nterm, _ = STORES[s_target]
    if f"{prefix}_r_pu" not in d0[s_target] or d0[s_target][f"{prefix}_r_pu"].shape[0] == 0:
        return None
    dim = nterm * FC
    c = 0  # first component of the store = the "missing" element
    edges = elem_edges(d0, s_target, c)
    if not edges:
        return None
    import torch
    Yr, Yi = _y_full(d0[s_target], prefix, dim, torch.float64, store=s_target)
    Ytrue = (Yr.numpy() + 1j * Yi.numpy())[c]
    cols = [a for a, _ in edges]
    results = []
    rowsA, rowsb = {a: [] for a in cols}, {a: [] for a in cols}
    K_max = max(K_list)
    for k in range(K_max):
        d = scen[k]
        per, tot, V = terminal_currents_all(d, n)
        I_t, sn, Vloc, conn, _ = per[s_target]
        # element's terminal current via KCL: at the node of each of its
        # terminals, current = -(sum of every OTHER terminal there)
        for a, nd_i in edges:
            others = tot[nd_i] - I_t[c, a]
            i_elem = -others
            # equation: Y[a, cols] @ Vloc[c, cols] = i_elem  (only connected slots)
            rowsA[a].append(Vloc[c, cols])
            rowsb[a].append(i_elem)
        for K in K_list:
            if k + 1 != K:
                continue
            Yrec = np.zeros((dim, dim), dtype=np.complex128)
            for a in cols:
                A = np.array(rowsA[a]); b = np.array(rowsb[a])
                x, *_ = np.linalg.lstsq(A, b, rcond=None)
                Yrec[a, cols] = x
            den = np.abs(Ytrue[np.ix_(cols, cols)]).sum() + 1e-30
            yerr = np.abs(Yrec[np.ix_(cols, cols)] - Ytrue[np.ix_(cols, cols)]).sum() / den
            # downstream: solve a HELD-OUT variant with the recovered Y patched in
            dh = scen[holdout]
            Ybus, rhs = build_ybus(dh, n)
            dY = np.zeros((dim, dim), dtype=np.complex128)
            dY[np.ix_(cols, cols)] = Yrec[np.ix_(cols, cols)] - Ytrue[np.ix_(cols, cols)]
            nodes = {a: nd_i for a, nd_i in edges}
            for a in cols:
                for bcol in cols:
                    Ybus[nodes[a], nodes[bcol]] += dY[a, bcol]
            Vt = dh["node"].V_r_pu.double().numpy() + 1j * dh["node"].V_i_pu.double().numpy()
            Vi = (dh["node"].V_r_init_pu.double().numpy()
                  + 1j * dh["node"].V_i_init_pu.double().numpy())
            vis = np.zeros(n, dtype=bool); vis[0] = True
            rel = ("vsource", "bus1", "node")
            if rel in dh.edge_types and dh[rel].edge_index.numel():
                vis[dh[rel].edge_index[1].numpy()] = True
            free = np.where(~vis)[0]; fix = np.where(vis)[0]
            b2 = rhs[free] - Ybus[np.ix_(free, fix)] @ Vt[fix]
            Vs = np.linalg.solve(Ybus[np.ix_(free, free)], b2)
            vskill = np.abs(Vs - Vt[free]).sum() / (np.abs(Vt[free] - Vi[free]).sum() + 1e-30)
            results.append((K, yerr, vskill))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-corpus", type=int, default=3)
    a = ap.parse_args()
    print(f"{'corpus/feeder':40s} {'elem':12s} {'K':>3s} {'Y_relerr':>10s} {'V_skill@rec':>11s}")
    for corp in CORPORA:
        fs = sorted(glob.glob(os.path.join(TD, corp, "*", "static.pt")))
        step = max(1, len(fs) // a.per_corpus)
        picked = 0
        for p in fs[::step]:
            if picked >= a.per_corpus:
                break
            fdir = os.path.dirname(p)
            name = f"{corp}/{os.path.basename(fdir)}"[:40]
            any_row = False
            for s_target in ("line", "transformer"):
                try:
                    res = recover(fdir, s_target)
                except Exception as e:
                    print(f"{name:40s} {s_target:12s} FAIL {type(e).__name__}: {str(e)[:60]}")
                    continue
                if not res:
                    continue
                any_row = True
                for K, yerr, vskill in res:
                    print(f"{name:40s} {s_target:12s} {K:3d} {yerr:10.2e} {vskill:11.2e}",
                          flush=True)
            picked += 1 if any_row else 0
    print("\nEXPECTATION: Y_relerr -> machine precision once K x FC >= connected slots;"
          "\nV_skill@rec should track it. One snapshot (K=1) stays underdetermined ->"
          "\nthe NN-prior estate. Zero learned parameters used here.")


if __name__ == "__main__":
    raise SystemExit(main())
