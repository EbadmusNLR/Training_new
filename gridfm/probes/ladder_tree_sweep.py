"""Does the ladder converge using ONLY tree accumulations -- no linear solver?

test_ladder.py proved the ladder splitting converges to ~1e-09 in 10 sweeps on SMART-DS
(rho = 0.03..0.18, independent of cond(Ybus) = up to 2.7e18, while Gauss-Jacobi diverges).
But it did that with a dense `np.linalg.solve(Yser, ...)` per sweep. The model may not call
a solver: that is an oracle, and the contract is explicit that the structural sweep "must
not read solved voltage or call a PF/linear solver".

The O(n) equivalent is a rooted-tree sweep, which is what a message-passing step can be:

    backward:  I_shunt = Y@V - Icomp        (physics decode, well-conditioned)
               I_series = subtree KCL        (dk_tree.reconstruct_full -- already exact)
    forward :  V_child  = B^-1 (I_bus1 - A@V_parent)   per series element, root -> leaf

The forward step uses the element's OWN primitive: for a 2-terminal series element,
I_bus1 = A@V1 + B@V2, so V2 = B^-1(I_bus1 - A@V1). That is general -- it covers lines
(B = -Ys) and transformers (B = the off-diagonal block, carrying the tap ratio) without
special-casing either, and needs no separate impedance table.

Iterating the two from a FLAT start is the classical backward-forward sweep. If it reaches
~1e-09 with tree ops only, the architecture claim is proven end-to-end: a 12-step network
can solve pf to machine precision provided its step is a sweep.

    python -m gridfm.probes.ladder_tree_sweep --corpus SMART-DS_1000 --n 4
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")

from core.scenario_store import FeederScenarios
from gridfm.dk_physics import (STORES, FC, store_size, node_count, terminal_slot,
                               element_currents, stored_currents, _y_full)
from gridfm.dk_tree import (build_recon_ctx, reconstruct_full, SHUNT_STORES,
                           AMBIG_STORES, _SID_INV)

TD = "/kfs2/projects/gogpt/Ebadmus/training_data"
SERIES_2T = ("line", "transformer")


def slot_nodes(d, s, nterm):
    """(comp, col) -> node for every terminal slot of a store."""
    sn = {}
    for t in range(1, nterm + 1):
        rel = (s, f"bus{t}", "node")
        if rel not in d.edge_types or not d[rel].edge_index.numel():
            continue
        ei = d[rel].edge_index
        k = terminal_slot(ei[0])
        for c, kk, nd in zip(ei[0].tolist(), k.tolist(), ei[1].tolist()):
            sn[(int(c), (t - 1) * FC + int(kk))] = int(nd)
    return sn


def sweep_tables(d):
    """Per-store Y blocks and slot->node maps, built ONCE.

    Building these inside the level loop rebuilds the whole store's [n,8,8] Y for every
    component at every level -- thousands of full rebuilds per sweep on a real feeder.
    """
    tab = {}
    for s in SERIES_2T:
        if s not in d.node_types or store_size(d, s) == 0:
            continue
        prefix, nterm, _ = STORES[s]
        dim = nterm * FC
        Yr, Yi = _y_full(d[s], prefix, dim, torch.float64, store=s)
        Y = Yr.numpy() + 1j * Yi.numpy()
        tab[s] = dict(A=Y[:, :FC, :FC], B=Y[:, :FC, FC:2 * FC], sn=slot_nodes(d, s, nterm))
    return tab


def _push(d, s, ci, cur, Vn, known, tab):
    """One element: V_bus2 = B^-1 (I_bus1 - A@V_bus1). Returns True if it wrote."""
    T = tab.get(s)
    if T is None:
        return False
    sn = T["sn"]
    # Only the ACTIVE slots. A 3-phase line populates slots 0,1,2 and has no neutral
    # slot 3, so requiring all FC slots skipped nearly every line and the sweep silently
    # wrote nothing (error frozen at the flat-start 4.6%).
    act1 = [a for a in range(FC) if (ci, a) in sn]
    act2 = [a for a in range(FC) if (ci, FC + a) in sn]
    if not act1 or len(act1) != len(act2):
        return False
    n1 = [sn[(ci, a)] for a in act1]
    n2 = [sn[(ci, FC + a)] for a in act2]
    A = T["A"][ci][np.ix_(act1, act1)]
    B = T["B"][ci][np.ix_(act1, act2)]
    I1 = (cur[s][0][ci, act1].numpy() + 1j * cur[s][1][ci, act1].numpy())
    try:
        V2 = np.linalg.solve(B, I1 - A @ Vn[n1])
    except np.linalg.LinAlgError:
        return False
    wrote = False
    for j, a in enumerate(act2):
        if not known[n2[j]]:
            Vn[n2[j]] = V2[j]; wrote = True
    return wrote


def forward_sweep(d, ltree, cur, V, known, tab, layers=6):
    """V_child = B^-1 (I_bus1 - A@V_parent), level by level from the roots.

    Pure accumulation along the rooted tree: each level only reads voltages its parents
    already hold, so one pass propagates the slack boundary condition to every leaf.

    TREE_STORES is ("line","reactor","capacitor") -- transformers are NOT tree edges, so
    the tree is a FOREST rooted at the slack AND at every transformer secondary. Sweeping
    lines alone leaves everything below a transformer at its flat start (123Bus froze at
    4.7e-02 that way). Transformers are pushed between line passes, which feeds the next
    subtree's root; `layers` covers the sub->primary->secondary nesting depth.
    """
    sid = ltree["sid"].numpy(); comp = ltree["comp"].numpy()
    level = ltree["level"].numpy()
    Vn = V.copy()
    order = {}
    for e in range(level.size):
        s = _SID_INV[int(sid[e])]
        if s in SERIES_2T:
            order.setdefault(int(level[e]), set()).add((s, int(comp[e])))
    nx = store_size(d, "transformer") if "transformer" in d.node_types else 0
    for _ in range(layers):
        for lv in sorted(order):
            for s, ci in order[lv]:
                _push(d, s, ci, cur, Vn, known, tab)
        for ci in range(nx):
            _push(d, "transformer", ci, cur, Vn, known, tab)
    return Vn


def _unused(d, ltree, cur, V, known, tab):
    order = {}
    for lv in sorted(order):
        for s, ci in order[lv]:
            pass
    return V


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="SMART-DS_1000")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--sweeps", type=int, default=12)
    a = ap.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(TD, a.corpus, "*")) if os.path.isdir(p))
    step = max(1, len(paths) // a.n)
    paths = paths[::step][:a.n]

    print(f"corpus={a.corpus}  feeders={len(paths)}  sweeps={a.sweeps}")
    print(f"{'feeder':30s} {'nodes':>6s} " + "".join(f"{'sw' + str(k):>10s}" for k in (1, 3, 6, 12)))
    for p in paths:
        d = FeederScenarios(p)[0]
        n = node_count(d)
        vr = d["node"].V_r_pu.double(); vi = d["node"].V_i_pu.double()
        Vt = vr.numpy() + 1j * vi.numpy()
        Vi0 = (d["node"].V_r_init_pu.double().numpy()
               + 1j * d["node"].V_i_init_pu.double().numpy())
        known = np.zeros(n, dtype=bool)
        rel = ("vsource", "bus1", "node")
        if rel in d.edge_types and d[rel].edge_index.numel():
            known[d[rel].edge_index[1].numpy()] = True
        known[0] = True
        ctx = build_recon_ctx(d)
        tab = sweep_tables(d)
        present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]

        V = Vi0.copy(); V[known] = Vt[known]
        den = np.abs(Vt[~known]).sum() + 1e-30
        errs = {}
        for k in range(1, a.sweeps + 1):
            tvr = torch.tensor(V.real); tvi = torch.tensor(V.imag)
            cur = {}
            for s in present:
                if s in SHUNT_STORES or s in AMBIG_STORES:
                    cur[s] = tuple(x.clone() for x in element_currents(d, s, tvr, tvi))
                else:
                    z = torch.zeros_like(stored_currents(d, s, dtype=torch.float64)[0])
                    cur[s] = (z, z.clone())
            cur = reconstruct_full(d, cur, tvr, tvi, ctx=ctx)
            V = forward_sweep(d, ctx["ltree"], cur, V, known, tab)
            V[known] = Vt[known]
            if k in (1, 3, 6, 12):
                errs[k] = float(np.abs(V[~known] - Vt[~known]).sum() / den)
        print(f"{os.path.basename(p)[-28:]:30s} {n:6d} " +
              "".join(f"{errs.get(k, float('nan')):10.2e}" for k in (1, 3, 6, 12)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
