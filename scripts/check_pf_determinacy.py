#!/usr/bin/env python3
"""PF determinacy gate: does physics actually DETERMINE the pf task's targets?

The pf task shows the model Y, Icomp and the slack/ground voltages, and asks for every
other voltage. That is only learnable if those inputs determine those targets. KCL over
every element's own admittance gives

    Ybus . V = sum(Icomp)          (Ibus = Y.V for passive, Ibus + Icomp = Y.V for active)

so with slack/ground voltages pinned the rest is determined iff that system is
nonsingular. Where it is not -- floating conductor chains, ungrounded secondary islands --
the "truth" voltage is arbitrary OpenDSS output and NO model can learn it. Training PF on
such a corpus produces an architecture mystery with a data cause.

This is not hypothetical: the previous effort measured ~16% determinacy before export
fixes and ~96-98% after, and its real SMART-DS corpus sat at effectively 0% with the
grounding/neutral export named as the blocker (see historical memory notes). SMART-DS is 95%
of this corpus by node count, so its determinacy decides whether PF training is viable at
all.

Reports per feeder: the residual of the pinned system at TRUTH voltages (is the truth even
consistent with the assembled physics?) and the recovered-voltage error from solving it.

    python scripts/check_pf_determinacy.py --corpus SMART-DS_1000 --feeders 8
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")

from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, FC, store_size, node_count, terminal_slot, _y_full

ROOT = "/kfs2/projects/gogpt/Ebadmus"


def build_ybus(d, n):
    """Ybus and the Icomp rhs, assembled from every element's own Y (dense fp64 complex).

    Dense costs n^2*16 bytes -- ~3 GB at the corpus's largest feeder (13738 nodes) -- which
    is affordable on a compute node and avoids a scipy dependency the venv does not have.
    """
    Ybus = np.zeros((n, n), dtype=np.complex128)
    rhs = np.zeros(n, dtype=np.complex128)
    for s in STORES:
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
        for (c, a), na in sn.items():
            if ic is not None:
                rhs[na] += ic[c, a]
            for b in range(dim):
                nb = sn.get((c, b))
                if nb is not None:
                    Ybus[na, nb] += Y[c, a, b]
    return Ybus, rhs


def known_mask(d, n):
    """slack (vsource bus1) + ground(node 0) -- exactly what mask_pf leaves visible."""
    m = np.zeros(n, dtype=bool)
    rel = ("vsource", "bus1", "node")
    if "vsource" in d.node_types and rel in d.edge_types and d[rel].edge_index.numel():
        m[d[rel].edge_index[1].numpy()] = True
    m[0] = True
    return m


def check(fdir, variant, tol):
    d = FeederScenarios(fdir)[variant]
    n = node_count(d)
    Ybus, rhs = build_ybus(d, n)
    Vtrue = d["node"].V_r_pu.double().numpy() + 1j * d["node"].V_i_pu.double().numpy()
    known = known_mask(d, n)

    # Is TRUTH consistent with the assembled physics at all? If this residual is large the
    # export itself is inconsistent, and no solve/model result downstream means anything.
    res = np.abs(Ybus @ Vtrue - rhs)
    res_free = float(res[~known].max()) if (~known).any() else 0.0

    # Pin the visible voltages, solve for the rest.
    A = Ybus.copy()
    b = rhs.copy()
    idx = np.where(known)[0]
    A[idx, :] = 0.0
    A[idx, idx] = 1.0
    b[idx] = Vtrue[idx]
    # Rank, not just a solve: a singular Ybus is exactly the floating-island case, and
    # numpy may return garbage rather than raise.
    try:
        V = np.linalg.solve(A, b)
        singular = not np.all(np.isfinite(V))
    except np.linalg.LinAlgError:
        V = np.full(n, np.nan, dtype=np.complex128)
        singular = True

    if singular:
        return dict(n=n, res_free=res_free, singular=True, frac=0.0, vmax=float("inf"))
    err = np.abs(V - Vtrue)
    free = ~known
    frac = float((err[free] < tol).mean()) if free.any() else 1.0
    return dict(n=n, res_free=res_free, singular=False, frac=frac,
                vmax=float(err[free].max()) if free.any() else 0.0,
                cond=float(np.linalg.cond(A)) if n <= 2000 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="SMART-DS_1000")
    ap.add_argument("--feeders", type=int, default=8)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    td = f"{ROOT}/training_data/{args.corpus}"
    paths = sorted(p for p in glob.glob(os.path.join(td, "*")) if os.path.isdir(p))
    step = max(1, len(paths) // args.feeders)
    paths = paths[::step][:args.feeders]

    print(f"corpus={args.corpus}  feeders={len(paths)}  variants={args.variants}  tol={args.tol}\n")
    print(f"{'feeder':40s} {'nodes':>6s} {'|Ybus.Vtrue-Icomp|':>19s} {'solved<tol':>11s} {'max|V-Vtrue|':>13s}")
    fracs = []
    for p in paths:
        for v in range(args.variants):
            r = check(p, v, args.tol)
            tag = "SINGULAR" if r["singular"] else f"{100*r['frac']:9.2f}%"
            fracs.append(0.0 if r["singular"] else r["frac"])
            print(f"{os.path.basename(p)[-38:]:40s} {r['n']:6d} {r['res_free']:19.3e} "
                  f"{tag:>11s} {r['vmax']:13.3e}  cond={r.get('cond', float('nan')):.2e}")
    if fracs:
        print(f"\nmean determinacy: {100*float(np.mean(fracs)):.2f}%  "
              f"(healthy corpus ~96-98%; PF is unlearnable where this is low)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
