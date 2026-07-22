#!/usr/bin/env python3
"""Physics parity test: does I = Y@V - Icomp (my dk_physics decode from edges)
reproduce the STORED into-element currents at TRUTH V, per family?

The datakit contract (docs/DATA_STRUCTURE.md, validation/validation.py) is the
authority: full row-major Y, local V overlaid per terminal/conductor slot,
I_into = Y@V (passive) or Y@V - Icomp (active). validation.py passes on the
corpus, so the DATA is self-consistent; any mismatch here is a bug in the
model's edge-based reconstruction (dk_physics), which is what training uses.

Run via srun (never the login node). Dumps per-family WAPE at truth V, the KCL
residual of the stored currents (must be ~0 independent of my decode), and the
worst offenders with full structure so the bug is visible.
"""
from __future__ import annotations

import glob
import os
import sys

import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
from Datakit.core.scenario_store import FeederScenarios  # noqa: E402

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from gridfm.dk_physics import (  # noqa: E402
    STORES, FC, store_size, node_count, terminal_slot,
    element_currents, stored_currents, local_voltages, nodal_kcl_residual,
)

ROOT = "/kfs2/projects/gogpt/Ebadmus/training_data/SMART-DS_1000"


def truth_V(data):
    nd = data["node"]
    return nd.V_r_pu.double(), nd.V_i_pu.double()


def main():
    feeders = sorted(glob.glob(os.path.join(ROOT, "*", "static.pt")), key=os.path.getsize)
    probe = feeders[:4] + feeders[len(feeders) // 2: len(feeders) // 2 + 2]
    agg = {}
    kcl_max = 0.0
    worst = {}  # store -> (wape, feeder, row, detail)
    for p in probe:
        d = FeederScenarios(os.path.dirname(p))[0]
        vr, vi = truth_V(d)
        per_store = {}
        for s in STORES:
            if s not in d.node_types or store_size(d, s) == 0:
                continue
            Ir, Ii = element_currents(d, s, vr, vi)               # my decode
            Tr, Ti = stored_currents(d, s, dtype=torch.float64)   # stored truth
            per_store[s] = (Tr, Ti)
            num = (Ir - Tr).abs().sum(1) + (Ii - Ti).abs().sum(1)
            den = Tr.abs().sum(1) + Ti.abs().sum(1) + 1e-12
            rw = (num / den)                                       # per-row WAPE
            a = agg.setdefault(s, [0.0, 0.0])
            a[0] += float(num.sum()); a[1] += float(den.sum())
            j = int(rw.argmax())
            if rw[j] > worst.get(s, (0,))[0]:
                worst[s] = (float(rw[j]), os.path.basename(os.path.dirname(p)), j,
                            Ir[j], Ii[j], Tr[j], Ti[j])
        res = nodal_kcl_residual(d, per_store)
        kcl_max = max(kcl_max, float(res.abs().max()))
    print(f"probed {len(probe)} feeders (truth V, fp64)")
    for s, (num, den) in agg.items():
        print(f"  {s:12s} I=Y@V-Icomp vs stored  WAPE={num/den:.3e}")
    print(f"  stored-current KCL |residual| max = {kcl_max:.3e}")

    # dump the worst line + transformer structure to expose any assembly bug
    for s in ("line", "transformer", "vsource", "capacitor"):
        if s not in worst:
            continue
        w, feeder, row, ir, ii, tr, ti = worst[s]
        print(f"\n=== worst {s}: WAPE={w:.3e}  feeder={feeder} row={row} ===")
        print(f"  computed Ir={ir.tolist()}")
        print(f"  stored   Ir={tr.tolist()}")
        print(f"  computed Ii={ii.tolist()}")
        print(f"  stored   Ii={ti.tolist()}")

    # deep-dump the worst line's edge structure + assembled local V
    if "line" in worst:
        _, feeder, row, *_ = worst["line"]
        d = FeederScenarios(os.path.join(ROOT, feeder))[0]
        vr, vi = truth_V(d)
        _, nterm, _ = STORES["line"]
        print(f"\n--- worst-line structure: feeder={feeder} row={row} ---")
        for t in range(1, nterm + 1):
            rel = ("line", f"bus{t}", "node")
            if rel not in d.edge_types:
                print(f"  bus{t}: NO edge_type"); continue
            ei = d[rel].edge_index
            comp, node = ei[0], ei[1]
            m = comp == row
            slot = terminal_slot(comp)[m]
            print(f"  bus{t}: n_edges_for_row={int(m.sum())} nodes={node[m].tolist()} "
                  f"slots={slot.tolist()} V_r@nodes={vr[node[m]].tolist()}")
        Vlr, Vli = local_voltages(d, "line", nterm, vr, vi)
        print(f"  assembled Vlr[row]={Vlr[row].tolist()}")
        print(f"  assembled Vli[row]={Vli[row].tolist()}")
        st = d["line"]
        yr = st["Yline_r_pu"][row].reshape(8, 8)
        print(f"  Yline_r diag={torch.diagonal(yr).tolist()}")
        print(f"  Yline_r row0={yr[0].tolist()}")


def test_tree():
    """Reconstruct STORED line currents by subtree KCL from STORED injections."""
    from gridfm.dk_tree import reconstruct_series
    feeders = sorted(glob.glob(os.path.join(ROOT, "*", "static.pt")), key=os.path.getsize)
    probe = feeders[:4] + feeders[len(feeders) // 2: len(feeders) // 2 + 2]
    num = den = 0.0
    worst = (0.0, None)
    for p in probe:
        d = FeederScenarios(os.path.dirname(p))[0]
        if "line" not in d.node_types or store_size(d, "line") == 0:
            continue
        cur = {}
        for s in STORES:
            if s in d.node_types and store_size(d, s) > 0:
                cur[s] = stored_currents(d, s, dtype=torch.float64)
        Rr, Ri = reconstruct_series(d, cur)
        Tr, Ti = cur["line"]
        n = (Rr - Tr).abs().sum() + (Ri - Ti).abs().sum()
        e = Tr.abs().sum() + Ti.abs().sum() + 1e-12
        num += float(n); den += float(e)
        w = float(((Rr - Tr).abs().sum(1) + (Ri - Ti).abs().sum(1)).max())
        if w > worst[0]:
            worst = (w, os.path.basename(os.path.dirname(p)))
    print(f"\n=== TREE line reconstruction (stored injections -> stored line) ===")
    print(f"  line WAPE = {num/den:.3e}   worst-row abs = {worst[0]:.3e} ({worst[1]})")


if __name__ == "__main__":
    import sys as _s
    if "--tree" in _s.argv:
        test_tree()


def test_tree_unified():
    """Reconstruct line+transformer+vsource from SHUNT-ONLY injections (+ each
    series element's stored common-mode) via ONE unified subtree-KCL sweep."""
    from gridfm.dk_tree import reconstruct_unified, SERIES_STORES
    feeders = sorted(glob.glob(os.path.join(ROOT, "*", "static.pt")), key=os.path.getsize)
    probe = feeders[:4] + feeders[len(feeders) // 2: len(feeders) // 2 + 2]
    agg = {}
    for p in probe:
        d = FeederScenarios(os.path.dirname(p))[0]
        cur = {s: stored_currents(d, s, dtype=torch.float64)
               for s in STORES if s in d.node_types and store_size(d, s) > 0}
        rec = reconstruct_unified(d, cur)
        for s, (Rr, Ri) in rec.items():
            Tr, Ti = cur[s]
            a = agg.setdefault(s, [0.0, 0.0])
            a[0] += float((Rr - Tr).abs().sum() + (Ri - Ti).abs().sum())
            a[1] += float(Tr.abs().sum() + Ti.abs().sum() + 1e-12)
    print("\n=== UNIFIED tree reconstruction (shunt-only injections) ===")
    for s, (n, dd) in agg.items():
        print(f"  {s:12s} WAPE = {n/dd:.3e}")


if __name__ == "__main__":
    import sys as _s
    if "--unified" in _s.argv:
        test_tree_unified()


def test_all():
    """reconstruct_all: line+transformer+vsource from SHUNT-ONLY injections."""
    from gridfm.dk_tree import reconstruct_all
    feeders = sorted(glob.glob(os.path.join(ROOT, "*", "static.pt")), key=os.path.getsize)
    probe = feeders[:4] + feeders[len(feeders) // 2: len(feeders) // 2 + 2]
    agg = {}
    for p in probe:
        d = FeederScenarios(os.path.dirname(p))[0]
        cur = {s: stored_currents(d, s, dtype=torch.float64)
               for s in STORES if s in d.node_types and store_size(d, s) > 0}
        rec = reconstruct_all(d, cur)
        for s in ("line", "transformer", "vsource"):
            if s not in rec:
                continue
            Rr, Ri = rec[s]; Tr, Ti = cur[s]
            a = agg.setdefault(s, [0.0, 0.0])
            a[0] += float((Rr - Tr).abs().sum() + (Ri - Ti).abs().sum())
            a[1] += float(Tr.abs().sum() + Ti.abs().sum() + 1e-12)
    print("\n=== reconstruct_all (shunt-only injections) ===")
    for s, (n, dd) in agg.items():
        print(f"  {s:12s} WAPE = {n/dd:.3e}")


if __name__ == "__main__":
    import sys as _s
    if "--all" in _s.argv:
        test_all()


def test_vec():
    """Vectorized reconstruct (precomputed plan, s=0) vs stored, from shunt-only."""
    from gridfm.dk_tree import build_tree_plan, reconstruct_vectorized
    feeders = sorted(glob.glob(os.path.join(ROOT, "*", "static.pt")), key=os.path.getsize)
    probe = feeders[:4] + feeders[len(feeders) // 2: len(feeders) // 2 + 2]
    agg = {}
    for p in probe:
        d = FeederScenarios(os.path.dirname(p))[0]
        cur = {s: stored_currents(d, s, dtype=torch.float64)
               for s in STORES if s in d.node_types and store_size(d, s) > 0}
        plan = build_tree_plan(d)
        rec = reconstruct_vectorized(plan, cur)
        for s in ("line", "transformer", "vsource"):
            if s not in rec:
                continue
            Rr, Ri = rec[s]; Tr, Ti = cur[s]
            a = agg.setdefault(s, [0.0, 0.0])
            a[0] += float((Rr - Tr).abs().sum() + (Ri - Ti).abs().sum())
            a[1] += float(Tr.abs().sum() + Ti.abs().sum() + 1e-12)
    print("\n=== reconstruct_vectorized (shunt-only, s=0) ===")
    for s, (n, dd) in agg.items():
        print(f"  {s:12s} WAPE = {n/dd:.3e}")


if __name__ == "__main__":
    import sys as _s
    if "--vec" in _s.argv:
        test_vec()
