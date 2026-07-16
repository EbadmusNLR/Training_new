"""Inventory + assumption audit across ALL corpora (SMART-DS, minimal_component,
dss_data). The new corpora should contain exactly the structures flagged as
UNTESTED: reactors, generators, series capacitors, multi-vsource, ...
"""
import glob, os, sys
from collections import Counter
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
from core.scenario_store import FeederScenarios
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from gridfm.dk_physics import STORES, FC, store_size
from gridfm.dk_tree import _slot_node_map, check_assumptions

TD = "/kfs2/projects/gogpt/Ebadmus/training_data"


def probe(corpus, limit=None):
    feeders = sorted(glob.glob(os.path.join(TD, corpus, "*", "static.pt")))
    if limit:
        feeders = feeders[:limit]
    inv = Counter(); nf = Counter(); viol = Counter(); nvar = 0
    ser_cap = 0; multi_vs = 0; vs_live2 = 0
    for p in feeders:
        try:
            fs = FeederScenarios(os.path.dirname(p))
            nvar += len(fs)
            d = fs[0]
        except Exception as e:
            viol[f"LOAD_FAIL:{type(e).__name__}"] += 1; continue
        for s in d.node_types:
            if s == "node": continue
            n = store_size(d, s)
            if n > 0:
                inv[s] += n; nf[s] += 1
        # structural checks
        if "capacitor" in d.node_types and store_size(d, "capacitor") > 0:
            m1 = _slot_node_map(d, "capacitor", 1); m2 = _slot_node_map(d, "capacitor", 2)
            for (c, sl), n1 in m1.items():
                if n1 != 0 and m2.get((c, sl), 0) != 0:
                    ser_cap += 1
        if "vsource" in d.node_types:
            if store_size(d, "vsource") > 1: multi_vs += 1
            m2 = _slot_node_map(d, "vsource", 2)
            vs_live2 += sum(1 for n in m2.values() if n != 0)
        for v in check_assumptions(d, raise_on_fail=False):
            viol[v.split(":")[0][:52]] += 1
    return feeders, nvar, inv, nf, viol, ser_cap, multi_vs, vs_live2


for corpus in ("SMART-DS_1000", "minimal_component", "dss_data"):
    if not os.path.isdir(os.path.join(TD, corpus)):
        print(f"\n### {corpus}: MISSING"); continue
    feeders, nvar, inv, nf, viol, sc, mv, vl = probe(corpus)
    print(f"\n### {corpus}: {len(feeders)} feeders, {nvar} variants")
    print("  inventory (elements / feeders-containing):")
    for s in sorted(inv):
        print(f"    {s:12s} {inv[s]:8d} / {nf[s]:5d}")
    print(f"  series-capacitor conductors : {sc}")
    print(f"  feeders w/ multiple vsources: {mv}")
    print(f"  vsource bus2 live (non-gnd) : {vl}")
    if viol:
        print("  GUARD FIRES:")
        for k, v in viol.most_common():
            print(f"    [{v:5d} feeders] {k}")
    else:
        print("  guard: no violations")
