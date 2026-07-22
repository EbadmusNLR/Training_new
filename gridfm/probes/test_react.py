"""Are reactors SHUNT (bus2=ground -> physics-decode) or SERIES (both live -> tree)?
Resolve with data instead of guessing. Also check generators (SHUNT_STORES)."""
import glob, os, sys
from collections import Counter
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
from Datakit.core.scenario_store import FeederScenarios
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from gridfm.dk_physics import STORES, FC, store_size, stored_currents, element_currents
from gridfm.dk_tree import _slot_node_map

TD = "/kfs2/projects/gogpt/Ebadmus/training_data/minimal_component"
feeders = sorted(glob.glob(os.path.join(TD, "*", "static.pt")))[:40]
kind = Counter(); ycheck = [0.0, 0.0]; gen = [0.0, 0.0]
for p in feeders:
    d = FeederScenarios(os.path.dirname(p))[0]
    if "reactor" in d.node_types and store_size(d, "reactor") > 0:
        m1 = _slot_node_map(d, "reactor", 1); m2 = _slot_node_map(d, "reactor", 2)
        for (c, sl), n1 in m1.items():
            n2 = m2.get((c, sl))
            if n2 is None:   kind["bus2_slot_ABSENT"] += 1
            elif n2 == 0:    kind["bus2=GROUND (shunt-like)"] += 1
            elif n1 == 0:    kind["bus1=GROUND"] += 1
            else:            kind["BOTH LIVE (series-like)"] += 1
        # does I = Y@V hold for reactors (i.e. is physics-decode valid)?
        vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
        Ir, Ii = element_currents(d, "reactor", vr, vi)
        Tr, Ti = stored_currents(d, "reactor", dtype=torch.float64)
        ycheck[0] += float((Ir-Tr).abs().sum()+(Ii-Ti).abs().sum())
        ycheck[1] += float(Tr.abs().sum()+Ti.abs().sum())
    if "generator" in d.node_types and store_size(d, "generator") > 0:
        vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
        Ir, Ii = element_currents(d, "generator", vr, vi)
        Tr, Ti = stored_currents(d, "generator", dtype=torch.float64)
        gen[0] += float((Ir-Tr).abs().sum()+(Ii-Ti).abs().sum())
        gen[1] += float(Tr.abs().sum()+Ti.abs().sum())
print("reactor terminal structure:")
for k, v in kind.most_common():
    print(f"   {k:28s} {v}")
print(f"\nreactor  I=Y@V-Icomp vs stored WAPE = {ycheck[0]/(ycheck[1]+1e-30):.3e}   |I|={ycheck[1]:.3e}")
print(f"generator I=Y@V-Icomp vs stored WAPE = {gen[0]/(gen[1]+1e-30):.3e}   |I|={gen[1]:.3e}")
