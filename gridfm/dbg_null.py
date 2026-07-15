"""WHAT are the undetermined DOF made of?

The joint transformer system is 9 rows short on IEEE 30 Bus and pinv silently sets
those modes to zero. Before designing a closure I need to know what they live on:
  * only BRIDGE (line) conductors -> pure-line loops, mesh_correct's Z-KVL suffices;
  * TRANSFORMER conductors too     -> the circulating mode goes THROUGH a winding,
                                      the ratio rescales it, and the graph-cycle
                                      "loop current" abstraction does not apply.
Those need different fixes, so guessing is not an option.
"""
import glob, os, sys
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios
from gridfm.dk_tree import (build_xfmr_system, _series_edges, _tree_from_edges,
                            SERIES_STORES)

CORPUS = os.environ.get("CORPUS", "dss_data")
TGT = os.environ.get("TGT", "IEEE 30 Bus")
TD = f"/kfs2/projects/gogpt/Ebadmus/training_data/{CORPUS}"
p = [x for x in sorted(glob.glob(os.path.join(TD, "*", "static.pt")))
     if TGT in os.path.basename(os.path.dirname(x))][0]
fs = FeederScenarios(os.path.dirname(p))
d = fs[0]
# go under build_recon_ctx: it REFUSES this feeder, and the refusal is exactly what
# is being diagnosed.
slack = d["node"].slack.tolist() if hasattr(d["node"], "slack") else []
slack_set = {i for i, v in enumerate(slack) if v}
Eline = _series_edges(d, ("line",))
ltree = _tree_from_edges(Eline, slack_set)
bridges = [Eline[i] for i in ltree["bridges"]]
xmaps = build_xfmr_system(d, bridges=bridges, unsolved=[],
                          comp_of=ltree.get("comp_of"))
print(f"feeder {os.path.basename(os.path.dirname(p))}  groups={len(xmaps)}")
print(f"bridges={len(bridges)}  mchords={len(ltree.get('mchords', []))}")

for g in xmaps:
    keys, NB = g["keys"], g.get("null")
    print(f"\ngroup: {len(keys)} unknowns | rows: kcl={g['nkcl']} cut={g['ncut']} "
          f"bridge={g['nbridge']} dirs={len(g['dirs'])} | cond={g['cond']:.2e}")
    if NB is None:
        print("  fully determined")
        continue
    print(f"  NULL space: {NB.shape[1]} modes")
    # where does the null space live?
    w = np.abs(NB).sum(axis=1)
    by_store = defaultdict(float)
    for k, wi in zip(keys, w):
        by_store[k[0]] += float(wi)
    tot = sum(by_store.values()) + 1e-30
    for st, v in sorted(by_store.items(), key=lambda kv: -kv[1]):
        print(f"    {st:12s} {100*v/tot:6.2f}% of null weight")
    # per-mode: which conductors carry it
    for j in range(NB.shape[1]):
        col = np.abs(NB[:, j])
        idx = np.argsort(-col)[:6]
        parts = [f"{keys[i][0][:5]}[c{keys[i][1]},s{keys[i][2]}]={col[i]:.2f}"
                 for i in idx if col[i] > 1e-8]
        print(f"    mode {j}: " + "  ".join(parts))
