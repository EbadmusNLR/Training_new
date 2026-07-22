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
from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_tree import (build_xfmr_system, _series_edges, _tree_from_edges,
                            _slack_xfmrsec_roots, classify_series, build_kvl_rows,
                            TREE_STORES, AMBIG_STORES)

CORPUS = os.environ.get("CORPUS", "dss_data")
TGT = os.environ.get("TGT", "IEEE 30 Bus")
TD = f"/kfs2/projects/gogpt/Ebadmus/training_data/{CORPUS}"
p = [x for x in sorted(glob.glob(os.path.join(TD, "*", "static.pt")))
     if TGT in os.path.basename(os.path.dirname(x))][0]
fs = FeederScenarios(os.path.dirname(p))
d = fs[0]
# Go under build_recon_ctx (it REFUSES this feeder, which is what is being
# diagnosed) -- but replicate its topology EXACTLY. Rooting at the slack alone
# instead of slack|xfmr-secondaries gives a different tree: 0 bridges, 56 unknowns,
# "fully determined". A probe that builds its own topology measures its own bug.
slack, xsec = _slack_xfmrsec_roots(d)
ser = {s: classify_series(d, s) for s in AMBIG_STORES}
Eline = [e for e in _series_edges(d, TREE_STORES)
         if e[0] not in AMBIG_STORES or e[1] in ser.get(e[0], set())]
ltree = _tree_from_edges(Eline, slack | xsec)
bridges = [Eline[i] for i in ltree["bridges"]]
kr = build_kvl_rows(d, Eline, ltree)
kvl = (kr[0], kr[1], kr[2], Eline) if kr else None
# comp_of=None: cut-set rows are RETRACTED (they regressed real feeders). Must match
# production or the probe measures a decoder that no longer exists.
xmaps = build_xfmr_system(d, bridges=bridges, unsolved=[], comp_of=None, kvl=kvl)
print(f"feeder {os.path.basename(os.path.dirname(p))}  groups={len(xmaps)}")
print(f"bridges={len(bridges)}  mchords={len(ltree.get('mchords', []))}")
mte = set(ltree["mparent_edge"].values())
bch = [i for i in ltree["bridges"] if i not in mte]
print(f"bridges that are mesh CHORDS (candidate loops): {len(bch)}   "
      f"mesh-TREE bridges (close no loop): {len(ltree['bridges']) - len(bch)}")
if kr:
    print(f"build_kvl_rows -> {kr[1].shape[0]} candidate loop rows")
for c in bch:
    s, cc, n1, n2, ca, cb = Eline[c]
    print(f"    chord bridge: {s}[c{cc}] n{n1}->n{n2} slots {ca}/{cb} "
          f"comp {ltree['comp_of'].get(n1)}->{ltree['comp_of'].get(n2)}")

for g in xmaps:
    keys, NB = g["keys"], g.get("null")
    print(f"\ngroup: {len(keys)} unknowns | rows: kcl={g['nkcl']} cut={g['ncut']} "
          f"bridge={g['nbridge']} kvl={g.get('nkvl')} dirs={len(g['dirs'])} | "
          f"cond={g['cond']:.2e}")
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
