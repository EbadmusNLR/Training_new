"""Do the far ends of the bridges leaving node 49 sit in the SAME line-tree?

If yes, each extra bridge closes a loop through LINES ONLY -> the fix is to feed
bridges into mesh_correct (KVL with line Z), not to model transformers as loop
branches. If they are in different trees, the loop really does pass through a
transformer and needs its impedance.
"""
import glob, os, sys
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios
from gridfm.dk_tree import (_series_edges, _slack_xfmrsec_roots, _tree_from_edges,
                            TREE_STORES, AMBIG_STORES, classify_series)
TD = "/kfs2/projects/gogpt/Ebadmus/training_data/dss_data"
p = [x for x in sorted(glob.glob(os.path.join(TD, "*", "static.pt"))) if "IEEE 30" in x][0]
d = FeederScenarios(os.path.dirname(p))[0]
ser = {s: classify_series(d, s) for s in AMBIG_STORES}
E = [e for e in _series_edges(d, TREE_STORES)
     if e[0] not in AMBIG_STORES or e[1] in ser.get(e[0], set())]
slack, xsec = _slack_xfmrsec_roots(d)
tr = _tree_from_edges(E, slack | xsec)
comp = tr["comp_of"]
print("bridge far-end components (root id of the tree each endpoint belongs to):")
for i in tr["bridges"]:
    s, c, n1, n2, ca, cb = E[i]
    print("  %-5s c=%-3d n1=%-4d comp=%-5s | n2=%-4d comp=%-5s  %s"
          % (s, c, n1, comp.get(n1), n2, comp.get(n2),
             "SAME TREE -> line-only loop" if comp.get(n1) == comp.get(n2) else "different trees"))
ncomp = len(set(comp.values()))
print("\n#line-tree components: %d   #bridges: %d   -> loops among bridges = %d"
      % (ncomp, len(tr["bridges"]), len(tr["bridges"]) - (ncomp - 1) if ncomp else 0))
