"""WHAT are the undetermined DOF in IEEE 30 Bus's transformer group (rank 47 < 56)?

Assemble the same rows build_xfmr_system does, take the NULL SPACE of that row
matrix, and read off which conductors it lives on. That names the missing physics
exactly instead of theorising about "loops through transformers".
"""
import glob, os, sys
from collections import Counter, defaultdict
import numpy as np
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios
from gridfm.dk_physics import FC, terminal_slot
from gridfm.dk_tree import (_series_edges, _slack_xfmrsec_roots, _tree_from_edges,
                            TREE_STORES, AMBIG_STORES, classify_series)

TD = os.path.join("/kfs2/projects/gogpt/Ebadmus/training_data",
                  os.environ.get("CORPUS", "dss_data"))
TGT = os.environ.get("TGT", "IEEE 30 Bus")
p = [x for x in sorted(glob.glob(os.path.join(TD, "*", "static.pt")))
     if TGT in os.path.basename(os.path.dirname(x))][0]
d = FeederScenarios(os.path.dirname(p))[0]
print("feeder:", os.path.basename(os.path.dirname(p)))

ser = {s: classify_series(d, s) for s in AMBIG_STORES}
E = [e for e in _series_edges(d, TREE_STORES)
     if e[0] not in AMBIG_STORES or e[1] in ser.get(e[0], set())]
slack, xsec = _slack_xfmrsec_roots(d)
tr = _tree_from_edges(E, slack | xsec)
bridges = [E[i] for i in tr["bridges"]]
print("slack=%s |xsec|=%d tree_edges=%d chords=%d bridges=%d"
      % (sorted(slack), len(xsec), len(tr["sid"]), len(tr["chords"]), len(bridges)))

st = d["transformer"]
Yr = st["Yxfmr_r_pu"].reshape(-1, 3*FC, 3*FC).double().numpy()
Yi = st["Yxfmr_i_pu"].reshape(-1, 3*FC, 3*FC).double().numpy()
slot_node = {}
for t in (1, 2, 3):
    rel = ("transformer", "bus%d" % t, "node")
    if rel in d.edge_types and d[rel].edge_index.numel():
        ei = d[rel].edge_index
        k = terminal_slot(ei[0])
        for c, kk, nd in zip(ei[0].tolist(), k.tolist(), ei[1].tolist()):
            slot_node[(int(c), (t-1)*FC + int(kk))] = int(nd)
act_of = {}
for row in range(Yr.shape[0]):
    diag = np.abs(np.diag(Yr[row] + 1j*Yi[row]))
    if diag.max() > 0:
        act_of[row] = [int(i) for i in np.where(diag > 1e-9*diag.max())[0]]

node_conds = defaultdict(list)
for r, a in act_of.items():
    for s in a:
        nd = slot_node.get((r, s), 0)
        if nd != 0:
            node_conds[nd].append(("transformer", r, s))
bpairs = []
for (bs, bc, bn1, bn2, bca, bcb) in bridges:
    ka, kb = (bs, bc, bca), (bs, bc, bcb)
    node_conds[bn1].append(ka); node_conds[bn2].append(kb)
    bpairs.append((ka, kb, bn1, bn2))
knodes = sorted(nd for nd in node_conds if nd in xsec)

keys = [("transformer", r, s) for r, a in act_of.items() for s in a]
for pr in bpairs:
    keys.append(pr[0]); keys.append(pr[1])
xi = {k: i for i, k in enumerate(keys)}
Nx = len(xi)
R = []
for nd in knodes:
    row = np.zeros(Nx, dtype=np.complex128)
    for k in node_conds[nd]:
        row[xi[k]] = 1.0
    R.append(row)
n_kcl = len(R)
for ka, kb, n1, n2 in bpairs:
    row = np.zeros(Nx, dtype=np.complex128)
    row[xi[ka]] = 1.0; row[xi[kb]] = 1.0
    R.append(row)
n_br = len(R) - n_kcl
for r in sorted(act_of):
    act = act_of[r]
    Y = Yr[r].astype(np.complex128) + 1j*Yi[r]
    Ya = Y[np.ix_(act, act)]
    _, S, Vh = np.linalg.svd(Ya)
    for j in range(len(S)):
        nv = Vh[j].conj()
        row = np.zeros(Nx, dtype=np.complex128)
        for k2, s in enumerate(act):
            row[xi[("transformer", r, s)]] = nv[k2]
        R.append(row)
R = np.array(R)
rank = np.linalg.matrix_rank(R, tol=1e-6)
print("\nunknowns Nx=%d rows=%d rank=%d -> SHORT BY %d" % (Nx, len(R), rank, Nx-rank))
print("  rows: KCL=%d bridge=%d directions=%d" % (n_kcl, n_br, len(R)-n_kcl-n_br))
print("  unknown inventory: %s" % Counter(k[0] for k in keys))
print("  bridge endpoints:")
for ka, kb, n1, n2 in bpairs[:16]:
    print("     line%-3d n1=%-4d xsec=%-5s slack=%-5s | n2=%-4d xsec=%-5s slack=%s"
          % (ka[1], n1, n1 in xsec, n1 in slack, n2, n2 in xsec, n2 in slack))
_, S2, Vh2 = np.linalg.svd(R)
ns = Vh2[rank:].conj()
print("\n  null space dim %d -- UNDETERMINED directions live on:" % ns.shape[0])
for i, vec in enumerate(ns[:12]):
    w = np.abs(vec)
    idx = np.argsort(-w)[:6]
    parts = ["%s%d.s%d(%.2f)" % (keys[j][0][:4], keys[j][1], keys[j][2], w[j])
             for j in idx if w[j] > 1e-6]
    print("    null[%d]: %s" % (i, "  ".join(parts)))

# --- do CUT-SET rows (sum of KCL over a whole rooted component) add anything?
comp_of = tr["comp_of"]
by_root = defaultdict(list)
for nd, root in comp_of.items():
    if nd != 0:
        by_root[root].append(nd)
Rc = list(R)
added = 0
for root, nodes in sorted(by_root.items()):
    ns = set(nodes)
    if ns & slack:
        continue
    ks = sorted({k for nd in nodes for k in node_conds.get(nd, ())})
    if not ks:
        continue
    row = np.zeros(Nx, dtype=np.complex128)
    for k in ks:
        row[xi[k]] = 1.0
    Rc.append(row); added += 1
Rc = np.array(Rc)
rank_c = np.linalg.matrix_rank(Rc, tol=1e-6)
print("\nCUT-SET rows added: %d  -> rank %d (was %d)  still short by %d"
      % (added, rank_c, rank, Nx - rank_c))
# how many components actually carry >1 unknown (i.e. could add info)?
multi = 0
for root, nodes in by_root.items():
    ks = {k for nd in nodes for k in node_conds.get(nd, ())}
    if len(ks) > 1:
        multi += 1
print("  components carrying >1 unknown: %d" % multi)
print("  NOTE: a component whose nodes are all xsec gives a cut-set = sum of KCL rows"
      "\n        we ALREADY have -> linearly dependent -> adds 0 rank.")
