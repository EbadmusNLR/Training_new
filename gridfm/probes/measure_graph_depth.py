"""Why is a 7092-node feeder only 4 hops deep, and what depth does the PHYSICS need?

measure_receptive_field.py found max BFS depth 4 from the slack -- so DKSolver's 12 steps
already reach every node and the v_skill plateau is NOT a receptive-field limit.

Suspicion: node 0 (ground) is a HUB. Every grounded neutral, capacitor, transformer and
load neutral touches it, so slack -> ground -> any grounded node is ~2 hops and the graph
looks small-world. Reachability is real, but every long-range message is then forced
through ONE node's hidden vector -- oversquashing, not distance. A global attention node
cannot fix that: ground already IS one.

Reports three depths per feeder:
  with-ground    : what message passing traverses today
  without-ground : the electrical distance ignoring the ground reference
  tree-level     : depth of dk_tree's rooted series tree -- the number of sequential
                   sweeps a backward/forward physics solve needs, which is the number
                   that actually governs a convergent iterative scheme
"""
import collections
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")

from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, store_size, node_count
from gridfm.dk_tree import build_recon_ctx

CORPUS = os.environ.get("CORPUS", "SMART-DS_1000")
NF = int(os.environ.get("NF", "6"))


def bfs_depth(d, n, skip_ground):
    adj = collections.defaultdict(set)
    slack = set()
    for s in STORES:
        if s not in d.node_types or store_size(d, s) == 0:
            continue
        _, nterm, _ = STORES[s]
        per_comp = collections.defaultdict(list)
        for t in range(1, nterm + 1):
            rel = (s, f"bus{t}", "node")
            if rel not in d.edge_types or not d[rel].edge_index.numel():
                continue
            ei = d[rel].edge_index
            for c, nd in zip(ei[0].tolist(), ei[1].tolist()):
                if skip_ground and int(nd) == 0:
                    continue
                per_comp[int(c)].append(int(nd))
            if s == "vsource" and t == 1:
                slack.update(int(x) for x in ei[1].tolist())
        for c, nodes in per_comp.items():
            for a in nodes:
                for b in nodes:
                    if a != b:
                        adj[a].add(b)
    depth = np.full(n, -1, dtype=np.int64)
    q = collections.deque()
    for sN in slack:
        if 0 <= sN < n:
            depth[sN] = 0; q.append(sN)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1; q.append(v)
    return depth


def ground_degree(d):
    """How many components touch node 0? A hub of this size is the oversquashing point."""
    deg = 0
    for s in STORES:
        if s not in d.node_types or store_size(d, s) == 0:
            continue
        _, nterm, _ = STORES[s]
        for t in range(1, nterm + 1):
            rel = (s, f"bus{t}", "node")
            if rel in d.edge_types and d[rel].edge_index.numel():
                deg += int((d[rel].edge_index[1] == 0).sum())
    return deg


td = f"/kfs2/projects/gogpt/Ebadmus/training_data/{CORPUS}"
paths = sorted(p for p in glob.glob(os.path.join(td, "*")) if os.path.isdir(p))
step = max(1, len(paths) // NF)
paths = paths[::step][:NF]

print(f"corpus={CORPUS}  feeders={len(paths)}\n")
print(f"{'feeder':30s} {'nodes':>6s} {'gnd_deg':>8s} {'with_gnd':>9s} {'no_gnd':>8s} "
      f"{'no_gnd_med':>11s} {'tree_lvls':>10s} {'unreach_nogd':>13s}")
for p in paths:
    d = FeederScenarios(p)[0]
    n = node_count(d)
    gd = ground_degree(d)
    dw = bfs_depth(d, n, skip_ground=False)
    dn = bfs_depth(d, n, skip_ground=True)
    ctx = build_recon_ctx(d)
    lvl = ctx["ltree"].get("level")
    tl = int(max(lvl)) + 1 if lvl is not None and len(lvl) else -1
    rw = dw[dw >= 0]; rn = dn[dn >= 0]
    print(f"{os.path.basename(p)[-28:]:30s} {n:6d} {gd:8d} {int(rw.max()):9d} "
          f"{int(rn.max()) if len(rn) else -1:8d} {int(np.median(rn)) if len(rn) else -1:11d} "
          f"{tl:10d} {int((dn < 0).sum()):13d}")

print("\nwith_gnd  = what message passing traverses today (ground is a hub)")
print("no_gnd    = electrical distance ignoring the ground reference")
print("tree_lvls = sequential sweeps a backward/forward physics solve needs")
