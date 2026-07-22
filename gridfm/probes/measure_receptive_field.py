"""Is the v_skill plateau a RECEPTIVE FIELD limit rather than a conditioning limit?

PF is 100% determinate on this corpus and cond(Ybus) measures ~4e9 (not the 1.25e18 my
notes blamed the plateau on). So the data determines V exactly. What might not is the
MODEL: DKSolver does `steps` rounds of message passing, so a node learns nothing about
anything more than `steps` hops away. The slack voltage is the boundary condition that
makes PF determinate -- if a node sits 300 hops from the slack and the model runs 12
steps, that node cannot see the slack AT ALL, and its voltage is unlearnable no matter how
long we train or how wide the network is.

Measures, per feeder, the BFS hop distance from the slack over the electrical node graph
(the same adjacency DKFeeder._pe builds), and reports what fraction of nodes fall inside
the model's receptive field at several step counts.

If most nodes are outside it, the fix is propagation depth (steps, or a tree sweep that
moves information root-to-leaf in ONE pass), not width, not training time.
"""
import collections
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")

from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, FC, store_size, node_count, terminal_slot

CORPUS = os.environ.get("CORPUS", "SMART-DS_1000")
NF = int(os.environ.get("NF", "8"))
STEPS = [4, 8, 12, 24, 48, 96]


def hop_depth(d, n):
    """BFS depth from the slack over the node graph a message-passing step can traverse:
    each component links every node on its terminals to every other (one MP step crosses
    one component)."""
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
                per_comp[int(c)].append(int(nd))
            if s == "vsource" and t == 1:
                slack.update(ei[1].tolist())
        for c, nodes in per_comp.items():
            for a in nodes:
                for b in nodes:
                    if a != b:
                        adj[a].add(b)
    depth = np.full(n, -1, dtype=np.int64)
    q = collections.deque()
    for sN in slack:
        depth[sN] = 0
        q.append(sN)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1
                q.append(v)
    return depth, slack


td = f"/kfs2/projects/gogpt/Ebadmus/training_data/{CORPUS}"
paths = sorted(p for p in glob.glob(os.path.join(td, "*")) if os.path.isdir(p))
step = max(1, len(paths) // NF)
paths = paths[::step][:NF]

print(f"corpus={CORPUS}  feeders={len(paths)}\n")
print(f"{'feeder':34s} {'nodes':>6s} {'maxdepth':>8s} {'median':>7s} " +
      "".join(f"{'<=' + str(s):>7s}" for s in STEPS) + f"{'unreach':>8s}")
allf = {s: [] for s in STEPS}
for p in paths:
    d = FeederScenarios(p)[0]
    n = node_count(d)
    depth, slack = hop_depth(d, n)
    reach = depth >= 0
    unreach = int((~reach).sum())
    dd = depth[reach]
    row = ""
    for s in STEPS:
        f = float((dd <= s).mean())
        allf[s].append(f)
        row += f"{100*f:6.1f}%"
    print(f"{os.path.basename(p)[-32:]:34s} {n:6d} {int(dd.max()):8d} {int(np.median(dd)):7d} "
          f"{row}{unreach:8d}")

print(f"\nfraction of nodes INSIDE the model's receptive field, mean over feeders:")
for s in STEPS:
    print(f"  steps={s:3d}  {100*float(np.mean(allf[s])):6.2f}%")
print("\nDKSolver default is steps=12. A node outside the receptive field cannot see the\n"
      "slack boundary condition that makes PF determinate -- its voltage is unlearnable.")
