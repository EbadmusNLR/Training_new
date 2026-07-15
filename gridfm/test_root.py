"""Are the unwritten line conductors GENUINE loops or a graph artifact?
For the line-conductor graph: chords = n_edges - (n_nodes_touched - n_components).
A truly radial feeder has chords = 0. Also report the |I| carried by chords."""
import glob, os, sys
from collections import defaultdict, deque
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
from core.scenario_store import FeederScenarios
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from gridfm.dk_physics import store_size, stored_currents, FC
from gridfm.dk_tree import _series_edges, _tree_from_edges, _slack_xfmrsec_roots

ROOT = "/kfs2/projects/gogpt/Ebadmus/training_data/SMART-DS_1000"
feeders = sorted(glob.glob(os.path.join(ROOT, "*", "static.pt")), key=os.path.getsize)
probe = feeders[:3] + feeders[len(feeders)//2: len(feeders)//2+3]

for p in probe:
    d = FeederScenarios(os.path.dirname(p))[0]
    if "line" not in d.node_types or store_size(d, "line") == 0:
        continue
    E = _series_edges(d, ("line",))
    # graph over non-ground endpoints
    adj = defaultdict(list); nodes = set()
    for i, (s, c, n1, n2, ca, cb) in enumerate(E):
        if n1 == 0 or n2 == 0:
            continue
        adj[n1].append(i); adj[n2].append(i); nodes.add(n1); nodes.add(n2)
    n_edges = sum(1 for (s, c, n1, n2, ca, cb) in E if n1 != 0 and n2 != 0)
    # count connected components
    seen = set(); ncomp = 0
    for u0 in nodes:
        if u0 in seen: continue
        ncomp += 1; seen.add(u0); dq = deque([u0])
        while dq:
            u = dq.popleft()
            for ei in adj[u]:
                s, c, n1, n2, ca, cb = E[ei]
                v = n2 if u == n1 else n1
                if v not in seen and v != 0:
                    seen.add(v); dq.append(v)
    chords = n_edges - (len(nodes) - ncomp)
    # |I| carried by chord conductors (those NOT tree edges)
    slack, xsec = _slack_xfmrsec_roots(d)
    tree = _tree_from_edges(E, slack | xsec)
    written = {(int(c), int(ca)) for c, ca in zip(tree["comp"].tolist(), tree["cola"].tolist())}
    Tr, Ti = stored_currents(d, "line", dtype=torch.float64)
    tot = float(Tr.abs().sum() + Ti.abs().sum())
    chord_I = 0.0
    for (s, c, n1, n2, ca, cb) in E:
        if (int(c), int(ca)) not in written:
            chord_I += float(Tr[c, ca].abs() + Ti[c, ca].abs() + Tr[c, cb].abs() + Ti[c, cb].abs())
    name = os.path.basename(os.path.dirname(p))[:28]
    print(f"{name:30s} edges={n_edges:5d} nodes={len(nodes):5d} comps={ncomp:3d} "
          f"CHORDS={chords:4d}  chord|I|/tot={chord_I/(tot+1e-12):.3f}")
