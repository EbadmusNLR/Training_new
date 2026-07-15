"""What is special about the ONE outlier feeder carrying 99.5% of the line error?
Check: mesh (chords), tree rooting/components, and which lines actually fail."""
import glob, os, sys
from collections import defaultdict, deque
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
from core.scenario_store import FeederScenarios
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from gridfm.dk_physics import STORES, FC, store_size, stored_currents, element_currents
from gridfm.dk_tree import (reconstruct_full, _series_edges, _tree_from_edges,
                            _slot_node_map, _slack_xfmrsec_roots, SHUNT_STORES)

ROOT = "/kfs2/projects/gogpt/Ebadmus/training_data/SMART-DS_1000"
BAD = [p for p in glob.glob(os.path.join(ROOT, "*", "static.pt")) if "p18uhs1_1247--p18udt2296" in p][0]
d = FeederScenarios(os.path.dirname(BAD))[0]
print("feeder:", os.path.basename(os.path.dirname(BAD)))

# 1. mesh check on the line-conductor graph
E = _series_edges(d, ("line",))
adj = defaultdict(list); nodes = set()
for i, (s, c, n1, n2, ca, cb) in enumerate(E):
    if n1 == 0 or n2 == 0: continue
    adj[n1].append(i); adj[n2].append(i); nodes.add(n1); nodes.add(n2)
ne = sum(1 for (s, c, n1, n2, ca, cb) in E if n1 != 0 and n2 != 0)
seen = set(); ncomp = 0
for u0 in nodes:
    if u0 in seen: continue
    ncomp += 1; seen.add(u0); dq = deque([u0])
    while dq:
        u = dq.popleft()
        for ei in adj[u]:
            s, c, n1, n2, ca, cb = E[ei]
            v = n2 if u == n1 else n1
            if v not in seen and v != 0: seen.add(v); dq.append(v)
print(f"line graph: edges={ne} nodes={len(nodes)} comps={ncomp} CHORDS={ne-(len(nodes)-ncomp)}")

# 2. are all components rooted at slack or xfmr-secondary?
slack, xsec = _slack_xfmrsec_roots(d)
roots = slack | xsec
unrooted = 0
seen = set()
for u0 in sorted(nodes):
    if u0 in seen: continue
    comp_nodes = set(); dq = deque([u0]); seen.add(u0); comp_nodes.add(u0)
    while dq:
        u = dq.popleft()
        for ei in adj[u]:
            s, c, n1, n2, ca, cb = E[ei]
            v = n2 if u == n1 else n1
            if v not in seen and v != 0:
                seen.add(v); comp_nodes.add(v); dq.append(v)
    if not (comp_nodes & roots): unrooted += 1
print(f"components with NO slack/xfmr-sec root: {unrooted} / {ncomp}")

# 3. which lines fail?
vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
cur = {}
for s in present:
    if s in SHUNT_STORES: cur[s] = element_currents(d, s, vr, vi)
    else:
        z = torch.zeros_like(truth[s][0]); cur[s] = (z, z.clone())
rec = reconstruct_full(d, cur, vr, vi)
Rr, Ri = rec["line"]; Tr, Ti = truth["line"]
err = (Rr-Tr).abs().sum(1) + (Ri-Ti).abs().sum(1)
se, idx = torch.sort(err, descending=True)
cum = torch.cumsum(se, 0) / (se.sum()+1e-30)
print(f"lines carrying 90% of this feeder's error: {int((cum<0.90).sum())+1} / {Tr.shape[0]}")
m1 = _slot_node_map(d, "line", 1); m2 = _slot_node_map(d, "line", 2)
for j in [int(x) for x in idx[:3]]:
    s1 = sorted(sl for (c, sl) in m1 if c == j); s2 = sorted(sl for (c, sl) in m2 if c == j)
    Ys_r = d["line"]["Ys_r_pu"][j].reshape(4,4); Ys_i = d["line"]["Ys_i_pu"][j].reshape(4,4)
    print(f"\n line row={j} err={float(err[j]):.3e}")
    print(f"   bus1 slots={s1} nodes={[m1[(j,s)] for s in s1]}")
    print(f"   bus2 slots={s2} nodes={[m2[(j,s)] for s in s2]}")
    print(f"   |Ys|diag={[round(float(Ys_r[k,k].abs()+Ys_i[k,k].abs()),3) for k in range(4)]}")
    print(f"   stored Ir={[round(float(x),6) for x in Tr[j]]}")
    print(f"   recon  Ir={[round(float(x),6) for x in Rr[j]]}")
# 4. transformer / vsource health on this feeder
for s in ("transformer", "vsource"):
    if s in rec:
        R, T = rec[s], truth[s]
        w = float(((R[0]-T[0]).abs().sum()+(R[1]-T[1]).abs().sum())/(T[0].abs().sum()+T[1].abs().sum()+1e-30))
        print(f"\n {s} WAPE on this feeder = {w:.3e}")
