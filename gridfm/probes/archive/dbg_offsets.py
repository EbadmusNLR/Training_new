"""Do MY merged offsets agree with PyG's actual batched indices?

reconstruct_full mixes two sources of truth on a batched graph:
  * build_q  uses ctx["inj"]      -- MY merged (comp+soff, col, node+node_off)
  * _full_residual uses _inj_index(data, s) -- recomputed from PyG's batched edge_index
If those disagree for ANY store, currents get scattered to the wrong rows/nodes and
feeders corrupt each other. This compares them element-wise.
"""
import os, sys
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from torch_geometric.data import Batch
from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, store_size, node_count, ensure_batch_schema
from gridfm.dk_tree import (build_recon_ctx, batch_recon_ctx, _inj_index,
                            SHUNT_STORES, SERIES_STORES)

keys = tuple(SHUNT_STORES) + tuple(SERIES_STORES)
datas, ctxs, ncs, scs = [], [], [], []
for fdir in sys.argv[1:]:
    d = FeederScenarios(fdir)[0]
    d["node"].num_nodes = node_count(d)
    for s in list(d.node_types):
        if s == "node":
            continue
        pre = STORES[s][0] if s in STORES else None
        d[s].num_nodes = d[s][f"{pre}_r_pu"].shape[0] if pre and f"{pre}_r_pu" in d[s] else 0
    datas.append(d)

# Schema must be reconciled ACROSS the batch BEFORE ctx/offsets are built, so every sample
# has the same stores/relations and PyG's edge-offset cumsum sees every sample's node count.
if os.environ.get("FIX_SCHEMA", "1") == "1":
    ensure_batch_schema(datas)
for d in datas:
    ctxs.append(build_recon_ctx(d)); ncs.append(node_count(d))
    scs.append({s: store_size(d, s) for s in keys if s in d.node_types and store_size(d, s) > 0})

print("per-feeder store sets:")
for i, sc in enumerate(scs):
    print(f"  feeder{i}: " + " ".join(f"{s}={n}" for s, n in sorted(sc.items())))
batch = Batch.from_data_list(datas)
bctx = batch_recon_ctx(ctxs, ncs, scs)

print("\nstore: MY merged inj  vs  PyG _inj_index(batch)")
for s in keys:
    mine = bctx["inj"].get(s)
    theirs = _inj_index(batch, s)
    if mine is None and theirs[0].numel() == 0:
        continue
    if mine is None:
        print(f"  {s:12s} MINE MISSING but PyG has {theirs[0].numel()} edges"); continue
    mc, mcol, mn = mine
    tc, tcol, tn = theirs
    if mc.numel() != tc.numel():
        print(f"  {s:12s} COUNT MISMATCH mine={mc.numel()} pyg={tc.numel()}"); continue
    # compare as sorted multisets of (comp,col,node) -- order may differ legitimately
    a = torch.stack([mc, mcol, mn], 1)
    b = torch.stack([tc, tcol, tn], 1)
    sa = a[torch.argsort(a[:, 0] * 1000000 + a[:, 1] * 1000 + a[:, 2])]
    sb = b[torch.argsort(b[:, 0] * 1000000 + b[:, 1] * 1000 + b[:, 2])]
    same = bool(torch.equal(sa, sb))
    tag = "OK" if same else "*** DIFFER ***"
    print(f"  {s:12s} n={mc.numel():6d}  {tag}")
    if not same:
        d0 = (sa != sb).any(1)
        k = int(d0.nonzero()[0]) if d0.any() else 0
        print(f"      first differing row: mine={sa[k].tolist()} pyg={sb[k].tolist()}")
        for j, nm in enumerate(("comp", "col", "node")):
            nd = int((sa[:, j] != sb[:, j]).sum())
            print(f"      {nm}: {nd} differ")
print("\nnode_off (mine):", [0] + list(torch.tensor(ncs).cumsum(0).tolist()))
print("batch node count:", batch["node"].num_nodes, " sum(ncs):", sum(ncs))
