"""Isolate the batched-recon bug: batch-of-1 vs per-feeder on EXPLICIT feeders.

Takes feeder dirs on argv (no corpus scan -- the scan was what made earlier runs time out).

  batch of 1  wrong  -> bug is in the merge/apply itself (all offsets are 0)
  batch of 1  exact  -> bug is CROSS-FEEDER (offsets / shared state)

Then re-run with 2 feeders (small + big) to confirm which.
"""
import os, sys
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from torch_geometric.data import Batch
from core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, store_size, stored_currents, element_currents, node_count
from gridfm.dk_tree import (reconstruct_full, build_recon_ctx, batch_recon_ctx,
                            SHUNT_STORES, AMBIG_STORES, SERIES_STORES, _pack_isolated_xfmr)

keys = tuple(SHUNT_STORES) + tuple(SERIES_STORES)


def load(fdir):
    d = FeederScenarios(fdir)[0]
    d["node"].num_nodes = node_count(d)
    for s in list(d.node_types):
        if s == "node":
            continue
        pre = STORES[s][0] if s in STORES else None
        d[s].num_nodes = d[s][f"{pre}_r_pu"].shape[0] if pre and f"{pre}_r_pu" in d[s] else 0
    return d


def mkcur(d, truth, present, vr, vi):
    cur = {}
    for s in present:
        if s in SHUNT_STORES or s in AMBIG_STORES:
            cur[s] = tuple(x.clone() for x in element_currents(d, s, vr, vi))
        else:
            z = torch.zeros_like(truth[s][0]); cur[s] = (z, z.clone())
    return cur


datas, ctxs, curs, ncs, scs, per = [], [], [], [], [], []
for fdir in sys.argv[1:]:
    d = load(fdir)
    vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
    present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
    truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
    cur = mkcur(d, truth, present, vr, vi)
    ctx = build_recon_ctx(d)
    nxf = store_size(d, "transformer") if "transformer" in d.node_types else 0
    niso = len(ctx.get("xpacked") or []); nother = len(_pack_isolated_xfmr(ctx["xmaps"])[1]) if ctx["xmaps"] else 0
    print(f"  {os.path.basename(fdir)[-28:]:30s} xfmr={nxf:4d} groups={len(ctx['xmaps']):4d} "
          f"iso_classes={niso} coupled={nother} chords={len(ctx['ltree'].get('chords',[]))}")
    rec = reconstruct_full(d, {k: tuple(x.clone() for x in v) for k, v in cur.items()}, vr, vi, ctx=ctx)
    per.append((present, truth, rec))
    datas.append(d); ctxs.append(ctx); curs.append(cur)
    ncs.append(node_count(d))
    scs.append({s: store_size(d, s) for s in keys if s in d.node_types and store_size(d, s) > 0})

batch = Batch.from_data_list(datas)
bctx = batch_recon_ctx(ctxs, ncs, scs)
bvr = torch.cat([d["node"].V_r_pu.double() for d in datas])
bvi = torch.cat([d["node"].V_i_pu.double() for d in datas])
bcur = {}
for s in keys:
    parts = [c[s] for c in curs if s in c]
    if parts:
        bcur[s] = (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]))
brec = reconstruct_full(batch, bcur, bvr, bvi, ctx=bctx)

soff = {s: [0] for s in keys}
for sc in scs:
    for s in keys:
        soff[s].append(soff[s][-1] + int(sc.get(s, 0)))
print()
for i, (present, truth, rec) in enumerate(per):
    for s in present:
        if s not in brec:
            continue
        a, b = soff[s][i], soff[s][i + 1]
        d0 = float((brec[s][0][a:b] - rec[s][0]).abs().max())
        d1 = float((brec[s][1][a:b] - rec[s][1]).abs().max())
        if max(d0, d1) > 1e-9:
            # how many rows differ, and is the batched one ZERO there?
            m = (brec[s][0][a:b] - rec[s][0]).abs().max(dim=1).values > 1e-9
            bz = float(brec[s][0][a:b][m].abs().sum()) if m.any() else 0.0
            print(f"  feeder{i} {s:12s} diff={max(d0,d1):.3e}  rows_differing={int(m.sum())}/{int(m.numel())}"
                  f"  |batched@diff|={bz:.3e}")
