"""Does reconstruct_full on a PyG-BATCHED graph (merged ctx) == per-feeder reconstruct_full?

This is the correctness gate for the model port: the model forward reconstructs currents on
the batched graph, so the batched path must reproduce the per-feeder exact decoder to ~1e-7.
Builds a real batch with torch_geometric.Batch (the same collate the model uses), runs both,
compares per-store.
"""
import glob, os, sys, time
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from torch_geometric.data import Batch
from core.scenario_store import FeederScenarios
from gridfm.dk_physics import (STORES, store_size, stored_currents, element_currents,
                               node_count, ensure_batch_schema, FC)
from gridfm.dk_tree import (reconstruct_full, build_recon_ctx, batch_recon_ctx,
                            SHUNT_STORES, AMBIG_STORES, SERIES_STORES, UnsupportedNetwork)

CORPUS = os.environ.get("CORPUS", "SMART-DS_1000")
NB = int(os.environ.get("NB", "4"))
TD = f"/kfs2/projects/gogpt/Ebadmus/training_data_ex/{CORPUS}"
paths = sorted(glob.glob(os.path.join(TD, "*", "static.pt")))[:NB]

datas, ctxs, curs, node_counts, store_counts = [], [], [], [], []
per_feeder = []
keys = tuple(SHUNT_STORES) + tuple(SERIES_STORES)
# Verify the FULL batched path: mix radial, coupled-bank, and chord feeders (skip only
# transmission bridges, which the batched path refuses loudly by design). Prefer feeders
# that exercise the hard paths so a regression there can't hide.
paths = [p for p in sorted(glob.glob(os.path.join(TD, "*", "static.pt")))]
from gridfm.dk_tree import _pack_isolated_xfmr
plain, hard = [], []
for p in paths:
    d0 = FeederScenarios(os.path.dirname(p))[0]
    try:
        c0 = build_recon_ctx(d0)
    except UnsupportedNetwork:
        continue                                  # refused per-feeder (e.g. IEEE30 bridges)
    if c0.get("bridges"):
        continue
    other = _pack_isolated_xfmr(c0["xmaps"])[1] if c0["xmaps"] else []
    (hard if (c0["ltree"].get("chords") or other) else plain).append(p)
    if len(hard) >= NB // 2 and len(plain) >= NB - NB // 2:
        break
paths = (hard[:NB // 2] + plain)[:NB]
print(f"selected {len(paths)} feeders ({len(hard[:NB//2])} with chords/coupled banks)")
for p in paths:
    d = FeederScenarios(os.path.dirname(p))[0]
    vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
    present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
    truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
    cur = {}
    for s in present:
        if s in SHUNT_STORES or s in AMBIG_STORES:
            cur[s] = tuple(x.clone() for x in element_currents(d, s, vr, vi))
        else:
            z = torch.zeros_like(truth[s][0]); cur[s] = (z, z.clone())
    # PyG Batch needs explicit node counts per store (the real __getitem__ sets these)
    d["node"].num_nodes = node_count(d)
    for s in list(d.node_types):
        st = d[s]
        if s == "node":
            continue
        prefix = STORES[s][0] if s in STORES else None
        nrow = st[f"{prefix}_r_pu"].shape[0] if prefix and f"{prefix}_r_pu" in st else 0
        st.num_nodes = nrow
    ctx = build_recon_ctx(d)
    rec = reconstruct_full(d, {k: tuple(x.clone() for x in v) for k, v in cur.items()}, vr, vi, ctx=ctx)
    per_feeder.append((present, truth, rec))
    datas.append(d); ctxs.append(ctx); curs.append(cur)
    node_counts.append(node_count(d))
    store_counts.append({s: store_size(d, s) for s in keys if s in d.node_types and store_size(d, s) > 0})

# Every sample must share ONE schema before batching: PyG's edge-offset cumsum skips
# samples lacking a relation, so a feeder without pvsystem/storage makes later feeders'
# edges point into ITS nodes. (Measured: 494 pvsystem + 182 storage node indices wrong.)
ensure_batch_schema(datas)
batch = Batch.from_data_list([d for d in datas])
bctx = batch_recon_ctx(ctxs, node_counts, store_counts)
bvr = torch.cat([d["node"].V_r_pu.double() for d in datas])
bvi = torch.cat([d["node"].V_i_pu.double() for d in datas])
# merged cur: concat each store's rows across samples in PyG order
bcur = {}
for s in keys:
    parts = [c[s] for c in curs if s in c]
    if parts:
        bcur[s] = (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]))
t0 = time.time()
brec = reconstruct_full(batch, bcur, bvr, bvi, ctx=bctx)
tb = time.time() - t0
print(f"batched reconstruct: {1000*tb:.0f} ms for {NB} feeders")

# compare: slice batched output back per feeder and diff vs per-feeder rec
soff = {s: [0] for s in keys}
for sc in store_counts:
    for s in keys:
        soff[s].append(soff[s][-1] + int(sc.get(s, 0)))
worst = 0.0
for i, (present, truth, rec) in enumerate(per_feeder):
    fmax = 0.0; fstore = None
    for s in present:
        if s not in brec:
            continue
        a, b = soff[s][i], soff[s][i + 1]
        Rb = (brec[s][0][a:b], brec[s][1][a:b])
        Rf = rec[s]
        d0 = float((Rb[0] - Rf[0]).abs().max()); d1 = float((Rb[1] - Rf[1]).abs().max())
        if max(d0, d1) > fmax:
            fmax = max(d0, d1); fstore = s
    worst = max(worst, fmax)
    if fmax > 1e-7:
        print(f"  feeder {i} {os.path.basename(os.path.dirname(paths[i]))[-24:]:26s} "
              f"diff {fmax:.2e} on {fstore}  (xfmr={store_counts[i].get('transformer',0)})")
print(f"max |batched - per_feeder| across all stores/feeders: {worst:.3e}")
# and batched WAPE vs truth
tn = td = 0.0
for i, (present, truth, rec) in enumerate(per_feeder):
    for s in present:
        if s not in brec:
            continue
        a, b = soff[s][i], soff[s][i + 1]
        R = (brec[s][0][a:b], brec[s][1][a:b]); T = truth[s]
        tn += float((R[0] - T[0]).abs().sum() + (R[1] - T[1]).abs().sum())
        td += float(T[0].abs().sum() + T[1].abs().sum())
print(f"batched WAPE vs truth: {tn/(td+1e-30):.3e}")
