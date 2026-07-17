"""Does the MODEL path now decode currents exactly? End-to-end, not a reimplementation.

Runs the real pipeline -- DKFeeder -> DKDataset -> make_dk_collate -> DKSolver
._completed_currents -- and feeds it TRUTH V, so any error is purely the decoder. That
isolates the decoder from the (separate, unsolved) voltage problem.

Compares:
  exact_decoder=True   reconstruct_full  (needs V + the batched recon ctx)
  exact_decoder=False  reconstruct_vectorized (the old model path; never took V)

Also checks the thing that silently breaks training: gradients must flow from the decoded
currents back to V. A decoder that is exact but detached teaches the model nothing.

  CORPUS=SMART-DS_1000 NF=4 python -m gridfm.probes.verify_model_decoder
"""
import glob
import os
import sys

import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")

from gridfm.dk_data import DKFeeder, DKDataset, make_dk_collate
from gridfm.dk_model import DKSolver
from gridfm.dk_physics import STORES

CORPUS = os.environ.get("CORPUS", "SMART-DS_1000")
NF = int(os.environ.get("NF", "4"))
NV = int(os.environ.get("NV", "2"))
TD = f"/kfs2/projects/gogpt/Ebadmus/training_data/{CORPUS}"

paths = sorted(p for p in glob.glob(os.path.join(TD, "*")) if os.path.isdir(p))
step = max(1, len(paths) // NF)
paths = paths[::step][:NF]
print(f"corpus={CORPUS}  feeders={len(paths)}  variants={NV}\n")

feeders = [DKFeeder(p) for p in paths]
ds = DKDataset(feeders, list(range(NV)), task="pf")
collate = make_dk_collate(feeders)
samples = [ds[i] for i in range(len(ds))]
batch, plan, rctx = collate(samples)
batch.tree_plan = plan
batch.recon_ctx = rctx
print(f"batch: {int(batch['node'].num_nodes)} nodes, {len(samples)} samples\n")


def run(exact, grad=False):
    model = DKSolver(hidden=32, steps=1, exact_decoder=exact)
    edges = model._edges(batch)
    nd = batch["node"]
    v = (nd.v_init + nd.dv).clone()          # TRUTH V: isolate the decoder
    if grad:
        v.requires_grad_(True)
    cur = model._completed_currents(batch, edges, None, v)
    return cur, v


for exact in (False, True):
    cur, _ = run(exact)
    tag = "reconstruct_full (EXACT)" if exact else "reconstruct_vectorized (OLD)"
    tot_n = tot_d = 0.0
    rows = []
    for s in cur:
        st = batch[s]
        tr, ti = st.ir, st.ii
        pr, pi = cur[s]
        n = float((pr - tr).abs().sum() + (pi - ti).abs().sum())
        d = float(tr.abs().sum() + ti.abs().sum())
        tot_n += n; tot_d += d
        rows.append((s, n / (d + 1e-30), float(tr.abs().sum())))
    print(f"--- {tag}")
    for s, w, mag in sorted(rows):
        flag = "  <-- SILENTLY ZERO" if abs(w - 1.0) < 1e-9 and mag > 1e-9 else ""
        print(f"      {s:12s} WAPE={w:.3e}   |I_truth|={mag:.3e}{flag}")
    print(f"      {'TOTAL':12s} WAPE={tot_n/(tot_d+1e-30):.3e}\n")

# gradient flow: exact but detached would train nothing
cur, v = run(True, grad=True)
loss = sum((c[0].pow(2).sum() + c[1].pow(2).sum()) for c in cur.values())
loss.backward()
g = v.grad
print(f"--- gradient flow through the exact decoder")
print(f"      dLoss/dV nonzero: {int((g.abs() > 0).sum())}/{g.numel()}   "
      f"max|grad|={float(g.abs().max()):.3e}")
print("      OK" if float(g.abs().max()) > 0 else "      DETACHED -- decoder teaches nothing")

# ---- is fp32 the accuracy floor? The reference reaches 4.6e-08 in fp64; the model
# decodes in fp32. If fp64 decoding buys orders of magnitude, the decoder should run in
# fp64 regardless of the network's dtype -- it is a physics map, not learned weights.
print(f"\n--- decoder dtype sweep (truth V, exact path)")
for dt in (torch.float32, torch.float64):
    model = DKSolver(hidden=32, steps=1, exact_decoder=True)
    edges = model._edges(batch)
    nd = batch["node"]
    v = (nd.v_init + nd.dv).to(dt)
    saved = {}
    for s in list(batch.node_types):
        st = batch[s]
        for k, val in list(st.items()):
            if torch.is_tensor(val) and val.dtype == torch.float32:
                saved[(s, k)] = val
                st[k] = val.to(dt)
    cur = model._completed_currents(batch, edges, None, v)
    tn = td = 0.0
    for s in cur:
        st = batch[s]
        tn += float((cur[s][0] - st.ir).abs().sum() + (cur[s][1] - st.ii).abs().sum())
        td += float(st.ir.abs().sum() + st.ii.abs().sum())
    for (s, k), val in saved.items():
        batch[s][k] = val
    print(f"      {str(dt):16s} TOTAL WAPE={tn/(td+1e-30):.3e}")
