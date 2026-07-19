"""v5 Y-codebook path smoke: fit_scales(Ycb) -> model with y_cb_head -> forward
-> losses (CE + log-scale) -> backward, on a couple of feeders. CPU, seconds.
Catches shape/dtype/wiring bugs before a queued GPU probe spends its slot."""
import os
import sys
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new/scripts")
import torch
from gridfm.dk_data import (DKFeeder, DKDataset, make_dk_collate, discover_feeders,
                            fit_scales)
from gridfm.dk_model import DKSolver
import dk_train

os.environ.setdefault("YCB_ABS_SCALE", "1")
os.environ.setdefault("YCB_ANALYTIC_SCALE", "1")
os.environ.setdefault("YCB_ANALYTIC_RCOND", "1e-6")

ROOT = "/kfs2/projects/gogpt/Ebadmus/training_data/minimal_component"
fdirs = discover_feeders(ROOT)[:3]
feeders = [DKFeeder(f, need_decoder=False) for f in fdirs]
scales = fit_scales(feeders, variants=[0, 1], max_feeders=3, max_variants=2)
print("Ycb stores:", {s: tuple(scales["Ycb"][s].shape) for s in scales["Ycb"]})

ds = DKDataset(feeders, [0, 1], task="param", use_feat=True, ctx_points=2)
collate = make_dk_collate(feeders, need_ctx=False)
batch, _, _ = collate([ds[0], ds[1], ds[2]])

# Regression for the random4 leakage fixed in T32f: the analytic RHS is
# I_bus + Icomp, so visible I_bus must never authorize truth Icomp when Icomp is
# hidden. Force that visibility combination on one power-conversion component.
leak_probe = None
for s in ("load", "generator", "pvsystem", "storage"):
    if s in batch.node_types and batch[s].yr.shape[0]:
        batch[s].vis_y[0] = False
        batch[s].vis_i[0] = True
        batch[s].vis_ic[0] = False
        leak_probe = (s, 0)
        break

model = DKSolver(hidden=64, steps=3, kcl_feedback=False, use_feat=True,
                 scales=scales, four_mask=True, use_pe=True, ctx_points=2)
model.skip_current = True
print("y_cb_head stores:", list(model.y_cb_head.keys()))

dv, cur, aux = model(batch)
print("y_est stores:", list(aux.get("y_est", {}).keys()),
      "| y_cb_logits:", list(aux.get("y_cb_logits", {}).keys()))
assert all(torch.isfinite(torch.stack(y, -1)).all()
           for y in aux.get("y_est", {}).values()), "analytic Y estimate is non-finite"
accepted = sum(int(v.sum()) for v in aux.get("y_cb_analytic_ok", {}).values())
eligible = sum(int(v.numel()) for v in aux.get("y_cb_analytic_ok", {}).values())
print(f"analytic scale accepted={accepted}/{eligible} "
      f"(rcond={os.environ['YCB_ANALYTIC_RCOND']})")
assert leak_probe is not None, "smoke corpus has no power-conversion component"
ls, li = leak_probe
assert not bool(aux["y_cb_analytic_ok"][ls][li]), \
    "analytic scale consumed hidden Icomp truth"

loss, m = dk_train.losses(batch, dv, cur, scales, use_feat=True, w_v=10.0,
                          w_i=0.0, w_kcl=0.0, norm=True, aux=aux, w_ic=1.0,
                          w_y=1.0, ic_d_only=True, ic_sce=True)
print(f"loss={float(loss):.4f}  y_wape={m['y_wape']:.1f}%  ic_wape={m['ic_wape']:.1f}%")
loss.backward()
gy = sum(p.grad.abs().sum().item() for p in model.y_cb_head.parameters()
         if p.grad is not None)
print(f"y_cb_head grad-sum={gy:.4e}  (nonzero => loss reaches the head)")
print("SMOKE OK")
