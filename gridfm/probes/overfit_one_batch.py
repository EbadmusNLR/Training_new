"""Can DKSolver overfit ONE batch? If not, no amount of data or epochs will help.

The seed tests sat at v_skill ~= 1.0 (no better than dv=0) with train V WORSE than the
null and not decreasing. Two suspects, both measurable here:

  1. Loss balance. losses() itself documents that without --norm-loss the current term
     outweighs the V term 100-800x and "the model largely ignored V -- measured: v_skill
     ~1.0 with the mixed loss vs 0.37 with V-only". The seed tests ran WITHOUT it.
  2. Decoder pullback. Since the exact-decoder port, i_mse backprops through
     reconstruct_full, so the current loss is a V loss reweighted by dI/dV -- gradient
     scale unknown, possibly swamping everything else through the clip.

Protocol: one fixed batch, N steps of Adam, several loss configs. Reports per-term
GRADIENT NORMS (the thing that actually decides what the optimizer hears) and the
train-batch v_skill trajectory. A healthy config must drive v_skill well under 1 and
v_mse toward 0 on data it sees every step.

    CONFIGS=mixed,norm,vonly,vonly_hi STEPS=400 python -m gridfm.probes.overfit_one_batch
"""
import glob
import os
import sys
import time

import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")

from gridfm.dk_data import DKFeeder, DKDataset, make_dk_collate, fit_scales
from gridfm.dk_model import DKSolver

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new/scripts")
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "dk_train", "/kfs2/projects/gogpt/Ebadmus/Training_new/scripts/dk_train.py")
dk_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dk_train)

CORPUS = os.environ.get("CORPUS", "SMART-DS_1000")
NF = int(os.environ.get("NF", "6"))
BS = int(os.environ.get("BS", "6"))
STEPS = int(os.environ.get("STEPS", "400"))
LR = float(os.environ.get("LR", "4e-4"))
CONFIGS = os.environ.get("CONFIGS", "mixed,norm,vonly,vonly_hi").split(",")

td = f"/kfs2/projects/gogpt/Ebadmus/training_data/{CORPUS}"
paths = sorted(p for p in glob.glob(os.path.join(td, "*")) if os.path.isdir(p))
step = max(1, len(paths) // NF)
paths = paths[::step][:NF]
dev = "cuda" if torch.cuda.is_available() else "cpu"

feeders = [DKFeeder(p) for p in paths]
ds = DKDataset(feeders, [0], task="pf")
collate = make_dk_collate(feeders)
batch, plan, rctx = collate([ds[i] for i in range(min(BS, len(ds)))])
batch = batch.to(dev)
batch.tree_plan = plan
batch.recon_ctx = rctx
scales = fit_scales(feeders, list(range(min(4, 100))))
print(f"device={dev} batch_nodes={int(batch['node'].num_nodes)} configs={CONFIGS} steps={STEPS} lr={LR}\n")

# loss-term settings per config: (w_v, w_i, w_kcl, norm)
SETTINGS = {
    "mixed":    (10.0, 1.0, 0.1, False),   # what the seed tests ran
    "norm":     (10.0, 1.0, 0.1, True),    # the documented fix
    "vonly":    (10.0, 0.0, 0.0, False),   # no decoder in the gradient at all
    "vonly_hi": (10.0, 0.0, 0.0, False),   # + 10x lr (is 4e-4 just too small for dv~1e-2?)
}


def grad_norm_of(model, term):
    model.zero_grad(set_to_none=True)
    term.backward(retain_graph=True)
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.pow(2).sum())
    return total ** 0.5


for cfg in CONFIGS:
    w_v, w_i, w_kcl, norm = SETTINGS[cfg]
    lr = LR * (10.0 if cfg == "vonly_hi" else 1.0)
    torch.manual_seed(0)
    model = DKSolver(hidden=256, steps=12, scales=scales).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    print(f"=== {cfg}  (w_v={w_v} w_i={w_i} w_kcl={w_kcl} norm={norm} lr={lr:g})")

    # per-term gradient norms at init: what does the optimizer actually hear?
    dv, cur = model(batch)
    _, m0 = dk_train.losses(batch, dv, cur, scales, w_v=w_v, w_i=w_i, w_kcl=w_kcl, norm=norm)
    nd = batch["node"]; msk = nd.msk_v
    v_mse = ((dv - nd.dv)[msk] ** 2).mean()
    v_term = v_mse / ((nd.dv[msk] ** 2).mean() + 1e-12) if norm else v_mse
    gv = grad_norm_of(model, w_v * v_term)
    gi = 0.0
    if w_i > 0:
        loss_full, _ = dk_train.losses(batch, dv, cur, scales, w_v=0.0, w_i=w_i, w_kcl=0.0, norm=norm)
        gi = grad_norm_of(model, loss_full)
    print(f"    grad norms at init:  |g_V|={gv:.3e}   |g_I(decoder)|={gi:.3e}   "
          f"ratio I/V={gi/max(gv,1e-30):.1f}")

    t0 = time.time()
    for it in range(1, STEPS + 1):
        dv, cur = model(batch)
        loss, m = dk_train.losses(batch, dv, cur, scales, w_v=w_v, w_i=w_i, w_kcl=w_kcl, norm=norm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it in (1, 25, 50, 100, 200, 300, STEPS):
            print(f"    step{it:4d}  loss={float(loss):.4e}  v_mse={m['v_mse']:.4e}  "
                  f"v_skill={m['v_skill']:.3f}  i_wape={m['i_wape']:.2f}%  "
                  f"|grad|={float(gn):.2e}")
    print(f"    ({time.time()-t0:.0f}s)\n")
