"""Where does the 0.6s/sample training step actually go?

Measured: 2429s for 4000 samples at bs=8 -> ~4.9s/batch, and bs=4 was ~2.1s/batch. Cost
scales with SAMPLES, not batches, so it is not kernel-launch overhead from small batches --
it is per-sample work. GPU sat at 35%.

Times the pieces separately on the real pipeline:
    collate            build_recon_ctx per variant + batching (dataloader worker cost)
    forward (MP only)  exact_decoder=False, so no reconstruct_full
    forward (exact)    with reconstruct_full
    backward           autograd through whichever forward

If the exact decoder dominates, the lever is the decoder (batching its per-group work),
not the network. If the collate dominates, the lever is workers/caching.

    python -m gridfm.probes.profile_step
"""
import os
import sys
import time

import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")

import glob

from gridfm.dk_data import DKFeeder, DKDataset, make_dk_collate
from gridfm.dk_model import DKSolver
from gridfm.losses import balanced_reconstruction_loss  # noqa: F401  (import cost realism)

CORPUS = os.environ.get("CORPUS", "SMART-DS_1000")
NF = int(os.environ.get("NF", "8"))
BS = int(os.environ.get("BS", "4"))
STEPS = int(os.environ.get("STEPS", "12"))
HID = int(os.environ.get("HID", "256"))
REPS = 3

td = f"/kfs2/projects/gogpt/Ebadmus/training_data/{CORPUS}"
paths = sorted(p for p in glob.glob(os.path.join(td, "*")) if os.path.isdir(p))
step = max(1, len(paths) // NF)
paths = paths[::step][:NF]

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev}  corpus={CORPUS}  feeders={len(paths)}  bs={BS} steps={STEPS} hidden={HID}")

t0 = time.perf_counter()
feeders = [DKFeeder(p) for p in paths]
print(f"DKFeeder build (incl. recon_topo cache): {time.perf_counter()-t0:.1f}s "
      f"for {len(paths)} feeders -> {(time.perf_counter()-t0)/len(paths):.2f}s/feeder")

ds = DKDataset(feeders, list(range(4)), task="pf")
collate = make_dk_collate(feeders)


def timeit(fn, reps=REPS):
    fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


# ---- collate (this is what a dataloader worker does per batch)
samples = [ds[i] for i in range(BS)]
t_get = timeit(lambda: [ds[i] for i in range(BS)])
t_col = timeit(lambda: collate([ds[i] for i in range(BS)]))
print(f"\n__getitem__ x{BS}          : {t_get:7.3f}s  ({t_get/BS:.3f}s/sample)")
print(f"collate x{BS} (ctx+batch)  : {t_col:7.3f}s  ({t_col/BS:.3f}s/sample)  "
      f"-- of which ctx+merge: {t_col-t_get:.3f}s")

batch, plan, rctx = collate([ds[i] for i in range(BS)])
batch = batch.to(dev); batch.tree_plan = plan; batch.recon_ctx = rctx
print(f"batch nodes: {int(batch['node'].num_nodes)}")

for exact in (False, True):
    model = DKSolver(hidden=HID, steps=STEPS, exact_decoder=exact).to(dev)
    tag = "exact decoder " if exact else "MP only (no rf)"

    def fwd():
        return model(batch)
    t_f = timeit(fwd)

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        dv, cur = model(batch)
        loss = dv.pow(2).mean() + sum(c[0].pow(2).mean() + c[1].pow(2).mean() for c in cur.values())
        loss.backward()
    t_fb = timeit(fwd_bwd)
    print(f"\n{tag}: forward {t_f:7.3f}s ({t_f/BS:.3f}s/sample)   "
          f"fwd+bwd {t_fb:7.3f}s ({t_fb/BS:.3f}s/sample)")
