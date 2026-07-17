"""Find DE-ENERGIZED feeders: OpenDSS says converged, but the network carries no power.

The W network (already excluded) had 92% of nodes at V=0 with converged=True. Such a
feeder scores WAPE 1.0 on a |I| of ~1e-8 -- it looks like a decoder failure and is
actually a dead circuit. Decide on the DATA (V and |I|), never on the WAPE.

Prints a verdict per feeder so the exclusion is auditable rather than a name match.
"""
import argparse, glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")


def one(path):
    from core.scenario_store import FeederScenarios
    from gridfm.dk_physics import STORES, store_size, stored_currents
    name = os.path.basename(os.path.dirname(path))
    try:
        d = FeederScenarios(os.path.dirname(path))[0]
        vm = (d["node"].V_r_pu.double() ** 2 + d["node"].V_i_pu.double() ** 2).sqrt()
        vm = vm[1:]                                   # node 0 is ground
        dead = float((vm < 1e-6).double().mean())
        tot = 0.0
        for s in STORES:
            if s in d.node_types and store_size(d, s) > 0:
                a, b = stored_currents(d, s, dtype=torch.float64)
                tot += float(a.abs().sum() + b.abs().sum())
        return (name, dead, float(vm.max()), tot, len(vm), None)
    except Exception as e:
        return (name, -1.0, 0.0, 0.0, 0, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="new_dss_data")
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    root = f"/kfs2/projects/gogpt/Ebadmus/training_data/{a.corpus}"
    fs = sorted(glob.glob(os.path.join(root, "*", "static.pt")))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
        rows = list(ex.map(one, fs, chunksize=1))
    dead = [r for r in rows if r[1] > 0.5]
    tiny = [r for r in rows if 0 <= r[1] <= 0.5 and r[3] < 1e-3]
    print(f"=== {a.corpus}: {len(rows)} feeders")
    print(f"  DE-ENERGIZED (>50% of nodes at V=0): {len(dead)}")
    for n, dd, vx, ii, nn, _e in sorted(dead, key=lambda r: -r[1]):
        print(f"    {n[:76]:78s} dead={100*dd:5.1f}%  Vmax={vx:.3f}  |I|={ii:.2e}")
    print(f"  ENERGIZED but |I|~0 (<1e-3): {len(tiny)}")
    for n, dd, vx, ii, nn, _e in sorted(tiny, key=lambda r: r[3])[:12]:
        print(f"    {n[:76]:78s} dead={100*dd:5.1f}%  Vmax={vx:.3f}  |I|={ii:.2e}")
    errs = [r for r in rows if r[1] < 0]
    print(f"  unreadable: {len(errs)}")
