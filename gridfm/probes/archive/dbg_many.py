"""Diagnose MANY feeders in ONE srun with a worker pool -- serial per-feeder srun probes
pay scheduler + store-load latency each. Usage:
  CORPUS=new_dss_data JACOBI=6 python gridfm/dbg_many.py <substr1> <substr2> ...
Prints TOTAL + worst-family WAPE per matched feeder.
"""
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
CORPUS = os.environ.get("CORPUS", "new_dss_data")
TD = f"/kfs2/projects/gogpt/Ebadmus/training_data/{CORPUS}"

def one(path):
    from core.scenario_store import FeederScenarios
    from gridfm.dk_physics import STORES, store_size, stored_currents, element_currents
    from gridfm.dk_tree import (reconstruct_full, build_recon_ctx, SHUNT_STORES,
                                AMBIG_STORES, UnsupportedNetwork)
    name = os.path.basename(os.path.dirname(path))
    try:
        d = FeederScenarios(os.path.dirname(path))[0]
        vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
        present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
        truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
        cur = {s: (element_currents(d, s, vr, vi) if (s in SHUNT_STORES or s in AMBIG_STORES)
                   else (torch.zeros_like(truth[s][0]), torch.zeros_like(truth[s][0]))) for s in present}
        rec = reconstruct_full(d, cur, vr, vi, ctx=build_recon_ctx(d))
        fam, tn, td = {}, 0.0, 0.0
        for s in present:
            R, T = rec.get(s, cur[s]), truth[s]
            num = float((R[0]-T[0]).abs().sum()+(R[1]-T[1]).abs().sum()); den = float(T[0].abs().sum()+T[1].abs().sum())
            fam[s] = (num/(den+1e-30), den); tn += num; td += den
        worst = sorted(((w, s) for s, (w, dn) in fam.items() if dn > 1e-9), reverse=True)[:3]
        return (name, tn/(td+1e-30), td, worst, None)
    except UnsupportedNetwork as e:
        return (name, -1.0, 0.0, [], f"REFUSED: {str(e)[:60]}")
    except Exception as e:
        return (name, -2.0, 0.0, [], f"{type(e).__name__}: {str(e)[:60]}")

subs = sys.argv[1:]
paths = [p for p in sorted(glob.glob(os.path.join(TD, "*", "static.pt")))
         if not subs or any(s in os.path.basename(os.path.dirname(p)) for s in subs)]
with ProcessPoolExecutor(max_workers=min(64, len(paths)), mp_context=mp.get_context("fork")) as ex:
    rows = list(ex.map(one, paths, chunksize=1))
for name, tot, den, worst, err in sorted(rows, key=lambda r: -r[1]):
    tag = err if err else f"{tot:.3e}  worst={[(s, f'{w:.1e}') for w,s in worst]}"
    print(f"  {name[-52:]:54s} |I|={den:8.2e}  {tag}")
