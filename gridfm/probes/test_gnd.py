"""Ground-touching SERIES conductors: excluded from the tree (`if n1==0 or n2==0`),
so their current is silently 0. How many are there, and do they carry real current?
Split by store, and by whether the ELEMENT is series or shunt (a shunt reactor's
grounded leg MUST stay out of the tree; a series line's grounded neutral must not)."""
import argparse, glob, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
TD = "/kfs2/projects/gogpt/Ebadmus/training_data"

def one(path):
    from Datakit.core.scenario_store import FeederScenarios
    from gridfm.dk_physics import store_size, stored_currents
    from gridfm.dk_tree import _series_edges, classify_series, TREE_STORES, AMBIG_STORES
    c = Counter()
    name = os.path.basename(os.path.dirname(path))
    try:
        d = FeederScenarios(os.path.dirname(path))[0]
        ser = {s: classify_series(d, s) for s in AMBIG_STORES
               if s in d.node_types and store_size(d, s) > 0}
        E = _series_edges(d, TREE_STORES)
        cur = {s: stored_currents(d, s, dtype=torch.float64)
               for s in {e[0] for e in E} if s in d.node_types}
        worst = 0.0
        for (s, comp, n1, n2, ca, cb) in E:
            if n1 != 0 and n2 != 0:
                continue
            iselem_series = (comp in ser[s]) if s in ser else True
            grp = f"{s}|{'series' if iselem_series else 'shunt'}"
            c[f"gnd_{grp}"] += 1
            Ir, Ii = cur[s]
            mag = float(Ir[comp, ca].abs() + Ii[comp, ca].abs()
                        + Ir[comp, cb].abs() + Ii[comp, cb].abs())
            if mag > 1e-9:
                c[f"gnd_{grp}_LIVE"] += 1
                worst = max(worst, mag)
        c["feeders"] += 1
        if worst > 1e-9: c["feeders_with_live_gnd_series"] += 1
        return c, ((name, worst) if worst > 1e-9 else None)
    except Exception as e:
        c[f"FAIL:{type(e).__name__}"] += 1
        return c, None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--corpus", default="dss_data")
    ap.add_argument("--workers", type=int, default=64); a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(TD, a.corpus, "*", "static.pt")))
    tot = Counter(); rows = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
        for c, r in ex.map(one, fs, chunksize=2):
            tot.update(c)
            if r: rows.append(r)
    print(f"=== {a.corpus} ===")
    for k in sorted(tot): print(f"  {k:34s} {tot[k]}")
    rows.sort(key=lambda r: -r[1])
    for nm, w in rows[:6]: print(f"    {nm[:44]:46s} worst |I| on a grounded series conductor = {w:.3e}")

if __name__ == "__main__":
    raise SystemExit(main())
