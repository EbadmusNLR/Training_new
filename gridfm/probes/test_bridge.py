"""How many feeders have BRIDGES (line between two rooted trees) or CHORDS?
Blast radius of the root pre-marking change, per corpus."""
import argparse, glob, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
TD = "/kfs2/projects/gogpt/Ebadmus/training_data"

def one(path):
    from core.scenario_store import FeederScenarios
    from gridfm.dk_tree import _series_edges, _slack_xfmrsec_roots, _tree_from_edges, TREE_STORES
    name = os.path.basename(os.path.dirname(path))
    c = Counter()
    try:
        d = FeederScenarios(os.path.dirname(path))[0]
        E = _series_edges(d, TREE_STORES)
        slack, xsec = _slack_xfmrsec_roots(d)
        tr = _tree_from_edges(E, slack | xsec)
        nb, nc = len(tr["bridges"]), len(tr["chords"])
        c["feeders"] += 1
        c["bridge_edges"] += nb; c["chord_edges"] += nc
        if nb: c["feeders_with_bridges"] += 1
        if nc: c["feeders_with_chords"] += 1
        bstores = Counter(E[i][0] for i in tr["bridges"])
        for k, v in bstores.items(): c[f"bridge_store_{k}"] += v
        return c, ((name, nb, nc) if nb else None)
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
    for k in sorted(tot): print(f"  {k:26s} {tot[k]}")
    rows.sort(key=lambda r: -r[1])
    for nm, nb, nc in rows[:12]: print(f"    {nm[:44]:46s} bridges={nb} chords={nc}")

if __name__ == "__main__":
    raise SystemExit(main())
