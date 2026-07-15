"""Corpus-wide scan: is ANY transformer left without a determined map?

Those are the silent-zero risk -- build_xfmr_maps skips them, so their current
stays at whatever it was initialised to. Reports counts by reason, plus the
terminal-count distribution (3-terminal center-taps must NOT be skipped).
"""
import argparse, glob, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")

TD = "/kfs2/projects/gogpt/Ebadmus/training_data"


def one(path):
    from core.scenario_store import FeederScenarios
    from gridfm.dk_physics import store_size
    from gridfm.dk_tree import build_xfmr_maps
    name = os.path.basename(os.path.dirname(path))
    c = Counter()
    try:
        d = FeederScenarios(os.path.dirname(path))[0]
        if "transformer" not in d.node_types or store_size(d, "transformer") == 0:
            return c, []
        n = store_size(d, "transformer")
        unsolved = []
        maps = build_xfmr_maps(d, unsolved=unsolved)
        c["xfmr_total"] += n
        c["xfmr_mapped"] += len(maps)
        for _, why in unsolved:
            c[f"UNSOLVED:{why.split(':')[0]}"] += 1
        # active terminals per transformer (3 = center-tap)
        for m in maps:
            nt = len({int(s) // 4 for s in m["act"].tolist()})
            c[f"terminals_{nt}"] += 1
        return c, [(name, why) for _, why in unsolved]
    except Exception as e:
        c[f"FAIL:{type(e).__name__}"] += 1
        return c, [(name, f"{type(e).__name__}: {e}")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="minimal_component")
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    feeders = sorted(glob.glob(os.path.join(TD, a.corpus, "*", "static.pt")))
    tot = Counter(); bad = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
        for c, b in ex.map(one, feeders, chunksize=4):
            tot.update(c); bad.extend(b)
    print(f"=== {a.corpus}: {len(feeders)} feeders ===")
    for k in sorted(tot):
        print(f"  {k:34s} {tot[k]}")
    print(f"  feeders with unsolved/failed transformers: {len({b[0] for b in bad})}")
    for nm, why in bad[:10]:
        print(f"    {nm[:44]:46s} {why}")


if __name__ == "__main__":
    raise SystemExit(main())
