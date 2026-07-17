"""Do a feeder's VARIANTS actually differ, or are we storing N copies of one sample?

IEEE 30 Bus / 123Bus / IEEE9500 have 100 variants whose V is bit-identical -- the load
never changes. Those contribute 100 duplicates to training, not 100 samples. 13Bus
varies correctly, so it is not universal and must be measured per feeder, per corpus.

Reports, per feeder: how many variants are bit-identical to variant 0, and the spread
of the load's |Icomp| across variants (the driver that is supposed to vary).
"""
import argparse, glob, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
TD = "/kfs2/projects/gogpt/Ebadmus/training_data"


def one(args):
    path, nv = args
    from core.scenario_store import FeederScenarios
    name = os.path.basename(os.path.dirname(path))
    try:
        fs = FeederScenarios(os.path.dirname(path))
        n = min(nv, len(fs)) if nv > 0 else len(fs)
        if n < 2:
            return (name, len(fs), -1, 0.0)
        v0 = fs[0]["node"].V_r_pu
        same = sum(1 for k in range(1, n) if torch.equal(fs[k]["node"].V_r_pu, v0))
        mags = []
        for k in range(0, n, max(1, n // 10)):
            d = fs[k]
            tot = 0.0
            for s in ("load", "pvsystem", "storage", "generator"):
                if s in d.node_types and f"Icomp_r_pu" in d[s]:
                    tot += float(d[s]["Icomp_r_pu"].abs().sum())
            mags.append(tot)
        spread = (max(mags) - min(mags)) / (abs(sum(mags) / len(mags)) + 1e-30) if mags else 0.0
        return (name, len(fs), same, float(spread))
    except Exception as e:
        return (name, 0, -2, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="dss_data")
    ap.add_argument("--variants", type=int, default=0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--feeders", type=int, default=0)
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(TD, a.corpus, "*", "static.pt")))
    if a.feeders:
        step = max(1, len(fs) // a.feeders)
        fs = fs[::step][:a.feeders]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
        for r in ex.map(one, [(p, a.variants) for p in fs], chunksize=1):
            rows.append(r)
    dup = [r for r in rows if r[2] >= 0 and r[1] > 1 and r[2] == min(a.variants or r[1], r[1]) - 1]
    err = [r for r in rows if r[2] == -2]
    print(f"=== {a.corpus}: {len(rows)} feeders ===")
    print(f"  feeders whose variants are ALL IDENTICAL to variant 0: {len(dup)} / {len(rows)}")
    if dup:
        print(f"  -> those store {sum(r[1] for r in dup)} samples that are really {len(dup)}")
        for name, nvar, same, spread in sorted(dup)[:15]:
            print(f"       {name[:46]:48s} {nvar:4d} variants, load spread {spread:.2e}")
    part = [r for r in rows if r[2] > 0 and r not in dup]
    if part:
        print(f"  feeders with SOME duplicate variants: {len(part)}")
        for name, nvar, same, spread in sorted(part, key=lambda r: -r[2])[:6]:
            print(f"       {name[:46]:48s} {same}/{nvar-1} identical")
    if err:
        print(f"  read errors: {len(err)}")
    ok = len(rows) - len(dup) - len(err)
    print(f"  feeders with genuinely varying variants: {ok} / {len(rows)}")


if __name__ == "__main__":
    raise SystemExit(main())
