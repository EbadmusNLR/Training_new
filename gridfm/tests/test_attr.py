"""Attribute the remaining minimal_component residual (line 9.09e-05, reactor
3.967e-04) across the FULL corpus, in parallel.

Two things the family WAPE cannot distinguish and this can:
  * SERIES vs SHUNT elements (different code paths entirely).
  * REAL error vs the de-energized METRIC ARTIFACT -- WAPE divides by |I_stored|,
    so an element carrying ~1e-13 of solver noise scores 1.0 while our exact 0 is
    the MORE correct answer. `floored` drops any element whose |I| is below FLOOR.
Reports raw and floored, so a residual that is pure artifact is visible as such.
"""
import argparse, glob, json, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")

TD = "/kfs2/projects/gogpt/Ebadmus/training_data"
FLOOR = 1e-9          # |I| below this is solver noise, not a current


def one(a):
    path, nv = a
    from Datakit.core.scenario_store import FeederScenarios
    from gridfm.dk_physics import STORES, store_size, stored_currents, element_currents
    from gridfm.dk_tree import (reconstruct_full, build_recon_ctx, classify_series,
                                SHUNT_STORES, AMBIG_STORES)
    name = os.path.basename(os.path.dirname(path))
    acc = Counter(); worst = []
    try:
        fs = FeederScenarios(os.path.dirname(path))
        d0 = fs[0]
        ser = {s: classify_series(d0, s) for s in AMBIG_STORES
               if s in d0.node_types and store_size(d0, s) > 0}
        ctx = None
        for v in range(min(nv, len(fs)) if nv > 0 else len(fs)):
            d = fs[v]
            ctx = build_recon_ctx(d, topo=ctx)
            vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
            present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
            truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
            cur = {}
            for s in present:
                if s in SHUNT_STORES or s in AMBIG_STORES:
                    cur[s] = element_currents(d, s, vr, vi)
                else:
                    z = torch.zeros_like(truth[s][0]); cur[s] = (z, z.clone())
            rec = reconstruct_full(d, cur, vr, vi, ctx=ctx)
            for s in ("line", "reactor", "capacitor", "transformer", "vsource"):
                if s not in present:
                    continue
                R, T = rec.get(s, cur[s]), truth[s]
                num_e = (R[0]-T[0]).abs().sum(1) + (R[1]-T[1]).abs().sum(1)
                den_e = T[0].abs().sum(1) + T[1].abs().sum(1)
                for c in range(den_e.shape[0]):
                    num, den = float(num_e[c]), float(den_e[c])
                    grp = ("series" if c in ser.get(s, set()) else "shunt") \
                        if s in AMBIG_STORES else s
                    acc[f"{s}|{grp}|num"] += num; acc[f"{s}|{grp}|den"] += den
                    if den >= FLOOR:
                        acc[f"{s}|{grp}|fnum"] += num; acc[f"{s}|{grp}|fden"] += den
                        if num/den > 1e-6:
                            # carry |I| and the ABSOLUTE error: a large relative error on a
                            # near-zero element is not a bug, and only |I| distinguishes them
                            worst.append((round(num/den, 9), name[:30], v, s, grp, c,
                                          float(den), float(num)))
                    else:
                        acc[f"{s}|{grp}|noise"] += 1
        worst.sort(reverse=True)
        return acc, worst[:3], None
    except Exception as e:
        return acc, [], f"{name}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="minimal_component")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--variants", type=int, default=0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default="runs/attr")
    ap.add_argument("--reduce", default=None)
    a = ap.parse_args()

    if a.reduce:
        tot = Counter(); worst = []; fails = []
        for f in sorted(glob.glob(os.path.join(a.reduce, "attr_*.json"))):
            j = json.load(open(f))
            tot.update({k: v for k, v in j["acc"].items()})
            worst.extend(j["worst"]); fails.extend(j["fails"])
        keys = sorted({k.rsplit("|", 1)[0] for k in tot})
        print(f"{'store|group':28s} {'raw WAPE':>11s} {'floored WAPE':>13s} "
              f"{'|I|':>11s} {'noise-elems':>12s}")
        for k in keys:
            num, den = tot.get(f"{k}|num", 0.0), tot.get(f"{k}|den", 0.0)
            fn, fd = tot.get(f"{k}|fnum", 0.0), tot.get(f"{k}|fden", 0.0)
            print(f"{k:28s} {num/(den+1e-30):11.3e} {fn/(fd+1e-30):13.3e} "
                  f"{den:11.4e} {tot.get(f'{k}|noise', 0):12d}")
        print("\n  worst real elements BY RELATIVE error (|I| >= FLOOR):")
        worst.sort(reverse=True)
        for w in worst[:8]:
            print(f"    WAPE={w[0]:.3e}  |I|={w[6]:.3e}  abs_err={w[7]:.3e}  "
                  f"{w[1]:32s} var{w[2]:<3d} {w[3]}|{w[4]} comp{w[5]}")
        print("\n  worst real elements BY ABSOLUTE error (what actually moves the aggregate):")
        worst.sort(key=lambda w: -w[7])
        for w in worst[:8]:
            print(f"    abs_err={w[7]:.3e}  |I|={w[6]:.3e}  WAPE={w[0]:.3e}  "
                  f"{w[1]:32s} var{w[2]:<3d} {w[3]}|{w[4]} comp{w[5]}")
        for f in fails[:10]:
            print(f"  FAIL {f}")
        return 0

    feeders = sorted(glob.glob(os.path.join(TD, a.corpus, "*", "static.pt")))
    tag = "solo"
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        feeders = feeders[i::n]; tag = f"{i}_of_{n}"
    os.makedirs(a.out, exist_ok=True)
    acc = Counter(); worst = []; fails = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
        for c, w, e in ex.map(one, [(p, a.variants) for p in feeders], chunksize=1):
            acc.update(c); worst.extend(w)
            if e: fails.append(e)
    worst.sort(reverse=True)
    json.dump({"acc": dict(acc), "worst": worst[:40], "fails": fails},
              open(os.path.join(a.out, f"attr_{tag}.json"), "w"))
    print(f"shard {tag} done -> {a.out}/attr_{tag}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
