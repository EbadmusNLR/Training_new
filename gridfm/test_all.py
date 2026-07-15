"""FULL-CORPUS decoder validation: EVERY feeder x EVERY variant, from V/Y/Icomp.

Sharded like datakit/make_training_pt.sbatch (--shard i/N -> feeders[i::N], slurm
array). Per feeder the topology precompute (tree / xfmr null-space maps / injection
indices) is built ONCE and reused across its variants -- that is what makes 100
variants affordable, and it is the same precompute the model needs.

  python gridfm/test_all.py --shard 0/20 --out runs/val        # one shard
  python gridfm/test_all.py --reduce runs/val                  # combine shards
"""
import argparse, glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")

ROOT = os.environ.get("CORPUS_ROOT",
        "/kfs2/projects/gogpt/Ebadmus/training_data/SMART-DS_1000")
FAMILIES = ("load", "pvsystem", "storage", "capacitor", "generator", "line", "transformer", "vsource")


def one(args):
    path, max_var = args
    from core.scenario_store import FeederScenarios
    from gridfm.dk_physics import STORES, store_size, stored_currents, element_currents
    from gridfm.dk_tree import reconstruct_full, build_recon_ctx, SHUNT_STORES
    name = os.path.basename(os.path.dirname(path))
    try:
        fs = FeederScenarios(os.path.dirname(path))
        nvar = len(fs) if max_var <= 0 else min(len(fs), max_var)
        acc = {}
        worst = (0.0, -1)
        ctx = None
        for vi_ in range(nvar):
            d = fs[vi_]
            # topology is reused across variants; the transformer maps are NOT --
            # variants retap the transformers, so Y (and A/B) change with them.
            ctx = build_recon_ctx(d, topo=ctx)
            vr, vim = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
            present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
            truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
            cur = {}
            for s in present:
                if s in SHUNT_STORES:
                    cur[s] = element_currents(d, s, vr, vim)
                else:
                    z = torch.zeros_like(truth[s][0]); cur[s] = (z, z.clone())
            rec = reconstruct_full(d, cur, vr, vim, ctx=ctx)
            vn = vd = 0.0
            for s in present:
                R = rec.get(s, cur[s]); T = truth[s]
                num = float((R[0]-T[0]).abs().sum() + (R[1]-T[1]).abs().sum())
                den = float(T[0].abs().sum() + T[1].abs().sum())
                a = acc.setdefault(s, [0.0, 0.0]); a[0] += num; a[1] += den
                vn += num; vd += den
            w = vn / (vd + 1e-30)
            if w > worst[0]:
                worst = (w, vi_)
        return {"name": name, "nvar": nvar, "acc": acc,
                "worst_wape": worst[0], "worst_var": worst[1], "err": None}
    except Exception as e:
        return {"name": name, "nvar": 0, "acc": {}, "worst_wape": 0.0,
                "worst_var": -1, "err": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None, help="'i/N': feeders[i::N]")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "64")))
    ap.add_argument("--variants", type=int, default=0, help="0 = all variants")
    ap.add_argument("--out", default="runs/val")
    ap.add_argument("--reduce", default=None, help="combine shard jsons in this dir")
    a = ap.parse_args()

    if a.reduce:
        agg, rows, fails, nv = {}, [], [], 0
        for f in sorted(glob.glob(os.path.join(a.reduce, "shard_*.json"))):
            for r in json.load(open(f)):
                if r["err"]:
                    fails.append((r["name"], r["err"])); continue
                nv += r["nvar"]
                fn = fd = 0.0
                for s, (num, den) in r["acc"].items():
                    x = agg.setdefault(s, [0.0, 0.0]); x[0] += num; x[1] += den
                    fn += num; fd += den
                rows.append((fn/(fd+1e-30), fn, r["name"], r["worst_wape"], r["worst_var"]))
        print(f"=== FULL CORPUS: {len(rows)} feeders, {nv} variants, {len(fails)} failed ===")
        tn = td = 0.0
        for s in FAMILIES:
            if s not in agg: continue
            num, den = agg[s]
            print(f"  {s:12s} WAPE = {num/(den+1e-30):.3e}")
            tn += num; td += den
        print(f"  {'AGGREGATE':12s} WAPE = {tn/(td+1e-30):.3e}")
        rows.sort(reverse=True)
        tot = sum(r[1] for r in rows) + 1e-30
        print("\n  worst feeders (mean WAPE | worst single variant):")
        for w, n, name, ww, wv in rows[:10]:
            print(f"    {name[:34]:36s} {w:.3e} | {ww:.3e} @var{wv}  ({n/tot*100:5.2f}% of abs err)")
        for k in (1, 5, 10):
            print(f"  top-{k} feeders carry {sum(r[1] for r in rows[:k])/tot*100:.1f}% of abs error")
        print(f"  feeders with mean WAPE > 1e-6: {sum(1 for r in rows if r[0] > 1e-6)} / {len(rows)}")
        print(f"  feeders with ANY variant > 1e-6: {sum(1 for r in rows if r[3] > 1e-6)} / {len(rows)}")
        for name, err in fails[:15]:
            print(f"  FAIL {name[:40]}: {err}")
        return 0

    feeders = sorted(glob.glob(os.path.join(ROOT, "*", "static.pt")))
    tag = "solo"
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        feeders = feeders[i::n]; tag = f"{i}_of_{n}"
    os.makedirs(a.out, exist_ok=True)
    print(f"shard {tag}: {len(feeders)} feeders, variants={a.variants or 'all'}, "
          f"{a.workers} workers", flush=True)
    t0 = time.time()
    res = []
    ctxm = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctxm) as ex:
        for i, r in enumerate(ex.map(one, [(p, a.variants) for p in feeders], chunksize=1), 1):
            res.append(r)
            if i % 10 == 0:
                print(f"  {i}/{len(feeders)}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(os.path.join(a.out, f"shard_{tag}.json"), "w"))
    print(f"shard {tag} done in {time.time()-t0:.0f}s -> {a.out}/shard_{tag}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
