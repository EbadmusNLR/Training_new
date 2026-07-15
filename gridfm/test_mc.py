"""Validate the decoder on minimal_component (reactors + generators) -- the corpus
that contains the structures SMART-DS never had. Sharded/parallel."""
import argparse, glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
FAM = ("load","generator","pvsystem","storage","capacitor","line","reactor","transformer","vsource")

def one(a):
    path, mv = a
    from core.scenario_store import FeederScenarios
    from gridfm.dk_physics import STORES, store_size, stored_currents, element_currents
    from gridfm.dk_tree import reconstruct_full, build_recon_ctx, SHUNT_STORES, AMBIG_STORES
    name = os.path.basename(os.path.dirname(path))
    try:
        fs = FeederScenarios(os.path.dirname(path))
        nvar = len(fs) if mv <= 0 else min(len(fs), mv)
        acc = {}; ctx = None
        for v in range(nvar):
            d = fs[v]
            ctx = build_recon_ctx(d, topo=ctx)   # topology reused; xfmr maps rebuilt (taps vary)
            vr, vim = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
            present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
            truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
            cur = {}
            for s in present:
                # decode EVERY 1-term shunt AND every ambiguous 2-term store; the
                # series-classified rows get overwritten by the tree inside recon.
                if s in SHUNT_STORES or s in AMBIG_STORES:
                    cur[s] = element_currents(d, s, vr, vim)
                else:
                    z = torch.zeros_like(truth[s][0]); cur[s] = (z, z.clone())
            rec = reconstruct_full(d, cur, vr, vim, ctx=ctx)
            for s in present:
                R = rec.get(s, cur[s]); T = truth[s]
                num = float((R[0]-T[0]).abs().sum()+(R[1]-T[1]).abs().sum())
                den = float(T[0].abs().sum()+T[1].abs().sum())
                x = acc.setdefault(s, [0.0, 0.0]); x[0] += num; x[1] += den
        return {"name": name, "nvar": nvar, "acc": acc, "err": None}
    except Exception as e:
        return {"name": name, "nvar": 0, "acc": {}, "err": f"{type(e).__name__}: {e}"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="minimal_component")
    ap.add_argument("--shard", default=None); ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--variants", type=int, default=0); ap.add_argument("--out", default="runs/valmc")
    ap.add_argument("--reduce", default=None)
    a = ap.parse_args()
    if a.reduce:
        agg={}; nf=0; nv=0; fails=[]; rows=[]
        for f in sorted(glob.glob(os.path.join(a.reduce, "shard_*.json"))):
            for r in json.load(open(f)):
                if r["err"]: fails.append((r["name"], r["err"])); continue
                nf+=1; nv+=r["nvar"]; fn=fd=0.0
                for s,(num,den) in r["acc"].items():
                    x=agg.setdefault(s,[0.0,0.0]); x[0]+=num; x[1]+=den; fn+=num; fd+=den
                rows.append((fn/(fd+1e-30), r["name"]))
        print(f"=== {nf} feeders, {nv} variants, {len(fails)} failed ===")
        tn=td=0.0
        for s in FAM:
            if s not in agg: continue
            num,den=agg[s]; tn+=num; td+=den
            print(f"  {s:12s} WAPE = {num/(den+1e-30):.3e}")
        print(f"  {'AGGREGATE':12s} WAPE = {tn/(td+1e-30):.3e}")
        rows.sort(reverse=True)
        print("  worst feeders:", [(f"{w:.1e}", n[:22]) for w, n in rows[:5]])
        print(f"  feeders WAPE>1e-6: {sum(1 for w,_ in rows if w>1e-6)}/{len(rows)}")
        for n,e in fails[:8]: print(f"  FAIL {n[:34]}: {e}")
        raise SystemExit(0)
    TD = f"/kfs2/projects/gogpt/Ebadmus/training_data/{a.corpus}"
    feeders = sorted(glob.glob(os.path.join(TD, "*", "static.pt")))
    tag="solo"
    if a.shard:
        i,n = (int(x) for x in a.shard.split("/")); feeders = feeders[i::n]; tag=f"{i}_of_{n}"
    os.makedirs(a.out, exist_ok=True)
    print(f"shard {tag}: {len(feeders)} feeders", flush=True)
    t0=time.time(); res=[]
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
        for i,r in enumerate(ex.map(one, [(p,a.variants) for p in feeders], chunksize=1),1):
            res.append(r)
    json.dump(res, open(os.path.join(a.out,f"shard_{tag}.json"),"w"))
    print(f"shard {tag} done {time.time()-t0:.0f}s", flush=True)
