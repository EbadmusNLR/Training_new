"""Are a feeder's VARIANTS actually different, and WHICH quantity is being perturbed?
Per feeder, across variants, measure the relative SPREAD of:
  V     - the solved voltage (the effect a perturbation should have)
  Icomp - load/gen compensation current (the usual perturbation TARGET)
  Yline - line series admittance (impedance perturbation, if any)
  Yxfmr - transformer admittance (tap perturbation, if any)
spread ~ 0  => that quantity is IDENTICAL across variants (not perturbed).
Reports how many feeders have DEAD variants (V identical => 100 duplicate samples).
"""
import argparse, glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")

def spread(vals):
    if len(vals) < 2: return 0.0
    t = torch.stack(vals); m = t.mean(0)
    return float((t - m).abs().mean() / (m.abs().mean() + 1e-30))

def one(path):
    from core.scenario_store import FeederScenarios
    name = os.path.basename(os.path.dirname(path))
    try:
        fs = FeederScenarios(os.path.dirname(path)); nv = len(fs)
        if nv < 2: return (name, nv, -1, {})
        K = min(nv, 20)
        V, IC, YL, YX = [], [], [], []
        v0 = None; ident = 0
        for k in range(K):
            d = fs[k]
            vv = torch.stack([d["node"].V_r_pu, d["node"].V_i_pu]).flatten()
            V.append(vv)
            if v0 is None: v0 = vv
            elif vv.shape==v0.shape and torch.equal(vv, v0): ident += 1
            ic = []
            for s in ("load","pvsystem","generator","storage"):
                if s in d.node_types and "Icomp_r_pu" in d[s]:
                    ic.append(d[s]["Icomp_r_pu"].flatten())
            IC.append(torch.cat(ic) if ic else torch.zeros(1))
            YL.append(d["line"]["Ys_r_pu"].flatten() if "line" in d.node_types and d["line"]["Ys_r_pu"].numel() else torch.zeros(1))
            YX.append(d["transformer"]["Yxfmr_r_pu"].flatten() if "transformer" in d.node_types and d["transformer"]["Yxfmr_r_pu"].numel() else torch.zeros(1))
        return (name, nv, ident, {"V":spread(V), "Icomp":spread(IC), "Yline":spread(YL), "Yxfmr":spread(YX)})
    except Exception as e:
        return (name, 0, -2, {"err": str(e)[:50]})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--corpus", default="SMART-DS_1000")
    ap.add_argument("--workers", type=int, default=64); ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    root = f"/kfs2/projects/gogpt/Ebadmus/training_data/{a.corpus}"
    fs = sorted(glob.glob(os.path.join(root, "*", "static.pt")))
    if a.n: fs = fs[::max(1,len(fs)//a.n)][:a.n]
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
        rows = list(ex.map(one, fs, chunksize=1))
    ok = [r for r in rows if r[2] >= 0]
    dead = [r for r in ok if r[3].get("V",0) < 1e-9]                       # V identical across variants
    icdead = [r for r in ok if r[3].get("Icomp",0) < 1e-9]                 # loads not perturbed
    print(f"=== {a.corpus}: {len(ok)} feeders (>=2 variants)")
    print(f"  V IDENTICAL across variants (DUPLICATES): {len(dead)}/{len(ok)}")
    print(f"  Icomp identical (loads NOT perturbed):    {len(icdead)}/{len(ok)}")
    import statistics as st
    for q in ("V","Icomp","Yline","Yxfmr"):
        sv = sorted(r[3].get(q,0) for r in ok)
        if sv:
            med = sv[len(sv)//2]; frac0 = sum(1 for x in sv if x<1e-9)/len(sv)
            print(f"  {q:6s}: median spread {med:.2e}   fraction ~0: {100*frac0:.1f}%")
    if dead[:5]:
        print("  sample DUPLICATE feeders:")
        for n,nv,idc,sp in dead[:5]: print(f"    {n[-50:]:52s} Icomp spread {sp.get('Icomp',0):.1e}")
