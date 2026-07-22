import glob, os, sys, time
sys.path.insert(0,"/kfs2/projects/gogpt/Ebadmus/datakit"); sys.path.insert(0,"/kfs2/projects/gogpt/Ebadmus/Training_new")
from Datakit.core.scenario_store import FeederScenarios
from gridfm.dk_tree import build_recon_ctx
TD="/kfs2/projects/gogpt/Ebadmus/training_data"
for corpus in ("SMART-DS_1000","minimal_component"):
    fs=sorted(glob.glob(os.path.join(TD,corpus,"*","static.pt")))
    step=max(1,len(fs)//4)
    for p in fs[::step][:4]:
        d=FeederScenarios(os.path.dirname(p))[0]
        t=time.time()
        try:
            ctx=build_recon_ctx(d)
            gs=[sum(len(v[0]) for v in g["scatter"].values()) for g in ctx["xmaps"]]
            print(f"{corpus[:18]:20s} {os.path.basename(os.path.dirname(p))[:26]:28s} "
                  f"groups={len(gs):4d} max_unknowns={max(gs) if gs else 0:4d} "
                  f"ctx={time.time()-t:5.1f}s")
        except Exception as e:
            print(f"{corpus[:18]:20s} {os.path.basename(os.path.dirname(p))[:26]:28s} FAIL {type(e).__name__}: {str(e)[:60]}")
