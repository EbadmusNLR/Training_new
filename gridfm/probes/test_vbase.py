"""What is the TRIVIAL baseline for the V target, and what is the model beating?

The model predicts dv = V_solved - V_init, and v_wape = |dv_err| / |V_true|. So
predicting dv = 0 (i.e. "the answer is V_init") already scores |dv| / |V|. If the
reported 4% sits near that number, the model has learned almost nothing and the
metric is flattering it -- V_init is ~1.0 pu and V is 0.95-1.05 pu, so |dv|/|V| is
small BY CONSTRUCTION and a 4% error is enormous against a 5%-wide target.

Reports, per corpus: the dv scale, the dv=0 baseline WAPE, and (as a skyline) the
WAPE of predicting dv's per-corpus MEAN.
"""
import argparse, glob, os, sys
import numpy as np
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios

TD = "/kfs2/projects/gogpt/Ebadmus/training_data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="SMART-DS_1000")
    ap.add_argument("--feeders", type=int, default=40)
    ap.add_argument("--variants", type=int, default=5)
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(TD, a.corpus, "*", "static.pt")))
    step = max(1, len(fs) // a.feeders)
    dvs, vts = [], []
    n_free_tot = 0
    for p in fs[::step][:a.feeders]:
        try:
            sc = FeederScenarios(os.path.dirname(p))
        except Exception:
            continue
        for v in range(min(a.variants, len(sc))):
            d = sc[v]
            nd = d["node"]
            vi = torch.stack([nd.V_r_init_pu, nd.V_i_init_pu], 1).double()
            vs = torch.stack([nd.V_r_pu, nd.V_i_pu], 1).double()
            n = vi.shape[0]
            vis = torch.zeros(n, dtype=torch.bool); vis[0] = True
            rel = ("vsource", "bus1", "node")
            if rel in d.edge_types and d[rel].edge_index.numel():
                vis[d[rel].edge_index[1]] = True
            m = ~vis
            if not m.any():
                continue
            dvs.append((vs - vi)[m].numpy()); vts.append(vs[m].numpy())
            n_free_tot += int(m.sum())
    dv = np.concatenate(dvs); vt = np.concatenate(vts)
    den = np.linalg.norm(vt, axis=1).sum()
    base0 = 100.0 * np.abs(dv).sum() / den                 # predict dv = 0
    mu = dv.mean(0, keepdims=True)
    basem = 100.0 * np.abs(dv - mu).sum() / den            # predict dv = mean(dv)
    vmag = np.linalg.norm(vt, axis=1)
    print(f"=== {a.corpus}: {n_free_tot} masked nodes ===")
    print(f"  |V| (target)        : mean={vmag.mean():.4f}  min={vmag.min():.4f}  max={vmag.max():.4f}")
    print(f"  |dv| per node       : mean={np.abs(dv).sum(1).mean():.3e}  "
          f"p50={np.median(np.abs(dv).sum(1)):.3e}  p99={np.percentile(np.abs(dv).sum(1),99):.3e}")
    print(f"  BASELINE dv=0       : v_wape = {base0:6.3f} %   <- what 'predict V_init' scores")
    print(f"  BASELINE dv=mean(dv): v_wape = {basem:6.3f} %   <- what a constant predictor scores")
    print(f"  => a model at 4% is {'BELOW' if base0 < 4 else 'above'} the dv=0 baseline")


if __name__ == "__main__":
    raise SystemExit(main())
