"""Isolate the residual error in reconstruct_full: break transformer into primary
(bus1) vs secondary (bus2/3), and lines, vs stored. Zero-init, stored shunts."""
import glob, os, sys
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
from core.scenario_store import FeederScenarios
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from gridfm.dk_physics import STORES, store_size, stored_currents
from gridfm.dk_tree import reconstruct_full

ROOT = "/kfs2/projects/gogpt/Ebadmus/training_data/SMART-DS_1000"
feeders = sorted(glob.glob(os.path.join(ROOT, "*", "static.pt")), key=os.path.getsize)
probe = feeders[len(feeders) // 2: len(feeders) // 2 + 3]


def wape(R, T):
    return float((R - T).abs().sum() / (T.abs().sum() + 1e-12))


agg = {"prim": [0.0, 0.0], "sec": [0.0, 0.0], "line": [0.0, 0.0]}
for p in probe:
    d = FeederScenarios(os.path.dirname(p))[0]
    cur = {s: stored_currents(d, s, dtype=torch.float64) for s in STORES
           if s in d.node_types and store_size(d, s) > 0}
    if "transformer" not in cur:
        continue
    rec = reconstruct_full(d, cur, d["node"].V_r_pu, d["node"].V_i_pu)
    Rr, Ri = rec["transformer"]; Tr, Ti = cur["transformer"]
    # primary = cols 0-3, secondary = cols 4-11
    for lab, cols in (("prim", slice(0, 4)), ("sec", slice(4, 12))):
        num = float((Rr[:, cols]-Tr[:, cols]).abs().sum() + (Ri[:, cols]-Ti[:, cols]).abs().sum())
        den = float(Tr[:, cols].abs().sum() + Ti[:, cols].abs().sum() + 1e-12)
        agg[lab][0] += num; agg[lab][1] += den
    lr, li = rec["line"]; tr, ti = cur["line"]
    agg["line"][0] += float((lr-tr).abs().sum()+(li-ti).abs().sum())
    agg["line"][1] += float(tr.abs().sum()+ti.abs().sum()+1e-12)
    # dump one transformer row: reconstructed vs stored
    if p == probe[0]:
        print("sample xfmr row 0:")
        print(f"  recon = {[round(x,4) for x in Rr[0].tolist()]}")
        print(f"  store = {[round(x,4) for x in Tr[0].tolist()]}")

print("\n=== reconstruct_full breakdown (zero-init) ===")
for k, (nu, de) in agg.items():
    print(f"  transformer-{k:5s} WAPE = {nu/de:.3e}" if k != "line" else f"  {k:17s} WAPE = {nu/de:.3e}")
