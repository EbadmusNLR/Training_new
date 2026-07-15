"""Is the decoder failure VARIANT-dependent? build_recon_ctx is cached from
variant 0 -- valid only if Y/topology are static across variants."""
import glob, os, sys
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, store_size, stored_currents, element_currents
from gridfm.dk_tree import reconstruct_full, build_recon_ctx, SHUNT_STORES, AMBIG_STORES

TD = "/kfs2/projects/gogpt/Ebadmus/training_data/minimal_component"
TGT = os.environ.get("TGT", "t1_0208_k10_storage_de")
p = [x for x in sorted(glob.glob(os.path.join(TD, "*", "static.pt"))) if TGT in x][0]
fs = FeederScenarios(os.path.dirname(p))
print("feeder:", os.path.basename(os.path.dirname(p)), " variants:", len(fs))

d0 = fs[0]
for key in ("Yxfmr_r_pu", "Yxfmr_i_pu"):
    a = d0["transformer"][key]
    same = [v for v in range(1, min(20, len(fs)))
            if not torch.equal(fs[v]["transformer"][key], a)]
    print(f"  {key}: variants differing from v0: {same[:10]}")
ei0 = d0[("transformer", "bus1", "node")].edge_index
print("  edge_index static:", all(torch.equal(fs[v][("transformer","bus1","node")].edge_index, ei0)
                                  for v in range(1, min(20, len(fs)))))

ctx0 = build_recon_ctx(d0)
print("\n  var   cached-ctx    fresh-ctx")
for v in (0, 1, 2, 5, 17):
    d = fs[v]
    vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
    present = [s for s in STORES if s in d.node_types and store_size(d, s) > 0]
    truth = {s: stored_currents(d, s, dtype=torch.float64) for s in present}
    out = []
    for ctx in (ctx0, build_recon_ctx(d)):
        cur = {}
        for s in present:
            if s in SHUNT_STORES or s in AMBIG_STORES:
                cur[s] = element_currents(d, s, vr, vi)
            else:
                z = torch.zeros_like(truth[s][0]); cur[s] = (z, z.clone())
        rec = reconstruct_full(d, cur, vr, vi, ctx=ctx)
        n = sum(float((rec.get(s, cur[s])[0]-truth[s][0]).abs().sum()
                    + (rec.get(s, cur[s])[1]-truth[s][1]).abs().sum()) for s in present)
        dd = sum(float(truth[s][0].abs().sum()+truth[s][1].abs().sum()) for s in present)
        out.append(n/(dd+1e-30))
    print(f"  v{v:<4d} {out[0]:.3e}    {out[1]:.3e}")
