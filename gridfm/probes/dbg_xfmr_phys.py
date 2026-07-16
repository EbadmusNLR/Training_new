"""Does the stored transformer current equal Yxfmr @ V at TRUTH V?

A transformer is linear and carries no Icomp, so I = Y@V must hold EXACTLY (~1e-14).
Every constitutive row in build_xfmr_system is nᵀI = (Yn)ᵀV -- derived from that identity.
If it does not hold, the system can be perfectly conditioned and still solve the WRONG
equations, which is exactly what trans_3w_center_tap looks like (fully determined,
cond 1.5e+01, transformer WAPE 8.3e-01).

Checks per transformer: |Y@V - I_stored| / |I_stored|, and which terminal is off.
"""
import glob, os, sys
import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios
from gridfm.dk_physics import FC, stored_currents, terminal_slot

CORPUS = os.environ.get("CORPUS", "dss_data")
TGT = os.environ.get("TGT", "trans_3w_center_tap")
TD = f"/kfs2/projects/gogpt/Ebadmus/training_data/{CORPUS}"
p = [x for x in sorted(glob.glob(os.path.join(TD, "*", "static.pt")))
     if TGT in os.path.basename(os.path.dirname(x))][0]
fs = FeederScenarios(os.path.dirname(p))
d = fs[0]
print("feeder:", os.path.basename(os.path.dirname(p)))

st = d["transformer"]
n = st["Yxfmr_r_pu"].shape[0]
Y = (st["Yxfmr_r_pu"].reshape(-1, 3 * FC, 3 * FC).double()
     + 1j * st["Yxfmr_i_pu"].reshape(-1, 3 * FC, 3 * FC).double())
Ir, Ii = stored_currents(d, "transformer", dtype=torch.float64)
I = Ir.double() + 1j * Ii.double()
vr, vi = d["node"].V_r_pu.double(), d["node"].V_i_pu.double()
V = vr + 1j * vi

# node behind every (comp, slot); slot = (terminal-1)*FC + phase
slot_node = {}
for t in (1, 2, 3):
    rel = ("transformer", f"bus{t}", "node")
    if rel not in d.edge_types or not d[rel].edge_index.numel():
        continue
    ei = d[rel].edge_index
    k = terminal_slot(ei[0])
    for c, kk, nd in zip(ei[0].tolist(), k.tolist(), ei[1].tolist()):
        slot_node[(int(c), (t - 1) * FC + int(kk))] = int(nd)

print(f"transformers: {Y.shape[0]}   Y block: {Y.shape[1]}x{Y.shape[2]}")
for c in range(Y.shape[0]):
    Vv = torch.zeros(3 * FC, dtype=torch.complex128)
    have = []
    for s in range(3 * FC):
        nd = slot_node.get((c, s))
        if nd is not None:
            Vv[s] = V[nd]          # V[0] == 0 (ground)
            have.append(s)
    pred = Y[c] @ Vv
    act = torch.tensor([s for s in range(3 * FC)
                        if abs(Y[c, s, s]) > 1e-9 * abs(torch.diag(Y[c])).max()])
    num = (pred[act] - I[c, act]).abs().sum()
    den = I[c, act].abs().sum() + 1e-30
    print(f"\n  xfmr {c}: active slots {act.tolist()}")
    print(f"    mapped slots {have}")
    print(f"    |Y@V - I|/|I| = {float(num/den):.3e}")
    for s in act.tolist():
        nd = slot_node.get((c, s), None)
        print(f"      slot {s:2d} (term {s//FC+1} ph {s%FC}) node={nd} "
              f"I_stored={complex(I[c,s]):+.6f}  Y@V={complex(pred[s]):+.6f}")
