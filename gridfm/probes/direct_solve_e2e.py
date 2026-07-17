"""E2E: does (model Icomp estimates -> ONE direct fp64 solve) beat the model's V head?

T15 v3 proved solve(Ybus_ff, rhs) is machine precision at TRUTH Icomp on 20/20
feeders. This measures the same pipeline at ESTIMATED Icomp: hidden entries from
the trained ic_head (se-lens masks), visible entries truth. Compare, per feeder:

    v_skill_head   -- the checkpoint's dv prediction (the 0.789-class number)
    v_skill_solve  -- direct solve with mixed truth/estimated Icomp
    (both = |err| / |dv|; 1.0 = predicting dv=0)

If solve << head, the GNN's V head is dead weight: train the estimator, decode V
with physics. Run on a compute node (dense fp64 solves).

Usage: direct_solve_e2e.py --ckpt runs/fix_lr2e4/last.pt --n-feeders 8
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new/scripts")
from core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES, FC, node_count
from gridfm.dk_data import (DKFeeder, DKDataset, make_dk_collate, discover_feeders,
                            split_feeders)
from gridfm.dk_tree import UnsupportedNetwork
from gridfm.dk_model import DKSolver
from gridfm.tests.test_ladder import build_ybus, SHUNT

ROOTS = ["/kfs2/projects/gogpt/Ebadmus/training_data/" + c for c in
         ("SMART-DS_1000", "new_dss_data", "dss_data", "minimal_component")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-feeders", type=int, default=8)
    ap.add_argument("--variant", type=int, default=90)  # outside train variants
    # MUST be a mask that hides Icomp: se/pf leave vis_ic all-True, making the
    # estimate-scatter a no-op and skill_solve trivially machine-precision.
    ap.add_argument("--task", default="random")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = ck["args"]
    model = DKSolver(hidden=args["hidden"], steps=args["steps"],
                     kcl_feedback=not args.get("no_kcl", False),
                     use_feat=not args.get("no_feat", False), scales=ck["scales"],
                     fb_points=args.get("fb_points", 0), vabs=args.get("vabs", False))
    model.load_state_dict(ck["model"]); model.eval()
    # same unseen split the trainer uses (pinned seed 42, per corpus, interleaved)
    from itertools import zip_longest
    per = [split_feeders(discover_feeders(r), seed=42) for r in ROOTS]
    unseen = [d for tup in zip_longest(*[c["unseen"] for c in per]) for d in tup if d]
    print(f"{'feeder':44s} {'n':>6s} {'skill_head':>10s} {'skill_solve':>11s} {'hid_ic%':>8s}")
    for fdir in unseen[: a.n_feeders * 3]:
        try:
            fd = DKFeeder(fdir)
        except UnsupportedNetwork as e:
            print(f"{os.path.basename(fdir)[:44]:44s} SKIP {e}")
            continue
        ds = DKDataset([fd], [a.variant], task=a.task,
                       use_feat=not args.get("no_feat", False))
        item = ds[0]
        batch, plan, rctx = make_dk_collate([fd])([item])
        batch.tree_plan = plan; batch.recon_ctx = rctx
        with torch.no_grad():
            dv, cur, aux = model(batch)
        nd = batch["node"]
        d = FeederScenarios(fdir)[a.variant]
        n = node_count(d)
        Ybus, rhs = build_ybus(d, n)  # truth rhs; estimated entries replace below
        # scatter estimate into rhs: for each shunt store, hidden terminals get est
        for s in aux.get("ic_est", {}):
            er, ei = aux["ic_est"][s]
            st = batch[s]
            hid = aux["ic_msk"][s]  # [ncomp] bool
            if not bool(hid.any()):
                continue
            _, nterm, _ = STORES[s]
            dsst = d[s]
            ic_t = (dsst["Icomp_r_pu"].reshape(er.shape[0], -1).double().numpy()
                    + 1j * dsst["Icomp_i_pu"].reshape(er.shape[0], -1).double().numpy())
            ic_e = er.double().numpy() + 1j * ei.double().numpy()
            delta = ic_e - ic_t  # zero where estimate == truth
            for t in range(1, nterm + 1):
                rel = (s, f"bus{t}", "node")
                if rel not in d.edge_types or not d[rel].edge_index.numel():
                    continue
                eidx = d[rel].edge_index
                from gridfm.dk_physics import terminal_slot
                kk = terminal_slot(eidx[0])
                for c, k, node in zip(eidx[0].tolist(), kk.tolist(), eidx[1].tolist()):
                    if hid[c]:
                        col = (t - 1) * FC + int(k)
                        if col < delta.shape[1]:
                            rhs[node] += delta[c, col]
        Vt = d["node"].V_r_pu.double().numpy() + 1j * d["node"].V_i_pu.double().numpy()
        Vi = (d["node"].V_r_init_pu.double().numpy()
              + 1j * d["node"].V_i_init_pu.double().numpy())
        vis = np.zeros(n, dtype=bool); vis[0] = True
        rel = ("vsource", "bus1", "node")
        if rel in d.edge_types and d[rel].edge_index.numel():
            vis[d[rel].edge_index[1].numpy()] = True
        free = np.where(~vis)[0]; fix = np.where(vis)[0]
        b = rhs[free] - Ybus[np.ix_(free, fix)] @ Vt[fix]
        Vs = np.linalg.solve(Ybus[np.ix_(free, free)], b)
        dvn = np.abs(Vt[free] - Vi[free]).sum() + 1e-30
        skill_solve = np.abs(Vs - Vt[free]).sum() / dvn
        # head skill on the SAME sample/mask (hidden-V nodes only, matching trainer)
        msk = nd.msk_v
        verr = (dv - nd.dv)[msk]
        skill_head = float(verr.abs().sum() / (nd.dv[msk].abs().sum() + 1e-30))
        hid_pct = []
        for s in aux.get("ic_msk", {}):
            m = aux["ic_msk"][s]
            hid_pct.append(float(m.float().mean()) * 100)
        print(f"{os.path.basename(fdir)[:44]:44s} {n:6d} {skill_head:10.3f} "
              f"{skill_solve:11.3f} {np.mean(hid_pct) if hid_pct else 0:8.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
