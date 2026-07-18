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
    # random hides Icomp (estimate-scatter live, the honest e2e); pf/se leave
    # vis_ic all-True, so their solve runs at truth Icomp = the zero-learned-params
    # machine-precision claim, measured through the same pipeline.
    ap.add_argument("--tasks", nargs="+", default=["random", "pf"])
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
    for task in a.tasks:
        print(f"\n=== lens: {task} ===")
        print(f"{'feeder':44s} {'n':>6s} {'skill_head':>10s} {'skill_solve':>11s} "
              f"{'skill_joint':>11s} {'hid_ic%':>8s}")
        run_lens(a, args, model, unseen, task)


def run_lens(a, args, model, unseen, task):
    for fdir in unseen[: a.n_feeders * 3]:
        try:
            fd = DKFeeder(fdir)
        except UnsupportedNetwork as e:
            print(f"{os.path.basename(fdir)[:44]:44s} SKIP {e}")
            continue
        ds = DKDataset([fd], [a.variant], task=task,
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
        hid_slot_nodes = []   # one entry per hidden ic slot: the node it injects at
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
                            hid_slot_nodes.append(node)
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
        # JOINT linear solve: the plain solve above IGNORES the mask's visible
        # interior V measurements -- but those are exactly what pins hidden Icomp
        # (the identifiability structure). Unknowns [V_hidden, dIc_hidden] around
        # the model estimate; equations = KCL at every node except ground and
        # vsource buses (their source current is not modeled). Determinate samples
        # solve exactly; underdetermined corners get the min-norm correction, i.e.
        # "project the model estimate onto the KCL-consistent manifold".
        skill_joint = float("nan")
        if n <= 4000:  # dense complex lstsq (SVD); bigger needs sparse lsqr
            vis_np = nd.vis_v.numpy()
            rows_ok = np.ones(n, dtype=bool); rows_ok[0] = False; rows_ok[fix] = False
            hidnodes = np.where(~vis_np)[0]; visnodes = np.where(vis_np)[0]
            A1 = Ybus[np.ix_(rows_ok.nonzero()[0], hidnodes)]
            E = np.zeros((int(rows_ok.sum()), len(hid_slot_nodes)), dtype=np.complex128)
            rowidx = -np.ones(n, dtype=int)
            rowidx[rows_ok] = np.arange(int(rows_ok.sum()))
            for j, na in enumerate(hid_slot_nodes):
                if rowidx[na] >= 0:
                    E[rowidx[na], j] = -1.0
            bj = rhs[rows_ok] - Ybus[np.ix_(rows_ok.nonzero()[0], visnodes)] @ Vt[visnodes]
            # Two-stage min-||delta||: naive lstsq on [V_hid, delta] min-norms V TOO,
            # pulling V toward zero on underdetermined samples (measured: 11.9 vs
            # plain solve 0.044). Instead: V is determined by KCL for ANY delta via
            # A1's pseudoinverse, so pick delta minimizing the projected residual
            # (ties -> min ||delta||, i.e. stay at the model estimate; delta=0
            # recovers the plain solve, so joint can never be worse).
            A1p = np.linalg.pinv(A1)
            PE = E - A1 @ (A1p @ E)
            Pb = bj - A1 @ (A1p @ bj)
            # rcond truncation: stiff-Y families (reactor/xfmr) give PE a broad
            # singular spectrum; unregularized lstsq pours huge delta into
            # near-nullspace directions for negligible residual gain (measured:
            # reactor 0.46 -> 981). Directions the physics cannot resolve keep
            # the model estimate (delta=0) instead of fitting fp64 noise.
            if E.shape[1]:
                delta, *_ = np.linalg.lstsq(PE, Pb, rcond=1e-8)
                if np.abs(E @ delta).sum() > 10 * np.abs(bj).sum():
                    delta = np.zeros(E.shape[1], dtype=np.complex128)  # unstable -> plain
                vh = A1p @ (bj - E @ delta)
            else:
                vh = A1p @ bj
            Vj = Vt.copy(); Vj[hidnodes] = vh
            skill_joint = float(np.abs(Vj[free] - Vt[free]).sum() / dvn)
        hid_pct = []
        for s in aux.get("ic_msk", {}):
            m = aux["ic_msk"][s]
            hid_pct.append(float(m.float().mean()) * 100)
        print(f"{os.path.basename(fdir)[:44]:44s} {n:6d} {skill_head:10.3f} "
              f"{skill_solve:11.3f} {skill_joint:11.2e} "
              f"{np.mean(hid_pct) if hid_pct else 0:8.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
