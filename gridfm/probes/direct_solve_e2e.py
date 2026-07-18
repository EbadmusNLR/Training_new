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
from gridfm.dk_physics import STORES, FC, node_count, terminal_slot
from gridfm.dk_data import (DKFeeder, DKDataset, make_dk_collate, discover_feeders,
                            split_feeders)
from gridfm.dk_tree import UnsupportedNetwork
from gridfm.dk_model import DKSolver
from gridfm.tests.test_ladder import build_ybus, SHUNT

ROOTS = ["/kfs2/projects/gogpt/Ebadmus/training_data/" + c for c in
         ("SMART-DS_1000", "new_dss_data", "dss_data", "minimal_component")]

# Visible-V sweep lenses (identifiability falsification): fixed visible-V fraction
# with 50% of PC Icomp hidden. Expectation: machine precision must DEGRADE as
# visible V drops and nullity rises -- if it does not, something is leaking.
from gridfm.dk_data import TASKS, _set_comp_masks, PC_STORES


def _sweep_mask(pv, pic):
    def m(data, rng):
        nd = data["node"]
        meas = torch.from_numpy(rng.random(int(nd.num_nodes)) < pv)
        nd.vis_v = nd.slack | nd.ground | meas
        nd.msk_v = ~nd.vis_v
        _set_comp_masks(data)
        for s in PC_STORES:
            if s not in data.node_types or s not in STORES:
                continue
            st = data[s]
            st.vis_ic = torch.from_numpy(rng.random(st.yr.shape[0]) < pic)
        return data
    return m


for _pv in (0, 20, 50, 80):
    TASKS[f"sw{_pv}"] = _sweep_mask(_pv / 100.0, 0.5)


# Contiguous-region lens (taxonomy T5, mission: "whole regions"): BFS a connected
# subgraph of ~frac of the nodes, hide its V AND the Icomp of every PC component
# attached inside it. Everything outside is visible. This is the "a whole
# neighborhood went dark" conditional -- hidden unknowns are spatially clustered,
# so identifiability comes only from the boundary, unlike iid random masks.
def _region_mask(frac):
    def m(data, rng):
        nd = data["node"]
        n = int(nd.num_nodes)
        adj = [[] for _ in range(n)]
        for s, (_, nterm, _) in STORES.items():
            if nterm < 2 or s not in data.node_types:
                continue
            per = {}
            for t in range(1, nterm + 1):
                rel = (s, f"bus{t}", "node")
                if rel not in data.edge_types or not data[rel].edge_index.numel():
                    continue
                ei = data[rel].edge_index
                for c, node in zip(ei[0].tolist(), ei[1].tolist()):
                    per.setdefault(c, []).append(node)
            for nodes in per.values():
                for u, v in zip(nodes, nodes[1:]):
                    adj[u].append(v); adj[v].append(u)
        slack = nd.slack.numpy(); ground = nd.ground.numpy()
        cand = np.where(~slack & ~ground)[0]
        seed = int(cand[rng.integers(len(cand))])
        target = max(2, int(frac * n))
        seen = {seed}; frontier = [seed]
        while frontier and len(seen) < target:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in seen and not ground[v] and not slack[v]:
                        seen.add(v); nxt.append(v)
            frontier = nxt
        reg = torch.zeros(n, dtype=torch.bool)
        reg[list(seen)] = True
        nd.vis_v = ~reg | nd.slack | nd.ground
        nd.msk_v = ~nd.vis_v
        _set_comp_masks(data)
        regn = reg.numpy()
        for s in PC_STORES:
            if s not in data.node_types or s not in STORES:
                continue
            st = data[s]
            nc = st.yr.shape[0]
            rel = (s, "bus1", "node")
            hid = np.zeros(nc, dtype=bool)
            if nc and rel in data.edge_types and data[rel].edge_index.numel():
                ei = data[rel].edge_index.numpy()
                np.logical_or.at(hid, ei[0], regn[ei[1]])
            st.vis_ic = torch.from_numpy(~hid)
        return data
    return m


for _fr in (10, 20, 40):
    TASKS[f"region{_fr}"] = _region_mask(_fr / 100.0)


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
    model.load_state_dict(ck["model"]); model.eval(); model.skip_current = True
    # same unseen split the trainer uses (pinned seed 42, per corpus, interleaved)
    from itertools import zip_longest
    per = [split_feeders(discover_feeders(r), seed=42) for r in ROOTS]
    unseen = [d for tup in zip_longest(*[c["unseen"] for c in per]) for d in tup if d]
    for task in a.tasks:
        print(f"\n=== lens: {task} ===")
        print(f"{'feeder':44s} {'n':>6s} {'skill_head':>10s} {'skill_solve':>11s} "
              f"{'skill_joint':>11s} {'joint0':>11s} {'nullity':>7s} {'v%':>5s} {'hid_ic%':>8s}")
        rows = run_lens(a, args, model, unseen, task)
        if rows:
            hd, so, jo, j0 = (np.array([r[k] for r in rows]) for k in (0, 1, 2, 3))
            nul = np.array([r[4] for r in rows])
            ok = ~np.isnan(jo)
            jo, j0k, nulk = jo[ok], j0[ok], nul[ok]
            if jo.size:
                det = nulk == 0
                print(f"--- {task} over {len(rows)} feeders: "
                      f"head med {np.median(hd):.3f} | solve med/max {np.median(so):.2e}/{so.max():.2e} | "
                      f"joint med/max {np.median(jo):.2e}/{jo.max():.2e} | "
                      f"joint0 med/max {np.median(j0k):.2e}/{j0k.max():.2e}")
                print(f"--- {task} identifiability: {int(det.sum())}/{det.size} nullity-0 "
                      f"(joint med {np.median(jo[det]):.2e})" if det.any() else
                      f"--- {task}: no nullity-0 samples", flush=True)
                if (~det).any():
                    print(f"--- {task} nullity>0 ({int((~det).sum())}): joint med "
                          f"{np.median(jo[~det]):.2e} vs joint0 med {np.median(j0k[~det]):.2e} "
                          f"(gap = model value)", flush=True)


def run_lens(a, args, model, unseen, task):
    rows = []
    for fdir in unseen[: a.n_feeders * 3]:
        try:
            fd = DKFeeder(fdir, need_decoder=False)  # mesh admitted; solve is topology-agnostic
        except UnsupportedNetwork as e:
            print(f"{os.path.basename(fdir)[:44]:44s} SKIP {e}")
            continue
        ds = DKDataset([fd], [a.variant], task=task,
                       use_feat=not args.get("no_feat", False))
        item = ds[0]
        batch, plan, rctx = make_dk_collate([fd], need_ctx=False)([item])
        batch.tree_plan = plan; batch.recon_ctx = rctx
        with torch.no_grad():
            dv, cur, aux = model(batch)
        nd = batch["node"]
        d = FeederScenarios(fdir)[a.variant]
        n = node_count(d)
        Ybus, _ = build_ybus(d, n)   # Y assembly only; rhs is rebuilt leak-proof below
        # LEAKAGE-PROOF rhs: built ONLY from visible-truth Icomp and model estimates.
        # The old form (truth_rhs + scatter(est - truth)) silently kept TRUTH at any
        # hidden slot the scatter loop missed -- the "one indexing error looks
        # magical" failure mode. Hidden truth is NaN-poisoned BEFORE estimates are
        # written over it: any hidden entry that reaches the system NaNs the solve
        # and trips the assert instead of silently scoring exact.
        hid_slot_nodes = []       # node of each hidden ic slot (E-matrix columns)
        rhs = np.zeros(n, dtype=np.complex128)
        est_part = np.zeros(n, dtype=np.complex128)   # estimate contribution alone
        for s in STORES:
            if s not in d.node_types or "Icomp_r_pu" not in d[s]:
                continue
            _, nterm, _ = STORES[s]
            dsst = d[s]
            ncomp = dsst["Icomp_r_pu"].shape[0]
            if not ncomp:
                continue
            ic = (dsst["Icomp_r_pu"].reshape(ncomp, -1).double().numpy()
                  + 1j * dsst["Icomp_i_pu"].reshape(ncomp, -1).double().numpy()).copy()
            # hiddenness comes from the EVALUATION mask on the batch, not from the
            # model's aux: if the model ever skips a hidden store, aux would say
            # nothing and truth would silently flow in. This way that path NaNs.
            hid = None
            if s in batch.node_types and hasattr(batch[s], "vis_ic"):
                hid_t = ~batch[s].vis_ic
                if bool(hid_t.any()):
                    hid = hid_t.numpy()
                    ic[hid, :] = np.nan                   # sentinel: truth is gone
                    if s in aux.get("ic_est", {}):
                        er, ei_ = aux["ic_est"][s]
                        est = er.double().numpy() + 1j * ei_.double().numpy()
                        w = min(est.shape[1], ic.shape[1])
                        ic[np.ix_(np.where(hid)[0], np.arange(w))] = est[hid, :w]
                        if ic.shape[1] > w:
                            ic[np.ix_(np.where(hid)[0], np.arange(w, ic.shape[1]))] = 0.0
                    else:
                        ic[hid, :] = 0.0   # hidden, no model estimate: zero prior
            for t in range(1, nterm + 1):
                rel = (s, f"bus{t}", "node")
                if rel not in d.edge_types or not d[rel].edge_index.numel():
                    continue
                eidx = d[rel].edge_index
                kk = terminal_slot(eidx[0])
                for c, k, node in zip(eidx[0].tolist(), kk.tolist(), eidx[1].tolist()):
                    col = (t - 1) * FC + int(k)
                    if col < ic.shape[1]:
                        rhs[node] += ic[c, col]
                        if hid is not None and hid[c]:
                            est_part[node] += ic[c, col]
                            hid_slot_nodes.append(node)
        assert not np.isnan(rhs).any(), "hidden Icomp truth leaked into rhs (sentinel)"
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
        skill_joint = skill_joint0 = float("nan"); nullity = -1
        pv_pct = 100.0 * float(nd.vis_v.float().mean())
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
            # identifiability audit: nullity of the augmented system. Expectation:
            # nullity 0 -> machine precision by algebra alone; nullity > 0 -> the
            # answer depends on the prior (this is the model's estate).
            nunk = A1.shape[1] + E.shape[1]
            nullity = int(nunk - np.linalg.matrix_rank(
                np.concatenate([A1, E], axis=1), tol=1e-8 * max(1.0, float(np.abs(Ybus).max()))))
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
            def joint_v(bvec):
                if E.shape[1]:
                    dl, *_ = np.linalg.lstsq(PE, bvec - A1 @ (A1p @ bvec), rcond=1e-8)
                    if np.abs(E @ dl).sum() > 10 * np.abs(bvec).sum():
                        dl = np.zeros(E.shape[1], dtype=np.complex128)  # unstable -> plain
                    return A1p @ (bvec - E @ dl)
                return A1p @ bvec

            Vj = Vt.copy(); Vj[hidnodes] = joint_v(bj)
            skill_joint = float(np.abs(Vj[free] - Vt[free]).sum() / dvn)
            # zero-prior baseline: same projection with the model estimate REMOVED
            # (hidden slots = 0). Determinate masks should not care; any gap between
            # joint and joint0 is the learned model's measured value.
            bj0 = bj - est_part[rows_ok]
            V0 = Vt.copy(); V0[hidnodes] = joint_v(bj0)
            skill_joint0 = float(np.abs(V0[free] - Vt[free]).sum() / dvn)
        hid_pct = []
        for s in aux.get("ic_msk", {}):
            m = aux["ic_msk"][s]
            hid_pct.append(float(m.float().mean()) * 100)
        print(f"{os.path.basename(fdir)[:44]:44s} {n:6d} {skill_head:10.3f} "
              f"{skill_solve:11.3f} {skill_joint:11.2e} {skill_joint0:11.2e} "
              f"{nullity:7d} {pv_pct:5.0f} "
              f"{np.mean(hid_pct) if hid_pct else 0:8.1f}", flush=True)
        rows.append((skill_head, skill_solve, skill_joint, skill_joint0, nullity))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
