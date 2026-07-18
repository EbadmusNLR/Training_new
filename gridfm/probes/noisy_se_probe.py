"""Noisy-SE probe: does a WEIGHTED solve survive noisy/bad visible V where the
naive joint solve (visible V substituted as EXACT into KCL rows) breaks?

MISSION clause under test: "reconstruct under noisy/bad visible entries".
Setting = the T28b estate: random mask, hidden Icomp forces the solve to lean on
visible interior V. Corruption: Gaussian sigma*med|V| on visible INTERIOR V
(vsource/ground stay exact -- substation reference; source current is unmodeled,
so vsource V must be a hard constraint either way) plus a gross-error fraction
(bad sensors: 20%-magnitude random-phase offsets).

Per feeder, same leak-proof rhs as direct_solve_e2e (NaN sentinel), then:
  naive  -- current joint pipeline with noisy V substituted as exact (stiff Y
            rows amplify measurement noise into KCL; expected >> sigma-floor)
  wls    -- visible V kept as SOFT measurement rows; row-normalized KCL rows
            exact-weighted (wk); delta-Ic prior rows (wp) tie hidden slots to
            the model estimate
  robust -- wls + MAD-normalized residual test on measurement rows, drop
            flagged, re-solve (classic bad-data rejection; precision/recall
            vs the planted gross errors is reported)
  clean  -- wls at sigma=0 (sanity: must stay ~machine precision or the
            weighting itself is broken)

floor = sum|planted noise| / dvn: the score of PERFECT hidden reconstruction
with measurements taken at face value. wls near floor + naive far above it =
the weighted layer is REQUIRED for the noisy stage (the architecture review's
adoption trigger, measured). Run on a compute node (dense fp64 lstsq).

Usage: noisy_se_probe.py --ckpt runs/dk_foundation_15261953/last.pt \
         --n-feeders 16 --sigma 0.01 --gross-frac 0.03 --subset-seed 401
"""
import argparse, os, sys, zlib
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
from gridfm.tests.test_ladder import build_ybus

ROOTS = ["/kfs2/projects/gogpt/Ebadmus/training_data/" + c for c in
         ("SMART-DS_1000", "new_dss_data", "dss_data", "minimal_component")]


def build_rhs(d, batch, aux, n):
    """Leak-proof rhs (NaN sentinel) -- verbatim logic from direct_solve_e2e."""
    hid_slot_nodes = []
    rhs = np.zeros(n, dtype=np.complex128)
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
        hid = None
        if s in batch.node_types and hasattr(batch[s], "vis_ic"):
            hid_t = ~batch[s].vis_ic
            if bool(hid_t.any()):
                hid = hid_t.numpy()
                ic[hid, :] = np.nan
                if s in aux.get("ic_est", {}):
                    er, ei_ = aux["ic_est"][s]
                    est = er.double().numpy() + 1j * ei_.double().numpy()
                    w = min(est.shape[1], ic.shape[1])
                    ic[np.ix_(np.where(hid)[0], np.arange(w))] = est[hid, :w]
                    if ic.shape[1] > w:
                        ic[np.ix_(np.where(hid)[0], np.arange(w, ic.shape[1]))] = 0.0
                else:
                    ic[hid, :] = 0.0
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
                        hid_slot_nodes.append(node)
    assert not np.isnan(rhs).any(), "hidden Icomp truth leaked into rhs (sentinel)"
    return rhs, hid_slot_nodes


def kcl_blocks(Ybus, rhs, Vt, fix, rows_ok, cols, hid_slot_nodes):
    """[Y[rows_ok, cols] | E] z = rhs[rows_ok] - Y[rows_ok, fix] @ Vt[fix]."""
    rid = np.where(rows_ok)[0]
    M1 = Ybus[np.ix_(rid, cols)]
    E = np.zeros((rid.size, len(hid_slot_nodes)), dtype=np.complex128)
    rowidx = -np.ones(Ybus.shape[0], dtype=int)
    rowidx[rid] = np.arange(rid.size)
    for j, na in enumerate(hid_slot_nodes):
        if rowidx[na] >= 0:
            E[rowidx[na], j] = -1.0
    bk = rhs[rid] - Ybus[np.ix_(rid, fix)] @ Vt[fix]
    return M1, E, bk


def naive_joint(Ybus, rhs, Vt, Vm, fix, free, rows_ok, hidnodes, visnodes,
                hid_slot_nodes):
    """direct_solve_e2e's two-stage joint, with noisy Vm substituted as exact."""
    A1, E, _ = kcl_blocks(Ybus, rhs, Vt, fix, rows_ok, hidnodes, hid_slot_nodes)
    rid = np.where(rows_ok)[0]
    bj = rhs[rid] - Ybus[np.ix_(rid, visnodes)] @ Vm[visnodes]
    A1p = np.linalg.pinv(A1)
    if E.shape[1]:
        PE = E - A1 @ (A1p @ E)
        dl, *_ = np.linalg.lstsq(PE, bj - A1 @ (A1p @ bj), rcond=1e-8)
        if np.abs(E @ dl).sum() > 10 * np.abs(bj).sum():
            dl = np.zeros(E.shape[1], dtype=np.complex128)
        vh = A1p @ (bj - E @ dl)
    else:
        vh = A1p @ bj
    V = Vm.copy()          # visible nodes at face value (the naive posture)
    V[hidnodes] = vh
    V[fix] = Vt[fix]
    return V


def wls(Ybus, rhs, Vt, Vm, fix, free, rows_ok, visI, hid_slot_nodes,
        wk=1e6, wp=1e-3, drop=None):
    """Unknowns z=[V_free, delta]; KCL rows row-normalized then exact-weighted;
    visible interior V are unit-weight soft measurements; wp ties delta to 0
    (= stay at the model estimate) and de-duplicates repeated hidden slots."""
    nfree = free.size
    nd_ = len(hid_slot_nodes)
    pos = -np.ones(Ybus.shape[0], dtype=int)
    pos[free] = np.arange(nfree)
    M1, E, bk = kcl_blocks(Ybus, rhs, Vt, fix, rows_ok, free, hid_slot_nodes)
    K = np.concatenate([M1, E], axis=1)
    rn = np.linalg.norm(K, axis=1)
    rn[rn == 0] = 1.0
    K *= (wk / rn)[:, None]
    bk = bk * (wk / rn)
    mI = visI if drop is None else visI[~drop]
    Mm = np.zeros((mI.size, nfree + nd_), dtype=np.complex128)
    Mm[np.arange(mI.size), pos[mI]] = 1.0
    P = np.zeros((nd_, nfree + nd_), dtype=np.complex128)
    if nd_:
        P[np.arange(nd_), nfree + np.arange(nd_)] = wp
    A = np.concatenate([K, Mm, P], axis=0)
    b = np.concatenate([bk, Vm[mI], np.zeros(nd_, dtype=np.complex128)])
    z, *_ = np.linalg.lstsq(A, b, rcond=None)
    V = Vt.copy()
    V[free] = z[:nfree]
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-feeders", type=int, default=16)
    ap.add_argument("--variant", type=int, default=90)
    ap.add_argument("--task", default="random")
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--gross-frac", type=float, default=0.03)
    ap.add_argument("--subset-seed", type=int, default=401)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = ck["args"]
    model = DKSolver(hidden=args["hidden"], steps=args["steps"],
                     kcl_feedback=not args.get("no_kcl", False),
                     use_feat=not args.get("no_feat", False), scales=ck["scales"],
                     fb_points=args.get("fb_points", 0), vabs=args.get("vabs", False))
    model.load_state_dict(ck["model"]); model.eval(); model.skip_current = True
    # protocol: RANDOM feeder subset -- stratified per-corpus shuffle, re-interleave
    from itertools import zip_longest
    srng = np.random.default_rng(a.subset_seed)
    per = [split_feeders(discover_feeders(r), seed=42) for r in ROOTS]
    pools = [list(c["unseen"]) for c in per]
    for p in pools:
        srng.shuffle(p)
    unseen = [d for tup in zip_longest(*pools) for d in tup if d]

    print(f"=== noisy-se: task={a.task} sigma={a.sigma} gross={a.gross_frac} "
          f"subset-seed={a.subset_seed} ckpt={a.ckpt} ===")
    print(f"{'feeder':44s} {'n':>6s} {'floor':>9s} {'naive':>9s} {'wls':>9s} "
          f"{'robust':>9s} {'clean':>9s} {'flag P/R':>9s}")
    rows = []
    done = 0
    for fdir in unseen:
        if done >= a.n_feeders:
            break
        try:
            fd = DKFeeder(fdir, need_decoder=False)
        except UnsupportedNetwork as e:
            print(f"{os.path.basename(fdir)[:44]:44s} SKIP {e}")
            continue
        ds = DKDataset([fd], [a.variant], task=a.task,
                       use_feat=not args.get("no_feat", False))
        item = ds[0]
        batch, plan, rctx = make_dk_collate([fd], need_ctx=False)([item])
        batch.tree_plan = plan; batch.recon_ctx = rctx
        with torch.no_grad():
            dv, cur, aux = model(batch)
        nd = batch["node"]
        d = FeederScenarios(fdir)[a.variant]
        n = node_count(d)
        if n > 4000:
            print(f"{os.path.basename(fdir)[:44]:44s} SKIP n={n} (dense cap)")
            continue
        Ybus, _ = build_ybus(d, n)
        rhs, hid_slot_nodes = build_rhs(d, batch, aux, n)
        Vt = d["node"].V_r_pu.double().numpy() + 1j * d["node"].V_i_pu.double().numpy()
        Vi = (d["node"].V_r_init_pu.double().numpy()
              + 1j * d["node"].V_i_init_pu.double().numpy())
        vis_np = nd.vis_v.numpy()
        fixm = np.zeros(n, dtype=bool); fixm[0] = True
        rel = ("vsource", "bus1", "node")
        if rel in d.edge_types and d[rel].edge_index.numel():
            fixm[d[rel].edge_index[1].numpy()] = True
        fix = np.where(fixm)[0]; free = np.where(~fixm)[0]
        rows_ok = ~fixm.copy(); rows_ok[0] = False
        hidnodes = np.where(~vis_np & ~fixm)[0]
        visnodes = np.where(vis_np | fixm)[0]
        visI = np.where(vis_np & ~fixm)[0]          # interior measurements
        dvn = np.abs(Vt[free] - Vi[free]).sum() + 1e-30

        # plant corruption (per-feeder deterministic)
        frng = np.random.default_rng(
            [a.subset_seed, zlib.crc32(os.path.basename(fdir).encode())])
        medv = float(np.median(np.abs(Vt[free]))) or 1.0
        Vm = Vt.copy()
        noise = a.sigma * medv * (frng.standard_normal(visI.size)
                                  + 1j * frng.standard_normal(visI.size)) / np.sqrt(2)
        Vm[visI] += noise
        ngross = int(round(a.gross_frac * visI.size))
        gidx = frng.choice(visI.size, size=ngross, replace=False) if ngross else \
            np.array([], dtype=int)
        Vm[visI[gidx]] += 0.2 * medv * np.exp(2j * np.pi * frng.random(ngross))
        planted = np.zeros(visI.size, dtype=bool); planted[gidx] = True
        floor = float(np.abs(Vm[visI] - Vt[visI]).sum() / dvn)

        def skill(V):
            return float(np.abs(V[free] - Vt[free]).sum() / dvn)

        s_naive = skill(naive_joint(Ybus, rhs, Vt, Vm, fix, free, rows_ok,
                                    hidnodes, visnodes, hid_slot_nodes))
        Vw = wls(Ybus, rhs, Vt, Vm, fix, free, rows_ok, visI, hid_slot_nodes)
        s_wls = skill(Vw)
        # robust: MAD-normalized measurement residuals, drop > 4 sigma, re-solve
        r = np.abs(Vw[visI] - Vm[visI])
        s_ = 1.4826 * np.median(r) + 1e-30
        flag = r > 4.0 * s_
        s_rob = skill(wls(Ybus, rhs, Vt, Vm, fix, free, rows_ok, visI,
                          hid_slot_nodes, drop=flag)) if flag.any() else s_wls
        s_clean = skill(wls(Ybus, rhs, Vt, Vt, fix, free, rows_ok, visI,
                            hid_slot_nodes))
        tp = int((flag & planted).sum())
        prec = tp / max(int(flag.sum()), 1)
        rec = tp / max(int(planted.sum()), 1)
        print(f"{os.path.basename(fdir)[:44]:44s} {n:6d} {floor:9.2e} "
              f"{s_naive:9.2e} {s_wls:9.2e} {s_rob:9.2e} {s_clean:9.2e} "
              f"{prec:4.2f}/{rec:4.2f}", flush=True)
        rows.append((floor, s_naive, s_wls, s_rob, s_clean, prec, rec,
                     int(planted.sum())))
        done += 1

    if rows:
        fl, na, wl, ro, cl = (np.array([r[k] for r in rows]) for k in range(5))
        pr = np.array([r[5] for r in rows]); rc = np.array([r[6] for r in rows])
        haveg = np.array([r[7] for r in rows]) > 0
        print(f"--- noisy-se sigma={a.sigma} gross={a.gross_frac} over {len(rows)} "
              f"feeders: floor med {np.median(fl):.2e} | "
              f"naive med/max {np.median(na):.2e}/{na.max():.2e} | "
              f"wls med/max {np.median(wl):.2e}/{wl.max():.2e} | "
              f"robust med/max {np.median(ro):.2e}/{ro.max():.2e}")
        print(f"--- clean sanity: wls med/max {np.median(cl):.2e}/{cl.max():.2e} "
              f"(must be ~machine precision)")
        if haveg.any():
            print(f"--- bad-data detection over {int(haveg.sum())} feeders w/ planted "
                  f"gross: precision med {np.median(pr[haveg]):.2f} recall med "
                  f"{np.median(rc[haveg]):.2f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
