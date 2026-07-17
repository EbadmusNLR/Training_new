#!/usr/bin/env python3
"""Train the iterative-solver GridFM on the datakit full-matrix corpus."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridfm.dk_data import (DKFeeder, DKDataset, discover_feeders, split_feeders,
                            fit_scales, feat, make_dk_collate)
from gridfm.dk_model import DKSolver
from gridfm.dk_physics import STORES, FC, terminal_slot, node_count

EPS = 1e-12


def kcl_of(batch, cur):
    n = node_count(batch); dev = batch["node"].V_r_init_pu.device
    rr = torch.zeros(n, device=dev); ri = torch.zeros(n, device=dev)
    for s, (ir, ii) in cur.items():
        _, nterm, _ = STORES[s]
        for t in range(1, nterm + 1):
            rel = (s, f"bus{t}", "node")
            if rel not in batch.edge_types or batch[rel].edge_index.numel() == 0:
                continue
            ei = batch[rel].edge_index
            comp, node = ei[0], ei[1]
            col = (t - 1) * FC + terminal_slot(comp)
            rr.index_add_(0, node, ir[comp, col]); ri.index_add_(0, node, ii[comp, col])
    res = torch.stack([rr, ri], 1); res[0] = 0.0
    return res


def losses(batch, dv, cur, scales, use_feat=True, w_v=10.0, w_i=1.0, w_kcl=0.1,
           norm=False):
    nd = batch["node"]
    msk = nd.msk_v
    # voltage: MSE on dv (small) + report WAPE
    verr = (dv - nd.dv)
    v_mse = (verr[msk] ** 2).mean() if msk.any() else verr.new_zeros(())
    vt = (nd.v_init + nd.dv)[msk].norm(dim=1).sum()
    v_wape = 100.0 * verr[msk].abs().sum() / (vt + EPS) if msk.any() else torch.zeros((), device=dv.device)
    # v_wape divides by |V| ~ 1.0 pu, but the SIGNAL is dv (|dv| ~ 0.04 pu on
    # SMART-DS). So "predict dv=0", which learns nothing, already scores v_wape
    # 4.4% there -- the metric flatters a null model into looking 96% accurate.
    # v_skill is the honest number: error / (error of dv=0). 1.0 = no skill.
    dvn = nd.dv[msk].abs().sum()
    v_skill = verr[msk].abs().sum() / (dvn + EPS) if msk.any() else torch.zeros((), device=dv.device)
    v_base = 100.0 * dvn / (vt + EPS) if msk.any() else torch.zeros((), device=dv.device)
    # currents: fitted per-family feature MSE + report pu WAPE (aggregate + per family)
    i_mse = dv.new_zeros(()); inum = dv.new_zeros(()); iden = dv.new_zeros(())
    fam = {}; nfam = 0
    for s, (ir, ii) in cur.items():
        st = batch[s]; sc = scales["I"][s]
        fr_p, fi_p = feat(ir, sc, use_feat), feat(ii, sc, use_feat)
        fr_t, fi_t = feat(st.ir, sc, use_feat), feat(st.ii, sc, use_feat)
        num_s = ((fr_p - fr_t) ** 2).mean() + ((fi_p - fi_t) ** 2).mean()
        if norm:
            # scale-free: divide by the family's own target power, so each term is
            # "fraction of variance unexplained" (1.0 = predicting zero) and the
            # weights actually mean something across families
            den_s = (fr_t ** 2).mean() + (fi_t ** 2).mean() + EPS
            num_s = num_s / den_s
        i_mse = i_mse + num_s
        nfam += 1
        fnum = (ir - st.ir).abs().sum() + (ii - st.ii).abs().sum()
        fden = st.ir.abs().sum() + st.ii.abs().sum()
        inum = inum + fnum; iden = iden + fden
        fam[f"i_{s}"] = float(100.0 * fnum / (fden + EPS))
    i_wape = 100.0 * inum / (iden + EPS)
    res = kcl_of(batch, cur)
    kcl = torch.asinh(res / scales["kcl"]).abs().mean()
    # The V term was ~100-800x smaller than the current term (w_v*v_mse ~ 8e-3 vs
    # i_mse ~ O(1-7)), so the model largely ignored V -- measured: v_skill ~1.0 with
    # the mixed loss vs 0.37 with V-only. `norm` makes every term "fraction of
    # variance unexplained" (1.0 = predict zero) so the weights are comparable.
    v_term = v_mse / ((nd.dv[msk] ** 2).mean() + EPS) if (norm and msk.any()) else v_mse
    # MEAN over families, not sum: i_mse summed ~7 normalised family terms, so it
    # still carried ~7x the weight of the single V term even after normalisation
    # (measured: mc norm-mixed 0.685 vs mc V-only 0.443 at the same epoch).
    i_term = i_mse / max(nfam, 1) if norm else i_mse
    loss = w_v * v_term + w_i * i_term + w_kcl * kcl
    m = {"v_wape": float(v_wape), "i_wape": float(i_wape), "v_skill": float(v_skill),
         "v_base": float(v_base), "v_mse": float(v_mse), "i_mse": float(i_mse),
         "kcl": float(kcl)}
    m.update(fam)
    return loss, m


def build_split(feeder_dirs, variants, task, use_feat, limit=None):
    if limit:
        feeder_dirs = feeder_dirs[:limit]
    feeders = [DKFeeder(d) for d in feeder_dirs]
    return DKDataset(feeders, variants, task=task, use_feat=use_feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kfs2/projects/gogpt/Ebadmus/training_data/SMART-DS_1000")
    ap.add_argument("--limit-feeders", type=int, default=None)
    ap.add_argument("--train-variants", type=int, default=80)
    ap.add_argument("--eval-variants", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--samples-per-epoch", type=int, default=4000)
    ap.add_argument("--no-feat", action="store_true")
    ap.add_argument("--no-kcl", action="store_true")
    ap.add_argument("--w-v", type=float, default=10.0)
    ap.add_argument("--w-i", type=float, default=1.0)
    ap.add_argument("--w-kcl", type=float, default=0.1)
    ap.add_argument("--norm-loss", action="store_true",
                    help="scale-free loss terms (fraction of variance unexplained)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "runs" / "dk_pf"))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_feat = not args.no_feat
    t0 = time.time()
    feeders = discover_feeders(args.root)
    sp = split_feeders(feeders, seed=42)
    lim = args.limit_feeders
    tr_dirs = sp["train"][:lim] if lim else sp["train"]
    un_dirs = sp["unseen"][:max(2, (lim // 8) if lim else len(sp["unseen"]))]
    tv = list(range(args.train_variants)); ev = list(range(args.train_variants, args.train_variants + args.eval_variants))
    train_ds = build_split(tr_dirs, tv, "pf", use_feat)
    unseen_ds = build_split(un_dirs, ev, "pf", use_feat)
    print(f"feeders train={len(tr_dirs)} unseen={len(un_dirs)}; "
          f"train_samples={len(train_ds)} unseen={len(unseen_ds)}; build={time.time()-t0:.1f}s", flush=True)

    # fit the ONE global per-family scaler on the train split (on-demand, cached)
    tf = time.time()
    scales = fit_scales(train_ds.feeders, tv)
    print(f"fitted scales in {time.time()-tf:.1f}s | kcl={scales['kcl']:.3e} | "
          + " ".join(f"{s}:I={scales['I'][s]:.2e}" for s in STORES if scales['I'][s] > 1e-8), flush=True)

    model = DKSolver(hidden=args.hidden, steps=args.steps,
                     kcl_feedback=not args.no_kcl, use_feat=use_feat, scales=scales).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    spe = min(args.samples_per_epoch, len(train_ds))
    sampler = torch.utils.data.RandomSampler(train_ds, num_samples=spe)
    tr_collate = make_dk_collate(train_ds.feeders)
    un_collate = make_dk_collate(unseen_ds.feeders)
    # persistent_workers: without it the pool is re-forked EVERY epoch, and each worker
    # re-imports torch/PyG (~60s of the epoch). prefetch keeps the GPU fed while a worker
    # builds the next batch's per-variant recon ctx.
    dl_kw = dict(num_workers=args.workers,
                 multiprocessing_context="fork" if args.workers else None)
    if args.workers:
        dl_kw.update(persistent_workers=True, prefetch_factor=4)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          collate_fn=tr_collate, **dl_kw)
    unseen_dl = DataLoader(unseen_ds, batch_size=args.batch_size, collate_fn=un_collate, **dl_kw)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train(); agg = {}
        te = time.time()
        for batch, plan, rctx in train_dl:
            batch = batch.to(dev); batch.tree_plan = plan; batch.recon_ctx = rctx
            dv, cur = model(batch)
            loss, m = losses(batch, dv, cur, scales, use_feat, w_v=args.w_v, w_i=args.w_i,
                             w_kcl=0.0 if args.no_kcl else args.w_kcl, norm=args.norm_loss)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        n = max(1, len(train_dl))
        tr = {k: v / n for k, v in agg.items()}
        # eval
        model.eval(); ea = {}
        with torch.no_grad():
            for batch, plan, rctx in unseen_dl:
                batch = batch.to(dev); batch.tree_plan = plan; batch.recon_ctx = rctx
                dv, cur = model(batch)
                _, m = losses(batch, dv, cur, scales, use_feat)
                for k, v in m.items():
                    ea[k] = ea.get(k, 0.0) + v
        ne = max(1, len(unseen_dl)); un = {k: v / ne for k, v in ea.items()}
        fam_str = " ".join(f"{k[2:]}={un[k]:.1f}" for k in sorted(un) if k.startswith("i_"))
        print(f"ep{epoch:03d} {time.time()-te:.0f}s | train V/I={tr['v_wape']:.3f}%/{tr['i_wape']:.3f}% "
              f"kcl={tr['kcl']:.3e} | unseen V/I={un['v_wape']:.3f}%/{un['i_wape']:.3f}%\n"
              f"        V skill: train={tr['v_skill']:.3f} unseen={un['v_skill']:.3f} "
              f"(1.000 = no better than dv=0; dv=0 scores v_wape {un['v_base']:.2f}%)\n"
              f"        unseen I/fam: {fam_str}", flush=True)
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "scales": scales},
                   Path(args.out) / "last.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
