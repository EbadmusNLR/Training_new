#!/usr/bin/env python3
"""Train the iterative-solver GridFM on the datakit full-matrix corpus."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
# The batched recon ctx ships hundreds of small tensors per batch through worker
# pipes; with fork sharing each is an fd, and 16 workers x prefetch 4 exhausts the
# limit ("received 0 items of ancdata", killed T12-s1 after ep001). file_system
# sharing uses /dev/shm files instead of fds.
torch.multiprocessing.set_sharing_strategy("file_system")
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
           norm=False, aux=None, w_ic=1.0, w_y=1.0):
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
    # injection estimation: loss on the Icomp ESTIMATE at hidden components (feat
    # space, like currents). The estimate also drove the physics decode, so i_mse and
    # kcl already pull on it; this term is the direct supervision.
    ic_mse = dv.new_zeros(()); icn = dv.new_zeros(()); icd = dv.new_zeros(()); nic_t = 0
    if aux and aux.get("ic_est"):
        for s, (er, ei_) in aux["ic_est"].items():
            st = batch[s]; mm = aux["ic_msk"][s]
            if not bool(mm.any()):
                continue
            sc = scales["I"][s]
            tr_r, tr_i = st.icr, st.ici
            fr = (feat(er[mm], sc, use_feat) - feat(tr_r[mm], sc, use_feat)) ** 2
            fi = (feat(ei_[mm], sc, use_feat) - feat(tr_i[mm], sc, use_feat)) ** 2
            term = fr.mean() + fi.mean()
            if norm:
                term = term / ((feat(tr_r[mm], sc, use_feat) ** 2).mean()
                               + (feat(tr_i[mm], sc, use_feat) ** 2).mean() + EPS)
            ic_mse = ic_mse + term; nic_t += 1
            icn = icn + (er[mm] - tr_r[mm]).abs().sum() + (ei_[mm] - tr_i[mm]).abs().sum()
            icd = icd + tr_r[mm].abs().sum() + tr_i[mm].abs().sum()
    ic_term = ic_mse / max(nic_t, 1)
    # parameter estimation: loss on the Y ESTIMATE at hidden components (feature
    # space, per-family scales). Same pattern as ic_term; the general four-array
    # mask makes Y a first-class target.
    y_mse = dv.new_zeros(()); yn = dv.new_zeros(()); yd = dv.new_zeros(()); ny_t = 0
    if aux and aux.get("y_est"):
        for s, (eyr, eyi) in aux["y_est"].items():
            st = batch[s]; mm = aux["y_msk"][s]
            if not bool(mm.any()):
                continue
            # FEAT space, like the ic loss: pu-space Y spans ~12 orders, so a pu MSE
            # is owned by the stiffest entries and the sinh decode explodes its
            # gradients (measured: par y_wape 1730% -- WORSE than predicting zero).
            # feat(inv_feat(z)) == z, so compare the head's z to feat(truth) directly.
            tr_st = torch.stack([st.yr[mm], st.yi[mm]], -1)
            zt = feat(tr_st, aux["y_scale"][s], use_feat)
            ze = aux["y_feat"][s][mm]
            nz = (tr_st.abs() > 1e-12)                    # structural-zero labels
            # spike-and-slab: BCE trains the gate on the sparsity pattern; the
            # magnitude MSE only speaks where the truth is nonzero, so zero
            # positions cost gate->0 instead of sinh(z)*bigscale pu garbage.
            gl = aux["y_gate"][s][mm]
            bce = torch.nn.functional.binary_cross_entropy_with_logits(gl, nz.float())
            mag = ((ze - zt) ** 2)[nz].mean() if bool(nz.any()) else ze.new_zeros(())
            if norm and bool(nz.any()):
                mag = mag / ((zt ** 2)[nz].mean() + EPS)
            term = mag + bce
            y_mse = y_mse + term; ny_t += 1
            es = torch.stack([eyr[mm], eyi[mm]], -1)
            tr = torch.stack([st.yr[mm], st.yi[mm]], -1)
            yn = yn + (es - tr).abs().sum(); yd = yd + tr.abs().sum()
    y_term = y_mse / max(ny_t, 1)
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
    loss = w_v * v_term + w_i * i_term + w_kcl * kcl + w_ic * ic_term + w_y * y_term
    ic_wape = 100.0 * float(icn) / (float(icd) + 1e-30) if nic_t else 0.0
    y_wape = 100.0 * float(yn) / (float(yd) + 1e-30) if ny_t else 0.0
    m = {"ic_wape": ic_wape, "y_wape": y_wape,
         "v_wape": float(v_wape), "i_wape": float(i_wape), "v_skill": float(v_skill),
         "v_base": float(v_base), "v_mse": float(v_mse), "i_mse": float(i_mse),
         "kcl": float(kcl)}
    m.update(fam)
    return loss, m


def build_split(feeder_dirs, variants, task, use_feat, limit=None, role="train"):
    if limit:
        feeder_dirs = feeder_dirs[:limit]
    # The decoder REFUSES networks it cannot reconstruct (UnsupportedNetwork: e.g.
    # meshed pure-line loops leave transformer groups underdetermined -- IEEE 30 Bus
    # class). Refusing beats silently-zero currents, but one such feeder must not
    # abort training on thousands of good ones. So: skip LOUDLY -- every exclusion is
    # named with its reason at startup -- and hard-fail if exclusions exceed 5%, so a
    # decoder regression cannot quietly hollow out the corpus.
    from gridfm.dk_tree import UnsupportedNetwork
    feeders, skipped = [], []
    for d in feeder_dirs:
        try:
            feeders.append(DKFeeder(d))
        except UnsupportedNetwork as exc:
            skipped.append((os.path.basename(d), str(exc)[:140]))
    if skipped:
        print(f"EXCLUDED {len(skipped)}/{len(feeder_dirs)} feeders (decoder refuses; see reasons):", flush=True)
        for name, why in skipped:
            print(f"  - {name}: {why}", flush=True)
        # Hard gate on TRAIN only: that is corpus-rot protection. Eval splits are small
        # diagnostics (30 feeders), where a handful of meshed feeders trips 5% instantly
        # -- measured: train 2/240 (0.8%) vs unseen 4/30 (13%) from the same interleave,
        # because dss_data/new_dss_data vendor meshed examples. Eval exclusions stay
        # LOUD (named above) and the claim they qualify is recorded in the ledger:
        # "unseen" means unseen RADIAL until batched-bridge support lands.
        if role == "train" and len(skipped) > 0.05 * len(feeder_dirs):
            raise RuntimeError(f"{len(skipped)} of {len(feeder_dirs)} feeders excluded (>5%): "
                               "decoder coverage regressed; fix that before training")
    return DKDataset(feeders, variants, task=task, use_feat=use_feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "/kfs2/projects/gogpt/Ebadmus/training_data/dss_data",
        "/kfs2/projects/gogpt/Ebadmus/training_data/minimal_component",
        "/kfs2/projects/gogpt/Ebadmus/training_data/new_dss_data",
        "/kfs2/projects/gogpt/Ebadmus/training_data/SMART-DS_1000",
    ], help="corpora to train over; splits are stratified per corpus")
    ap.add_argument("--limit-feeders", type=int, default=None)
    ap.add_argument("--train-variants", type=int, default=80)
    ap.add_argument("--eval-variants", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--samples-per-epoch", type=int, default=4000)
    ap.add_argument("--no-feat", action="store_true",
                    help="linear feature normalization instead of asinh compression "
                         "(same fitted scales; the asinh-ablation flag)")
    ap.add_argument("--vabs", action="store_true",
                    help="node head predicts ABSOLUTE V (v_std gauge) instead of dv")
    ap.add_argument("--no-kcl", action="store_true")
    ap.add_argument("--eval-feeders", type=int, default=48,
                    help="cap on unseen eval feeders when --limit-feeders is unset. "
                         "The full corpus has 319 unseen feeders; evaluating all of "
                         "them x variants x 3 lenses on rank 0 alone blew NCCL's "
                         "10-min watchdog (job 15256412: rank0 stalled at allreduce "
                         "5006 while ranks 1-3 entered the next epoch)")
    ap.add_argument("--w-v", type=float, default=10.0)
    ap.add_argument("--w-i", type=float, default=1.0)
    ap.add_argument("--w-kcl", type=float, default=0.1)
    ap.add_argument("--w-ic", type=float, default=1.0,
                    help="hidden-Icomp estimate loss. T17 measured estimate->solve V at "
                         "~100x the V head, so this is the loss that buys V accuracy")
    ap.add_argument("--w-y", type=float, default=1.0,
                    help="hidden-Y estimate loss (four-array mask; T22: excitation-"
                         "limited, so this trains the structural prior)")
    ap.add_argument("--norm-loss", action="store_true",
                    help="scale-free loss terms (fraction of variance unexplained)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--task", default="pf",
                    choices=("pf", "se", "injection", "random_safe", "random", "random4", "param"),
                    help="training objective. Eval stays pf so runs are comparable.")
    ap.add_argument("--small-first", action="store_true",
                    help="order each split by static.pt size ascending before --limit-feeders: "
                         "gate runs train on the smallest feeders, where steps are cheap")
    ap.add_argument("--fb-points", type=int, default=0,
                    help="mid-rollout line-residual feedback points (0 = off)")
    ap.add_argument("--warmup", type=float, default=0.03,
                    help="warmup fraction of total steps before the cosine decay")
    ap.add_argument("--seed", type=int, default=0,
                    help="model init + data order. The feeder SPLIT stays pinned at 42 so\n                          seeds measure training variance, not split variance.")
    ap.add_argument("--no-cur", action="store_true",
                    help="FAST path: skip current decode + recon-ctx entirely "
                         "(requires w_i=0, w_kcl=0, fb_points=0; the dominant CPU "
                         "cost feeds outputs those losses never read)")
    ap.add_argument("--bf16", action="store_true",
                    help="autocast forward+loss to bfloat16 (H100); no GradScaler needed")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="run the (expensive) eval+lenses every N epochs")
    ap.add_argument("--prof", action="store_true",
                    help="per-epoch wall-time split: data-wait / H2D / forward / "
                         "backward+step (adds cuda syncs; probe-only)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(reduce-overhead, dynamic=True) on the model")
    ap.add_argument("--out", default=str(ROOT / "runs" / "dk_pf"))
    args = ap.parse_args()
    if args.no_cur:
        assert args.w_i == 0.0 and (args.no_kcl or args.w_kcl == 0.0) and args.fb_points == 0, \
            "--no-cur skips current reconstruction; w_i/w_kcl/fb_points must be 0"
    # four-array architecture (I_bus inputs, Y head) follows the objective; stored
    # in the ckpt args so evaluators rebuild the right encoder widths.
    args.four_mask = args.task in ("random4", "param")
    # DDP under torchrun: one process per GPU. Data-dependent control flow in
    # reconstruct_full means the autograd graph differs per rank, so
    # find_unused_parameters is REQUIRED (absent stores also leave unused heads).
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        # Rank 0 runs the whole eval while the other ranks charge into the next
        # epoch and block on their first allreduce -- that wait is legitimate and
        # must outlive the watchdog (default 10 min killed the first full launch).
        from datetime import timedelta
        torch.distributed.init_process_group("nccl", timeout=timedelta(hours=4))
        torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # H100: TF32 matmuls are ~2x fp32 at negligible loss for O(1) feature nets;
    # the physics that needs precision runs in fp64 outside the model.
    torch.set_float32_matmul_precision("high")

    dev = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    use_feat = not args.no_feat
    t0 = time.time()
    # Split PER CORPUS (pinned at 42 regardless of --seed), then interleave round-robin.
    # Foundation training must not fit one corpus: a union split under --limit-feeders
    # would be dominated by whatever sorts first (63% of the union by count is
    # minimal_component; by static.pt size the smallest files all are), so any gate run
    # would quietly become a synthetic-only run. Interleaving guarantees every limit
    # takes a balanced mix of all four corpora, and per-corpus hashing keeps the split
    # feeder-disjoint within each.
    from itertools import zip_longest
    per_corpus = []
    for root in args.roots:
        spr = split_feeders(discover_feeders(root), seed=42)
        if args.small_first:
            for k in spr:
                spr[k] = sorted(spr[k], key=lambda d: os.path.getsize(os.path.join(d, "static.pt")))
        per_corpus.append(spr)
    sp = {k: [d for tup in zip_longest(*[c[k] for c in per_corpus]) for d in tup if d]
          for k in ("train", "unseen", "test")}
    lim = args.limit_feeders
    tr_dirs = sp["train"][:lim] if lim else sp["train"]
    un_dirs = sp["unseen"][:max(2, (lim // 8) if lim else args.eval_feeders)]
    tv = list(range(args.train_variants)); ev = list(range(args.train_variants, args.train_variants + args.eval_variants))
    train_ds = build_split(tr_dirs, tv, args.task, use_feat)
    # One unseen feeder set, three EVAL LENSES over it. The foundation objective trains
    # on random conditionals; capability is CLAIMED per determinate lens: pf (state from
    # boundary), se (state from partial measurements), injection (Icomp from state).
    unseen_ds = build_split(un_dirs, ev, "pf", use_feat, role="eval")
    lens_ds = {"se": DKDataset(unseen_ds.feeders, ev, task="se", use_feat=use_feat),
               "inj": DKDataset(unseen_ds.feeders, ev, task="injection", use_feat=use_feat)}
    if args.four_mask:
        lens_ds["par"] = DKDataset(unseen_ds.feeders, ev, task="param", use_feat=use_feat)
    print(f"feeders train={len(tr_dirs)} unseen={len(un_dirs)}; "
          f"train_samples={len(train_ds)} unseen={len(unseen_ds)}; build={time.time()-t0:.1f}s", flush=True)

    # fit the ONE global per-family scaler on the train split (on-demand, cached)
    tf = time.time()
    scales = fit_scales(train_ds.feeders, tv)
    print(f"fitted scales in {time.time()-tf:.1f}s | kcl={scales['kcl']:.3e} | "
          + " ".join(f"{s}:I={scales['I'][s]:.2e}" for s in STORES if scales['I'][s] > 1e-8), flush=True)

    model = DKSolver(hidden=args.hidden, steps=args.steps,
                     kcl_feedback=not args.no_kcl, use_feat=use_feat, scales=scales,
                     fb_points=args.fb_points, vabs=args.vabs,
                     four_mask=args.four_mask).to(dev)
    model.skip_current = args.no_cur
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                                fused=torch.cuda.is_available())
    except (RuntimeError, TypeError):
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # Warmup + cosine anneal. Constant LR leaves the last order(s) of magnitude on the
    # table: the reference PINN's low-error runs annealed (s03_clean68_anneal), and its
    # 7.5e-08 run trained 400 epochs -- reaching tiny error needs a tiny final LR.
    steps_total = max(1, args.epochs * max(1, min(args.samples_per_epoch, len(train_ds)) // args.batch_size))
    warm = max(1, int(args.warmup * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: (k + 1) / warm if k < warm
        else 0.5 * (1.0 + np.cos(np.pi * (k - warm) / max(1, steps_total - warm))))
    spe = min(args.samples_per_epoch, len(train_ds)) // max(1, world)
    # each rank draws an independent random subset (seeded differently); gradients
    # are averaged by DDP, so this is plain data parallelism over samples
    gen = torch.Generator(); gen.manual_seed(args.seed * 7919 + rank)
    sampler = torch.utils.data.RandomSampler(train_ds, num_samples=spe, generator=gen)
    tr_collate = make_dk_collate(train_ds.feeders, need_ctx=not args.no_cur)
    un_collate = make_dk_collate(unseen_ds.feeders, need_ctx=not args.no_cur)
    # persistent_workers: without it the pool is re-forked EVERY epoch, and each worker
    # re-imports torch/PyG (~60s of the epoch). prefetch keeps the GPU fed while a worker
    # builds the next batch's per-variant recon ctx.
    dl_kw = dict(num_workers=args.workers,
                 multiprocessing_context="fork" if args.workers else None,
                 pin_memory=torch.cuda.is_available())
    if args.workers:
        dl_kw.update(persistent_workers=True, prefetch_factor=4)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          collate_fn=tr_collate, **dl_kw)
    unseen_dl = DataLoader(unseen_ds, batch_size=args.batch_size, collate_fn=un_collate, **dl_kw)
    lens_dl = {k: DataLoader(v, batch_size=args.batch_size, collate_fn=un_collate, **dl_kw)
               for k, v in lens_ds.items()}
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train(); agg = {}
        te = time.time()
        prof = {"data": 0.0, "h2d": 0.0, "fwd": 0.0, "bwd": 0.0} if args.prof else None
        t_mark = time.time()
        for batch, plan, rctx in train_dl:
            if prof is not None:
                prof["data"] += time.time() - t_mark; t_mark = time.time()
            batch = batch.to(dev, non_blocking=True)
            batch.tree_plan = plan; batch.recon_ctx = rctx
            if prof is not None:
                torch.cuda.synchronize(); prof["h2d"] += time.time() - t_mark; t_mark = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                dv, cur, aux = model(batch)
                loss, m = losses(batch, dv, cur, scales, use_feat, w_v=args.w_v, w_i=args.w_i,
                                 w_kcl=0.0 if args.no_kcl else args.w_kcl, norm=args.norm_loss,
                                 aux=aux, w_ic=args.w_ic, w_y=args.w_y)
            if prof is not None:
                torch.cuda.synchronize(); prof["fwd"] += time.time() - t_mark; t_mark = time.time()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            if prof is not None:
                torch.cuda.synchronize(); prof["bwd"] += time.time() - t_mark; t_mark = time.time()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        if prof is not None:
            tot = sum(prof.values()) + 1e-9
            print(f"PROF ep{epoch:03d}: " + " ".join(
                f"{k}={v:.1f}s({100*v/tot:.0f}%)" for k, v in prof.items()), flush=True)
        n = max(1, len(train_dl))
        tr = {k: v / n for k, v in agg.items()}
        # eval + logging on rank 0 only; other ranks proceed to the next epoch
        if world > 1:
            torch.distributed.barrier()
        if rank != 0:
            continue
        if epoch % args.eval_every and epoch != args.epochs:
            # skipped-eval epoch: still checkpoint (crash safety) + short train line
            state = (model.module if world > 1 else model).state_dict()
            torch.save({"model": state, "args": vars(args), "epoch": epoch, "scales": scales},
                       Path(args.out) / "last.pt")
            print(f"ep{epoch:03d} {time.time()-te:.0f}s | train V={tr['v_wape']:.3f}% "
                  f"skill={tr['v_skill']:.3f} ic_wape={tr['ic_wape']:.1f}% "
                  f"y_wape={tr['y_wape']:.1f}% (eval skipped)", flush=True)
            continue
        model.eval(); ea = {}
        with torch.no_grad():
            for batch, plan, rctx in unseen_dl:
                batch = batch.to(dev); batch.tree_plan = plan; batch.recon_ctx = rctx
                dv, cur, aux = model(batch)
                _, m = losses(batch, dv, cur, scales, use_feat, aux=aux)
                for k, v in m.items():
                    ea[k] = ea.get(k, 0.0) + v
        ne = max(1, len(unseen_dl)); un = {k: v / ne for k, v in ea.items()}
        lens = {}
        with torch.no_grad():
            for lname, dl in lens_dl.items():
                la = {}
                for batch, plan, rctx in dl:
                    batch = batch.to(dev); batch.tree_plan = plan; batch.recon_ctx = rctx
                    dv, cur, aux = model(batch)
                    _, m = losses(batch, dv, cur, scales, use_feat, aux=aux)
                    for k, v in m.items():
                        la[k] = la.get(k, 0.0) + v
                nl = max(1, len(dl))
                lens[lname] = {k: v / nl for k, v in la.items()}
        fam_str = " ".join(f"{k[2:]}={un[k]:.1f}" for k in sorted(un) if k.startswith("i_"))
        print(f"ep{epoch:03d} {time.time()-te:.0f}s | train V/I={tr['v_wape']:.3f}%/{tr['i_wape']:.3f}% "
              f"kcl={tr['kcl']:.3e} | unseen V/I={un['v_wape']:.3f}%/{un['i_wape']:.3f}%\n"
              f"        V skill: train={tr['v_skill']:.3f} unseen={un['v_skill']:.3f} "
              f"(1.000 = no better than dv=0; dv=0 scores v_wape {un['v_base']:.2f}%)\n"
              f"        unseen I/fam: {fam_str}\n"
              f"        lenses: se v_skill={lens['se']['v_skill']:.3f} I={lens['se']['i_wape']:.2f}% | "
              f"inj ic_wape={lens['inj']['ic_wape']:.2f}% I={lens['inj']['i_wape']:.2f}%"
              + (f" | par y_wape={lens['par']['y_wape']:.2f}%" if "par" in lens else ""),
              flush=True)
        state = (model.module if world > 1 else model).state_dict()
        torch.save({"model": state, "args": vars(args), "epoch": epoch, "scales": scales},
                   Path(args.out) / "last.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
