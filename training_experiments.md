# Training experiment ledger

One line of config-delta and one of verdict per experiment. Full forensic detail stays in
`experiments.md` (§25 for the 2026-07-17 decoder/plateau work); this file is the quick map.
Conventions: v_skill = unseen |err|/|dv| (1.0 = no better than predicting dv=0);
gate config = 60 feeders balanced over all 4 corpora, small-first, bs8, spe1600, 40 ep,
lr 4e-4 anneal, norm-loss, residual gauge, task=random, exact decoder.

| id | jobs | delta vs gate config | result | verdict |
|---|---|---|---|---|
| T01 seed tests | 15237232-34 | PRE-fix: 40f SMART-DS only, mixed loss, no gauge/anneal, 8 ep | unseen v_skill ~1.0 all seeds, train V > null | FAIL — known-bad loss (no norm) + ~10x too short |
| T02 overfit sweep | 15240844 | ONE fixed batch (6 SMART-DS feeders), 400 Adam steps/config | FINAL@400: mixed 0.293, norm 0.422, **vonly 0.236** (+ best i_wape 1.25%, NO current loss), vonly_hi 0.389 | optimization works; decoder-pullback theory REFUTED; horizon was the killer |
| T03 gates x3 | 15241690-92 | gate config, seeds 0/1/2 | train 0.60/0.71/0.76, unseen 0.92/0.90/0.96 flat; kcl->2e-6, I~2% | NO LAUNCH: learns but does not transfer at 60f — generalization gap, not optimization |
| T04 abl pf | 15241727 | task=pf | train 0.437, unseen **0.816** (best) | matched conditional wins at small scale; random needs topology diversity to pay off |
| T05 abl lr | 15241728 | lr=1e-3 | unseen 0.907 | no gain over 4e-4 |
| T06 abl steps | 15241729 | steps=24 | unseen 0.914 FINAL (ep40) | no effect either way; BOTH mid-run reads (0.905@10, 1.092@28) were noise — only finals count |
| T07 abl width | 15241730 | hidden=384 | unseen 0.910 | no capacity win at this scale |
| T08 abl nonorm | 15242531 | mixed loss (NORM=0) | unseen 0.905 FINAL | norm vs mixed indistinguishable in generalization |
| T09 ddp smoke | 15241743 | 2-rank torchrun, tiny run | QUEUED (QOS) | validates the 4-GPU full path |
| T10 scale gate | 15248741-43 (v4) | feeders 60->240 (random s0/s1 + pf control), 30 ep | RUNNING | THE decision: unseen skill responds to feeder count -> full launch; flat -> model work first. v1-v3 died at startup: IEEE30 refusal, bridge-chord collate refusal, 5% gate on a 30-feeder eval split -- all three now handled (loud-skip; train-only gate) |
| T10 FINAL | 15248741-43 | -- | random s0 0.92->**0.824**, pf 0.816->**0.765**, random s1 train-stalled 0.85 -> unseen 1.049 | SCALE WORKS (2/3); s1 = optimization wedge, suspect = ic_head sinh explosion (ic_wape 16713% in ddp smoke) -> clamped z to +-8 |
| T11 vonly | 15254796 | 240f, W_I=0 W_KCL=0 | FINAL unseen **0.789** (train 0.573) -- BEST of campaign | Confirms T02 at scale: exact decoder couples I to V, so explicit I loss is redundant AND injects stiff reactor/xfmr gradients (reactor 8.7e14%, xfmr 2666%) that fight V. Drop I loss; currents stay machine-exact from decoder. THE full-launch config |
| T12 stability | 15254801 (s2) | 240f s2: ic-clamp + warmup 0.1 | FINAL unseen **0.848** COMPLETED -- looked wedged at 0.98 mid-run, converged late | WEDGE FIXED (clamp+warmup); mid-run reads deceive again. s1 (15254800/15255130) died 3x on shm crashes -- see shm note below. Gate evidence: s0 0.824 / vonly 0.789 / s2 0.848, three clean runs within 0.06 -> FULL LAUNCH JUSTIFIED |
| T13 fix matrix | 15254842-44, 15255068 | 240f seed0, one var each: PE hopcap 150 / lr 2e-4 / bs16 / fb-points 3 | FINALS: pe150 1.182 FAIL (train 0.554 = memorizes, no transfer); **lr2e4 0.789 TIES vonly** (late anneal 0.833->0.789); bs16 1.308 FAIL; fb3 died ep13 (1.103, failing) | TWO winners at 0.789: vonly@4e-4 and I-loss-on@2e-4. Lower peak LR is a real generalization knob |
| T14 FULL LAUNCH | 15256412 (s0), 15256576 (s1) | 4xH100 DDP, ALL feeders, task=random, vonly (W_I=0). s0=lr4e-4 120ep/40h; s1=lr2e-4 48ep/16h backfill (combo bet from lr2e4 tie; replaced 15256442) | s0 DIED end of ep1 (NCCL watchdog): rank 0 evals 319 unseen feeders x 10 variants x 3 lenses ALONE (~9.4k forwards) while ranks 1-3 enter ep2 and their allreduce times out at 10 min. Probe-scale evals were small, so the bug only bites at full scale. s1 killed before hitting the same wall | FIX: --eval-feeders cap 48 + 4h pg timeout (commit b21b51d). RELAUNCHED as 15259366 (s0) / 15259367 (s1) |
| T16 pf-first probes | 15259369-72 (minimal), 15259373-74 (240f steps), 15259375->15259503 (e2e) | USER DIRECTION: random masking stays THE objective, but nail the pf mask first (V hidden, Icomp+Y+slack visible = OpenDSS's own linear problem: Ysystem + Norton Icomp, one solve -- manual confirms). Curriculum: minimal_component alone x {base, --no-feat (no asinh), --vabs (absolute V head), both}; multi-corpus 240f steps 12 vs 32 (does pf need MP depth?); e2e = fix_lr2e4 Icomp estimates -> direct fp64 solve vs its V head | minimal FINALS (unseen pf skill): base 0.407, nofeat 0.397, vabs 0.339, **vabs+nofeat 0.240** (train 0.087; its c10 abort was the nonfatal teardown race -- all 40 eps ran). pf240_s12 FINAL 0.770 (matches T10 0.765) | **USER'S BOTH HYPOTHESES WIN AND COMPOUND on minimal: -41% vs baseline.** asinh alone a wash but interacts with vabs. inj ic_wape 100% expected: pf mask never hides ic, so ic_head is untrained in pf-only probes. Migration probes launched: pf240_vabs 15259783, pf240_vn 15259784, rnd240_vn 15259785 (+15259782 vn replicate on minimal) |
| T18 migration + steps + ic | 15259782-85, 15259374, 15259636-37 | vabs/nofeat to multi-corpus (pf + random); steps 12 vs 32; w_ic-focused training | pf240_vabs 0.910, pf240_vn 0.900 vs s12 base **0.770**; rnd240_vn 1.062 vs vonly 0.789; s32 0.850; mc replicate 0.326 (vs 0.240 -- run variance). ic_wape finals: ic_only 107.2, ic_heavy 102.2, vonly base 107.7, fix_lr2e4 156.0 | **vabs+nofeat is a MINIMAL-ONLY win -- does NOT transfer** (SMART-DS dv ~4% makes residual frame a strong prior; minimal dv 36.6% favors absolute). Full-run config (dv-mode vonly) stands. More MP depth loses at probe budget. ic quality saturates ~100-107% WAPE under every loss recipe -> estimator needs a mechanism (physics/iteration), not weights. e2e on ic ckpts: 15260346/47 |
| T17b pf lens exact | 15259626 | multi-lens e2e: pf mask (truth Icomp) through the FULL data pipeline -> solve | skill_solve 0.000 on ALL feeders incl. 10471-node SFO P13U (head: 0.21-1.96). Only skips = 37Bus/IEEE9500, blocked by the TREE DECODER's radial requirement, not the solve | pf capability = machine-exact, zero learned params, corpus-wide. Mesh feeders are decoder-blocked only -> no-recon mode (task #9) unlocks them |
| T17 E2E SOLVE WINS | 15259503 | fix_lr2e4 ckpt, random mask, 15 unseen feeders: hidden-Icomp ESTIMATES + truth visible -> one fp64 Ybus solve, vs the same ckpt's V head | **skill_solve 0.000-0.043 vs skill_head 0.68-2.05 -- ~100x, on EVERY feeder**, even at 49-67% hidden Icomp (IEEE123 0.010 vs 0.675; 1638-node SFO P6U 0.008 vs 2.045; 5037-node P9U 0.004 vs 0.756) | **ARCHITECTURE DECIDED (measured): the GNN's V head is dead weight. Model = hidden-Icomp estimator; V = direct fp64 solve of Ybus (topology-agnostic: mesh or radial). pf lens exact by construction; random/se lens ~0.01 TODAY with an ic_head that was never the training focus. Next: solve-decoder eval path + pivot training weight to ic quality** |
| T15 solver probes | 15256418/491/636/688 (v1-v3+diag) | ladder vs DIRECT fp64 solve of full Ybus, truth Icomp, 5 feeders x 4 corpora | **DIRECT SOLVE: 20/20 machine precision, worst 4.4e-8** (yd-xfmr that DIVERGED the ladder: 1.6e-11). Ladder 18/20; its one divergence = As singular in delta zero-seq, grounded only by shunt load Y (diag) | **SOLVER LAYER = ONE SPARSE fp64 SOLVE of Ybus.** cond is ~4e9 here, not the old 1e18 -- no iteration, no splitting, no xfmr null-space needed for V. P10U gain ~500 persists through direct = REAL physical stiffness of zero-seq injection modes (problem property, 5 orders below V->I 2e7). Design: model estimates hidden Icomp (directly supervised); V = solve(Ybus, Icomp). pf lens becomes EXACT by construction. Next: e2e = fix_lr2e4 ckpt (trained ic_head, 0.789) ic estimates -> direct solve V vs its V head |

**shm crash note**: 'could not unlink shared memory file' aborts (s1 x3, fb3) happened MID-RUN,
not at teardown, and both victims shared a node with another of our jobs -- file_system sharing
+ co-tenant /dev/shm is the correlate. Full run holds its node alone (4/4 GPUs) so exposure is low.
Durable fix if it recurs: consolidate batched recon-ctx into a few large tensors.

## Standing decisions (why the gate config looks like this)
- **exact decoder in-model** (6.7e-01 -> 3.4e-06 on truth V; vsource was silently zero before)
- **norm-loss default**: without it the current term drowned V (v_skill ~1.0, measured twice)
- **residual gauge + anneal**: the reference PINN's 7.5e-08 recipe (t5_kcl_fp64_gauge)
- **task=random**: one random conditional per sample over V+Icomp; pf/se/injection are
  inference-time masks, evaluated as lenses — not training categories
- **multi-corpus stratified splits**: naive union under limit = 100% minimal_component
- **plateau cause**: MP step = Gauss-Jacobi = divergent on Ybus; sweep architecture is the
  path to machine precision (1.23e-12 shown transformer-free; xfmr null-space OPEN)

## Qualified claims (keep these honest)
- "unseen" currently means unseen RADIAL topologies: meshed/bridge-chord feeders are
  loudly excluded from train AND eval until batch_recon_ctx merges kvl/binj
  (the batched-bridge gap). IEEE 30 Bus additionally needs the 3 missing DOF.
- exclusion counts at 240f: train 2/240 (0.8%), eval 4/30 (13%) -- the vendored corpora
  are meshed-heavy relative to SMART-DS.

## Slurm strategy (fast acceptance)
- Ask the SHORTEST honest walltime: gates finish ~35 min — request 1:00:00, not 4:00:00.
  Short jobs backfill into scheduling gaps; a 4h request waits for a 4h hole.
- Probes/gates -> `gpu-h100s` (4h cap, backfills well). Smokes -> `debug-gpu`
  (fast but QOSMaxJobsPerUser=1). Only the full multi-day run -> `gpu-h100`.
- Check `sinfo -p <part> -o "%P %F"` (A/I/O/T) before choosing; 0 idle = queue anyway
  but expect backfill only if walltime is short.
