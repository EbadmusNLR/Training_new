# Training experiment ledger

One line of config-delta and one of verdict per experiment. Full forensic detail stays in
`experiments.md` (§25 for the 2026-07-17 decoder/plateau work); this file is the quick map.
Conventions: v_skill = unseen |err|/|dv| (1.0 = no better than predicting dv=0);
gate config = 60 feeders balanced over all 4 corpora, small-first, bs8, spe1600, 40 ep,
lr 4e-4 anneal, norm-loss, residual gauge, task=random, exact decoder.

| id | jobs | delta vs gate config | result | verdict |
|---|---|---|---|---|
| T01 seed tests | 15237232-34 | PRE-fix: 40f SMART-DS only, mixed loss, no gauge/anneal, 8 ep | unseen v_skill ~1.0 all seeds, train V > null | FAIL — known-bad loss (no norm) + ~10x too short |
| T02 overfit sweep | 15240844 | ONE fixed batch (6 SMART-DS feeders), 400 Adam steps/config | mixed 0.293; norm 0.51@300 (faster early, worse late); I/V grad ratio 2.1 | optimization works; decoder-pullback theory REFUTED; horizon was the killer |
| T03 gates x3 | 15241690-92 | gate config, seeds 0/1/2 | RUNNING (~40s/ep) | go/no-go for full launch: all seeds <1.0, trending down, within ~0.1 |
| T04 abl pf | 15241727 | task=pf | RUNNING | is the foundation objective harder than single-task on the pf lens? |
| T05 abl lr | 15241728 | lr=1e-3 | RUNNING | LR headroom |
| T06 abl steps | 15241729 | steps=24 | RUNNING | recurrence depth |
| T07 abl width | 15241730 | hidden=384 | RUNNING | capacity |
| T08 abl nonorm | 15242531 | mixed loss (NORM=0) | RUNNING | T02 says mixed wins late on a fixed batch; does it in generalization? |
| T09 ddp smoke | 15241743 | 2-rank torchrun, tiny run | QUEUED (QOS) | validates the 4-GPU full path |

## Standing decisions (why the gate config looks like this)
- **exact decoder in-model** (6.7e-01 -> 3.4e-06 on truth V; vsource was silently zero before)
- **norm-loss default**: without it the current term drowned V (v_skill ~1.0, measured twice)
- **residual gauge + anneal**: the reference PINN's 7.5e-08 recipe (t5_kcl_fp64_gauge)
- **task=random**: one random conditional per sample over V+Icomp; pf/se/injection are
  inference-time masks, evaluated as lenses — not training categories
- **multi-corpus stratified splits**: naive union under limit = 100% minimal_component
- **plateau cause**: MP step = Gauss-Jacobi = divergent on Ybus; sweep architecture is the
  path to machine precision (1.23e-12 shown transformer-free; xfmr null-space OPEN)

## Slurm strategy (fast acceptance)
- Ask the SHORTEST honest walltime: gates finish ~35 min — request 1:00:00, not 4:00:00.
  Short jobs backfill into scheduling gaps; a 4h request waits for a 4h hole.
- Probes/gates -> `gpu-h100s` (4h cap, backfills well). Smokes -> `debug-gpu`
  (fast but QOSMaxJobsPerUser=1). Only the full multi-day run -> `gpu-h100`.
- Check `sinfo -p <part> -o "%P %F"` (A/I/O/T) before choosing; 0 idle = queue anyway
  but expect backfill only if walltime is short.
