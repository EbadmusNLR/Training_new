# Handoff — DG foundation model (session ending 2026-07-18)

Read this, then `training_experiments.md` (rows T28b–T32e are this session) and the
memory file `dgfm-capability-matrix.md`. Everything below is committed on branch
`datakit-tree-current`.

## THE MISSION (do not narrow it)
The datakit reduction makes the entire grid **four arrays: V, I_bus, Icomp, Y**.
Every task (pf/se/injection/param) is just a *visibility pattern* over them — four
named examples out of arbitrarily many. There is **no P/Q** in this representation.
The objective is the general conditional: **mask any subset, reconstruct the rest**,
including under noisy/bad visible entries, whole missing components, whole dark regions.
Judge every design decision against that, never against one task's score.

## Standing user policies (must preserve)
- **No full-scale/long runs without explicit user sign-off.** Probes only.
- **Replication protocol**: every experiment on a RANDOM feeder subset; a config is
  believed only after it replicates on a *different* random subset (301/302 are the
  two standard subset seeds).
- Always commit to git as work progresses; one ledger row per experiment.
- Kestrel: never heavy CPU/IO on login nodes — wrap in `sbatch`.
- Mid-run reads deceive. **Only finals count** (burned repeatedly, incl. this session).

## State of the four arrays

| array | status |
|---|---|
| **V** | SOLVED. fp64 direct solve of Ybus = machine precision. `pf` lens exact by construction. |
| **Icomp** | Estimator beats the zero prior *at full scale* (T28b: 6.7x in the nullspace). Probe scale cannot resolve this either way. |
| **noisy/bad entries** | SOLVED at solver level (T29/c/d). Naive solve breaks (max ~1e5); null-space WLS + MAD bad-data rejection is **sub-floor**, precision 1.00 on gross errors. |
| **Y** | PARTIALLY solved — see below. The one open piece. |

### Key results this session
- **T28b**: first measured model value *through* the physics solve — joint med 8.02e-07
  vs zero-prior joint0 5.35e-06 on underdetermined feeders. Flips T21.
- **T29/c/d**: noise stage characterized end-to-end. `wp` (trust in model Icomp) dose
  response never turns over through wp=10; robust median goes *below* the measurement
  noise floor on both seeds. Only limit is information-theoretic (gross/noise < ~4-5).
- **T29b/T30**: region20 (20% dark) closed by physics (4.5e-11). region40 shows the
  iid-mask estimator transfers *negatively*; `REGION_MIX` implemented but **REJECTED at
  probe scale** (measured iid cost, no transfer fix). It is a full-scale-only hypothesis.
- **T31**: the Y codebook is real, global and **saturating** — 95% of lines are ~34
  normalized families (vs 128 free DOF), stable 60→240 feeders.
- **T32→T32e**: Y-head arc, below.

## Where Y stands (the live thread)

Design: `Y = s · P`. The model **classifies the pattern P** (codebook, learned) and the
**scale s** is obtained by closed-form least squares from visible terminal currents:
`s = Re⟨Pv, i⟩ / ‖Pv‖²`  (v5.4, env `YCB_ANALYTIC_SCALE=1`).
This is the campaign's proven split: net supplies discrete structure, physics supplies
the continuous part.

Progression of `par` lens y_wape on line (100% = predict-zero baseline):
free-entry head 388k–3.2M% → codebook+free scale 3616–6349% → decoupled clamped scale
489–2421% → **analytic scale 76–91% (sub-100%)**.

**T32e (last experiment, replicated on 301 and 302):**
- shm crashes **fixed** (0 aborts; first clean 40-epoch probes of the arc).
- **V trunk is the best of this config: unseen 0.659 / 0.726.** Analytic Y does not harm
  the trunk — earlier 1.4–2.2 readings were mid-run artifacts of crashed runs.
- Y line reaches **76–91% through ~epoch 20, then DEGRADES** (139→264→11088% on 301;
  272→6369% on 302). Same pattern both subsets.
- Per-store at the good point: storage 0%, capacitor 2–18%, transformer 36–56%, line
  76–91%; load/pvsystem/reactor ~100% (their Y is near-zero, so predict-zero is ~right).

### START HERE: the next fix (root cause already identified in code)
`gridfm/dk_model.py`, v5.4 analytic branch. The guard on the projection denominator is
**absolute**:
```python
s_an = num / den.clamp(min=1e-20)     # den = ||Pv||^2
```
When the classifier becomes confident and picks a pattern that nearly annihilates `v`,
`den ~ 1e-15` and `s` explodes. That is exactly the late-training degradation.

Apply the **relative rcond discipline the joint solve already uses** (T19c, which fixed
the identical failure there):
1. Accept the analytic scale only when well-conditioned, e.g.
   `den / (‖P‖²·‖v‖² + eps) > tol` (start tol ~1e-6, sweep it).
2. Otherwise fall back to the learned clamped scale (already implemented, `YCB_ABS_SCALE`).
3. Additionally clamp `s` to the store's global observed range (`ycbglob_{s}` buffer
   holds `[mean, std]` of log-scale — already registered).
Then re-probe on subsets 301 and 302 (protocol) and ledger as T32f.

Expected outcome: line holds 76–91% for all 40 epochs instead of diverging. If it does,
Y is solved *where observable* and the four-array capability is complete in principle.

### Honest scope note on Y (do not overclaim)
T22 established Y is **excitation-limited from V-only snapshots** — load variants excite
~1 of 6 modes, so line Y is genuinely unidentifiable there, and no capacity fixes it.
v5.4 does **not** contradict T22: it uses a *richer conditional* (terminal currents
visible), where the scale is algebraically determined. State the capability
conditionally: **line Y is unrecoverable from voltages alone, recoverable when terminal
currents are visible.** The memory note currently phrases the line result as a flat
"~100% floor" — refine it to this conditional form once T32f confirms.

## Config of record (four-array training)
`--no-cur --ctx-points 2 --ic-d-only --ic-sce` with `SCE_MAG=0.5`, `REGION_MIX=0`.
Replicated (T27): unseen 0.710 / 0.783. Add `YCB_ANALYTIC_SCALE=1` once T32f lands.

## Infrastructure notes (hard-won)
- **shm aborts** (`could not unlink the shared memory file`) killed many runs. Cause:
  six DataLoaders (train + unseen + 4 lenses), each with `persistent_workers` × 16
  workers ≈ 96 processes holding `/dev/shm`. **Fixed**: eval loaders now `num_workers=0`.
  Keep it that way.
- **Scheduling**: gpu-h100s idle nodes are reserved for the long partitions, so short
  jobs get walled out. Levers that work: short honest walltime (90 min, not 2 h),
  `--partition=debug-gpu` with `--mem=170G` (its QOS caps at 180G, 1 job/user), or
  `--partition=gpu-h100` where those nodes are allocatable. debug-gpu is the reliable one.
- Probes: 240 feeders, 40 epochs ≈ 100–125 s/epoch ≈ 75 min.
- Smoke test before any GPU probe: `gridfm/probes/ycb_smoke.py` (seconds, CPU, shared
  partition) — exercises codebook build → forward → loss → backward.

## Queued / not yet done
- **T32f**: the relative-rcond fix above. *Do this first.*
- Per-component **learned confidence (learned-GLS)** for the weighted solve — the wp
  dose-response says a global knob is already strong, so per-component weights should pay.
- **Full-scale run** of the config of record on the mesh-inclusive corpus (`--with-mesh`,
  ~100x faster build). **READY BUT REQUIRES USER SIGN-OFF.** Open questions to fold in:
  whether to include a small `REGION_MIX` dose (full-scale-only hypothesis), and whether
  the learned-confidence head goes in this run or the next.
- Multi-snapshot excitation for Y (switching/topology events) if line Y is ever needed
  from V-only data — the only route past T22.

## Useful probes
- `gridfm/probes/direct_solve_e2e.py` — multi-lens e2e scorecard (leak-proof rhs, joint
  vs zero-prior, nullity audit). Lenses incl. region10/20/40, sw0/20/50/80.
- `gridfm/probes/noisy_se_probe.py` — naive vs WLS vs null-space WLS vs robust under
  planted noise + gross errors.
- `gridfm/probes/y_family_survey.py` — Y codebook family statistics per store.
- `gridfm/probes/ycb_smoke.py` — fast correctness smoke for the Y-head path.
