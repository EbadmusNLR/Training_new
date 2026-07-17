# Experiments

## Promotion contract

1. No topology-specific baseline derived from solved voltage targets.
2. No solver-assisted model output.
3. Slack voltage and every-bus `V_init` are always known.
4. Train/seen/unseen feeder sets are asserted disjoint where required.
5. Metrics are split-level WAPE percentages; slack voltage is excluded.
6. Truth-physics and tiny-overfit gates precede scaling.

## E0 — edge-state scaffold (2026-07-12)

- Commit: `6a7a8ba`; data: validated scenario-store dependency.
- Model: terminal-edge recurrence + global state + complex dV/drop heads; baseline `V_init`.
- Gates: syntax PASS; two static contract tests PASS.
- Compute: Slurm socket denied by agent sandbox; no login-node PyG/training fallback.
- Verdict: implementation ready for compute smoke; no accuracy claim.

## E1–E7 — current isolation (2026-07-12)

| Run | Feeders | Seen V / Ibus WAPE | Unseen V / Ibus WAPE | Verdict |
|---|---:|---:|---:|---|
| one-feeder V only | 1 | 0.0348% / n.a. | n.a. | voltage capacity passes |
| one-feeder I only | 1 | n.a. / 6.87% | n.a. | current head has finite interpolation floor |
| E4 normalized | 400 | 0.624% / 5.13% | 3.322% / 70.84% | non-line devices learn; unseen line flow fails |
| E5 physical WAPE | 200 | 1.378% / 4.42% | 3.305% / 73.66% | helps seen current, hurts voltage |
| E6 topology scale | 1000 | 2.287% / 7.00% | 4.664% / 48.19% | final epoch 50; topology coverage is strongest current lever |

- Oracle decode using truth voltage: unseen aggregate current WAPE `0.0000118%`; data and current decoder pass.
- Predicted voltage through the stiff physics decoder is unstable; direct current heads remain necessary.
- Old-corpus issue: TriplexLine `I_scale=9.41e-10`. E7 separates scale-only `v3f` from `det2f`, which also carries the later wiring/determinacy fixes; both use floored scales.

## E7 — corrected corpus gate (2026-07-12)

- `det2f`: exact `det2` re-encode with current floor only; all Y features unchanged.
- Fixed inherited flag-cache bug: cache covered 40/2,000 feeders; missing Line/TriplexLine families now come from baseline JSON and fail closed.
- Clean committed validator: `2,000/2,000` stores PASS (`Ibus + Icomp = YV`, KCL, float64 schema).
- GPU matrix: normalized 400/1,000/2,000; raw 400; physical-Ibus-WAPE 400; hidden-256 400.

## E8–E10 — topology/current solution (2026-07-12)

| Model | Seen V / Ibus | Unseen V / Ibus | Verdict |
|---|---:|---:|---|
| mean H128, 2,000 | 1.832% / 8.673% | 2.904% / 28.213% | topology scale breaks old current floor |
| WAPE H128, 2,000 | 2.001% / 6.084% | 3.011% / 24.261% | physical current objective helps |
| WAPE H256 + tree current | 0.845% / 1.633% | 2.021% / 6.851% | first strong structural result |
| WAPE H256, 12 steps + tree | n.a. | 1.864% / 6.808% | deeper propagation helps voltage |
| WAPE H384 + tree | 0.570% / 1.368% | 1.767% / 6.732% | selected on unseen validation |

- Tree current reconstructs paired line series flow by subtree KCL; it never reads voltage or invokes a PF/linear solve.
- Selected unseen family WAPE: line `6.582%`, transformer `3.678%`, load `3.345%`, Vsource `7.108%`.
- Oracle tree-current WAPE: `0.00000538%`; decoded-current contract is effectively exact.
- Mean local aggregation beats naive sum/local-sum; explicit structural accumulation belongs in the current decoder.
- H384 test, opened after hash-pinned selection: `2.124% V / 6.888% Ibus`; line
  `6.988%`, transformer `4.209%`, load `4.682%`, Vsource `6.655%`, KCL `2.13e-5 pu`.
- Late-checkpoint averaging was worse (`1.771% / 6.932%` unseen) and rejected.

## E11-E28 — current + foundation correction (2026-07-12)

- Best PF validation so far: E15 full exposure, `1.556% V / 6.371% Ibus`; E14 remains slightly better on current alone (`6.348%`).
- Near-zero currents are not inflating WAPE: truth below `1e-4 pu` supplies `0.49%` of the error numerator; truth above `0.1 pu` supplies `88.6%`.
- Large-current thresholds, reactor-only loss, H512 scaling, and reactor-as-line structural decoding were negative.
- Foundation audit invalidated old `topo/sysid/ctrl` claims: connectivity was visible, masked-Y PE leaked truth, single-snapshot joint Y/Icomp was underdetermined, and control labels do not exist.
- Leakage-free tasks are PF, known-injection SE, one-entry Y completion, Icomp completion, and paired random masking. Mask gate PASS; selection uses worst required task-field WAPE.
- Clean E19 unseen @20: PF `1.790% / 9.889%`, SE `2.048% / 11.178%`, Y `1.494%`, Icomp `2.550%` (direct heads).
- Role heads E21 @20: Y `1.252%`, Icomp `1.998%`; PF/SE current remains the bottleneck. Full-exposure, task-conditioned, structural-PF, directional, and staged-random continuations are active.

## E29-E56 — broad foundation selection (2026-07-12)

| Run | PF V / I direct | SE V / I | Y / Icomp | Worst Y / Icomp scale | Verdict |
|---|---:|---:|---:|---:|---|
| E32 aggregate | 1.708% / 9.683% | 1.902% / 10.669% | 0.840% / 0.742% | 4.101% / 1.695% | current specialist |
| E40 store-balanced | 1.694% / 9.782% | 1.892% / 10.823% | 0.857% / 0.487% | 2.776% / 1.246% | broad baseline |
| E51 transformer 0.1 | 1.691% / 9.753% | 1.876% / 10.713% | 0.843% / 0.479% | 2.628% / 1.200% | broad winner |

- Canonical identifiable random on E40: V `1.751%`, Ibus `9.946%`, Y `0.843%`, Icomp `0.494%`.
- Simultaneous all-field stress is underdetermined: E40 `9.396% / 44.982% / 15.210% / 14.909%`; 5-10% stress training improves it but harms core tasks.
- Weight soups, task conditioning, directional sweeps, H512, reactor losses, and stronger transformer weight `0.3` were rejected.
- Exact dense PF ceiling gives `0.021%` V but line `Y_s(V1-V2)` remains numerically ill-conditioned on singular/fallback cases; V WAPE alone does not certify current.
- Local Jacobi reduces V to `1.563%` at 32 steps but cannot make stiff-Y current safe. Hybrid device physics + tree KCL gives E32 `6.371%` Ibus; exact `jY_h(V1+V2)/2` shunt decoding is correct but only a small gain.
- Current error is not a near-zero metric artifact: >`0.1 pu` truth supplies `88.6%` of its numerator. Transformer/reactor and accumulated branch flow are the remaining learned bottleneck.
- Checkpoint selection now fails closed on aggregate tasks plus worst family-scale fields; zero-denominator raw storage-Y WAPE no longer prevents checkpoint creation.
- E51 final: seen PF `0.702% / 1.644%` structural; unseen `1.691% / 6.510%` best structural-hybrid; fixed test `2.106% / 6.535%`. Test Y/Icomp are `0.918% / 0.481%`; safe-random V/I/Y/Icomp are `2.140% / 9.619% / 0.939% / 0.462%`.
- Promoted artifact: `runs/foundation_best`, SHA-256 `1c9a97b9183e0527c42439e8d052135bdaef83d3e9598f7ab35961b5a821ee17`.

## D-attrib — the 6.5% unseen current wall is REACTOR current (2026-07-12)

New diagnostics on the promoted checkpoint (`scripts/attrib_current.py`,
`test_decode.py`, `v_sensitivity.py`, `probe_react.py`, `series_structure.py`),
unseen PF. See memory `dgfm-current-error-is-reactor`.

- **The decode is already solved.** `hybrid(shunt incl. reactor Y·V) + tree_line
  + kcl_vsource` gives aggregate Ibus **0.16% at truth V**, and **0.175% at model
  V when reactor currents are set to truth**. So currents are a well-conditioned
  physics function of V; no better decode exists for E51 as-is.
- **Reactor is the single lever.** Attribution: truth reactor current takes line
  7.7%→0.22% and aggregate 8.0%→2.4%; truth V alone does NOT help line, nor does
  truth shunt. Zeroing reactor → line/vsource 94% (both are KCL-reconstructed
  FROM reactor injections). Reactors are significant carriers: truth sum|I| line
  3.5 / reactor 0.94 / load 0.35; mean|I| reactor ~4× line.
- **Reactor current is stiff Y·V** (exact at truth V, amp ~5×): learned head
  7.4% beats Y·V-from-model-V 12.5%. Folding reactor into tree_line as a paired
  series edge is WRONG (12.4%): reactor truth is ~17× its downstream load, so it
  is a shunt/series reactance current, not a downstream-KCL quantity.
- **Current ≈ 2.7·V** (v_sensitivity, physics decode, Gaussian V noise): line
  2.6×, reactor 4.8×. <1% aggregate ⇒ V<~0.37%. Reactor-adjacent nodes have
  V WAPE 1.34% vs 0.90% elsewhere (low-|V| buses, mean |V| 0.76).
- **E51 has no `lambda_reactor_wape`** — reactor is under-supervised (only the
  balanced recon + aggregate ibus_wape where it is a small magnitude fraction).

### R-probes — both refute fine-tuning fixes (2026-07-12)
- **R1** `configs/r1_reactor_wape.yaml` (E51 + reactor 0.3 + line 0.2 direct-head
  current WAPE, 15 ep): reactor 7.15%→6.95%, line 10.4%→10.3%, V 1.70%→1.67%.
  **NULL.** The reactor head is not supervision-limited; it is at a stiff
  V1−V2 wall (it cannot output Y·(V1−V2) more precisely than the model's
  internal voltage supports).
- **R2** `configs/r2_reactor_physics.yaml` (physics reactor-consistency:
  minimize WAPE(Y_reactor·V_pred, truth reactor), weight 0.3): train loss ~1000,
  V **degraded** 1.70%→3.25%. The stiff Y·V gradient dominates under grad-clip
  and wrecks global voltage. **DESTABILIZES.** (Loss knobs
  `lambda_reactor_physics_wape` / `lambda_line_physics_wape` added to losses.py,
  guarded, default 0.)

### KCL-residual feedback (learned iterative solver) — honest V fix
Added a Donon/PowerFlowNet-style learned iterative solver: each recurrent step
computes the exact nodal KCL residual of the current V estimate (observed
Y/Icomp + predicted V, no solve) and feeds it back (`gridfm/kcl_feedback.py`,
`model.kcl_feedback`, zero-init so a checkpoint reproduces exactly). Directly
attacks the root cause (V), can break the seen ceiling by enforcing physics at
inference. Smoke: zero-init reproduces base exactly; gradient flows from E51.
- **KCL1** (feedback, fine-tune E51, lr 3e-5): DIVERGES — feedback compounds
  over 12 steps, unseen V 1.7%→24% by ep3. Added `node_norm` re-normalization
  after the feedback to bound it.
- **KCL1b** (feedback, FROZEN backbone + re-norm, train only feedback+V-heads,
  lr 8e-5): stable early but V degrades to 6.3% (ep3) then train loss climbs
  0.23→0.86 — the trainable node_head relearns under feedback and loses E51's
  accuracy. Does not beat E51.
- Lesson: E51 is a delicate optimum; the iterative solver must be trained
  FROM SCRATCH so the whole model co-adapts. Launched `configs/kcl_scratch.yaml`
  (from scratch, kcl_feedback, 70 ep, lr 4e-4) as job 15108869 — the honest best
  shot; a single run is unlikely to beat E51's many-experiment tuning, but tests
  whether the iterative-solver inductive bias lowers the V generalization floor.

## PIVOT to real SMART-DS data + task-agnostic iterative solver (2026-07-13)

Decisive finding: **reactors are a minimal_component ARTIFACT** — real SMART-DS
feeders (`training_data/smartds1000/*/static.pt`) have node types line/capacitor/
load/pvsystem/storage/transformer/vsource and NO reactor (gso12 manifest lists
`reactor.*` as missing_required_keys; dss_data_v1 reactors are unit-test feeders).
So the minimal_component 6.5% wall is largely synthetic; the real path is training
on SMART-DS, where current is voltage-bounded and the transformer (many/feeder,
stiff) is the element to watch. (dss_data does contain reactors, so the model
still handles them — but they don't gate the SMART-DS foundation goal.)

**Task-agnostic learned iterative solver** (`gridfm/kcl_feedback.py`,
`model.kcl_feedback`), per user directive "maintain estimates of every hidden var,
compute residuals from completed current estimates, iterate": each recurrent step
computes the nodal KCL residual of the COMPLETED terminal-current estimates
(observed-where-visible, predicted-where-masked) — Σ Ibus, O(1)/well-conditioned
AND differentiable (dr/dIbus=1, no detach) — and feeds it back so the net refines
its hidden state toward physical consistency, whatever variable is masked
(PF/SE/param/injection). First attempt used the stiff Y·V residual (V-only, needed
detach, NaN'd from scratch); the completed-current form is the correct one.
Smoke-validated stable from scratch.

**Real-data pipeline (all validated end-to-end on the pilot corpus):**
- Data gen: `make_training_pt.py` needs `.venv` (opendssdirect); pilot-built 40
  feeders into valid stores.
- Schema: real vsource omits the Icomp block (no Norton compensation = zero) —
  `DG_FM_Training/data.py resolve()` now zero-fills missing optional fields
  (backward-compatible; complete corpora unaffected).
- **PE performance fix** (the blocker): the FeederCache PE was dense O(n²/n³)
  (co-incidence matmul, RWSE P^k, n-iteration Bellman-Ford) — fine for ~100-node
  minimal_component, impossible for 5k-8k-node SMART-DS. Added a size-guarded
  sparse path (scipy csgraph shortest paths + sparse RWSE) that MATCHES the dense
  PE to 9e-8 on small feeders and builds a 5183-node feeder in 2.3s; results are
  cached per-topology to `pe_cache_v1.pt` (0.01s reload). Small feeders keep the
  exact dense code → E51 reproducibility untouched.
- End-to-end smoke on real feeders: dataset build 72.5s, iterative-solver forward/
  backward on 5k-node graphs (no OOM), eval/decode all work.

**Corpus build + training (running):** full `SMART-DS_1000` build (job 15116719,
64 workers, restarted with prebuilt `--scaler`/`--ranges` to skip the ~40-min
serial fit) → 30000 variants; SMART-DS training (job 15117139, `configs/
smartds_kcl.yaml`, H256 kcl_feedback, normalize_features off) auto-launches via
`afterok` dependency. Watch: unseen V and whether current tracks V (no reactor
artifact). See memory [[dgfm-iterative-solver-design]], [[dgfm-current-error-is-reactor]].

## (minimal_component) Conclusion — superseded by the SMART-DS pivot above
The unseen current wall is **not** fixable by fine-tuning the E51 model. It is a
**voltage-generalization** problem gated by reactor stiffness: current ≈ 2.7·V,
so <1% current needs unseen V < ~0.37% (E51 is 1.7%; seen is 0.7% — the
seen→unseen gap is the barrier). Neither supervising the reactor head (R1) nor
forcing V through reactor physics (R2) moves it. The remaining levers are
fundamental: (a) more training-topology diversity (prior note: "topology
coverage is strongest current lever"), (b) a better-generalizing architecture
(e.g. residual-feedback / learned-iterative-solver for V), or (c) recognizing
these stiff reactors as a minimal_component artifact and validating the pipeline
(exact at truth reactor: 0.175%) on realistic dss/SMART-DS data.

# 2026-07-15 — Decoder finished (3 corpora) + the V metric was lying to us

## 1. Current decoder: DONE, verified on 308,400 samples across 3 corpora
| corpus | samples | decoder WAPE |
|---|---|---|
| SMART-DS_1000 | 100,000 | **6.050e-08** (diffuse fp64 accumulation) |
| minimal_component | 200,000 | **7.859e-10** (0/2000 feeders > 1e-6) |
| dss_data (real IEEE) | 8,400 | **1.078e-06** (82/84 clean, 1 refused) |

Seven bugs, **all in my code, none in the data** — every corpus reproduces
`I = Y@V - Icomp` at truth V to ~1e-14, `build_synthetic_corpus` included. Every
one presented identically: **a current of exactly ZERO, silently**.
- transformer null-space map min-normed the primary ZERO-SEQUENCE to 0
  (grounded-wye/floating-wye). `n^T I = (Yn)^T V` is exact for EVERY n; null-ness
  buys CONDITIONING, not correctness -> take the |U| least-stiff independent
  directions, no threshold.
- `build_recon_ctx` cached per feeder: **variants RETAP the transformers**, so
  Yxfmr is NOT static (edge_index is). Variant 0 was exact BY CONSTRUCTION, which
  is why dissecting feeders kept "proving" the worst feeder clean.
- `mesh_correct` was line-only -> series reactor chords silently 0 (the entire
  3.967e-4 reactor residual). Generalised: for the pi primitive [[A,B],[B^T,D]]
  the through-branch admittance is **(A-B)/2**, no element-type knowledge.
- transformer K uniqueness per-transformer -> open-wye/open-delta banks HALVED;
  global -> transmission buses solved NOTHING. Both are "KCL at a shared node
  gives only the SUM" -> **joint solve per group** of node-sharing transformers.
- `_tree_from_edges` marked roots lazily -> BFS adopted one root as another's
  CHILD (37Bus regulator jumper). Roots pre-marked; non-tree edges split into
  CHORD (real loop -> KVL/mesh) vs BRIDGE (different trees -> joins the joint system).
- ground-touching edges dropped -> 4-wire lines' grounded NEUTRAL stayed 0.
  Ground is now a root, but only SERIES-element conductors may enter the tree.

**KNOWN GAP**: IEEE 30 Bus = a LOOP THROUGH A TRANSFORMER (rank 47 < 56). Needs
mesh analysis extended to transformer branches. Now **REFUSED** loudly, not faked.

## 2. The "4% V error" is the DO-NOTHING baseline  <-- the real headline
`test_vbase.py`, SMART-DS, 743,185 masked nodes:
```
|dv| per node     mean = 4.452e-02
BASELINE dv=0          v_wape = 4.415 %   <- predict V_init, learn NOTHING
BASELINE dv=mean       v_wape = 4.414 %
```
`v_wape` divides by |V| ~ 1.0 pu while the signal |dv| is 0.044 pu, so a null model
scores 4.4% and looks "96% accurate". Added `v_skill = |err|/|dv|` (1.0 = no skill).

## 3. Why: pf is well-posed and LINEAR, but Ybus is unusable for local relaxation
`test_wellposed.py`: mask_pf hides only non-slack V; every Y and Icomp is visible,
so `Ybus @ V = sum(Icomp)`. A direct solve recovers the hidden V to **1e-9..1e-14**
on all 3 corpora -> the information IS there; the task is not ill-posed.
BUT `cond(Ybus)` reaches **1.25e+18**. `test_ladder.py`:
```
feeder                 cond      GJ@10     GJ@60      GS@10     GS@60
ihs0_1247--idt740    1.25e+18  4.23e-02  3.88e+10   2.60e-02  2.58e-02
p18uhs12_1247        8.85e+09  4.07e-02  3.22e+12   3.22e-02  3.20e-02
```
**Gauss-Jacobi DIVERGES; Gauss-Seidel STALLS at 2.6e-2** (no progress 10->60).
Message passing IS local relaxation, so the model sitting at the baseline is the
MATH, not a tuning failure. More width/steps/epochs cannot fix a divergent scheme.

## 4. Implied direction: tree sweeps, not Ybus relaxation
The classical distribution answer is the **backward-forward (ladder) sweep**:
backward = shunt currents from V + tree KCL (**we already have this EXACT**);
forward = `V_child = V_parent - Z_branch @ I_branch` from the slack. Both halves are
O(1)-conditioned TREE accumulations that move information across the whole feeder in
ONE pass, which is exactly what Jacobi cannot do. Open problem for the forward half:
V_sec from V_prim across a transformer without reintroducing a stiff inverse
(the turns ratio lives in the null space of its YPrim).
Also: **minimal_component's dv=0 baseline is 70.1%** (|dv| 0.78 vs 0.044) — ~18x more
voltage signal than SMART-DS and no free ride from a flat start; better V target.

## 5. fp64 corpus vs fp32 model
The regenerated corpus is fp64; the model trains fp32. Cast added at the DATASET
boundary (`DKDataset.__getitem__`) so the reference decoder keeps fp64 inputs.

## 6. PROVEN: the ladder sweep escapes the conditioning wall (architecture answer)
`gridfm/ladfast.py` — splitting `Y_series V^{k+1} = sum(Icomp) - Y_shunt V^k`:
```
feeder                    n      cond   rho_lad     LAD@1     LAD@3    LAD@10     GS@60
p24uhs9                   3  2.56e+00  1.75e-04  1.99e-06  9.41e-15  2.89e-16  1.94e-16
p35uhs0_4                53  4.30e+07  1.89e-02  1.53e-03  4.06e-07  1.23e-11  1.86e-02
p11uhs11                148  1.00e+17  2.79e-02  1.72e-04  3.88e-08  7.57e-10  1.13e-01
p20uhs15                222  4.39e+06  5.17e-02  2.67e-04  3.56e-07  1.42e-13  2.53e-02
p5rhs1                  240  9.98e+14  6.00e-02  2.32e-03  9.76e-08  4.27e-12  1.80e-01
```
**3 sweeps -> 1e-7; 10 sweeps -> 1e-10..1e-13 — on feeders where cond(Ybus)=1e17 and
Gauss-Seidel is STUCK at 1.1e-1 after 60 iterations.** The ladder after ONE sweep beats
GS after SIXTY. Cause: `rho_lad = 1.75e-4 .. 6e-2` << 1 — convergence is set by the
shunt/series admittance ratio (loads ~1e-2 vs lines ~1e6), **independent of cond(Ybus)**.
Confirmed AT SCALE on the largest / worst-conditioned feeders (full test_ladder.py):
```
feeder                    cond     GJ@60     GS@60     LAD@3    LAD@10   rho_lad
ihs0_1247--idt740     1.25e+18  3.88e+10  2.58e-02  1.11e-05  1.07e-09  1.95e-01
p18uhs12_1247         8.85e+09  3.22e+12  3.20e-02  6.40e-06  2.99e-09  1.25e-01
p24uhs0_1247          1.27e+18  1.96e+12  3.29e-02  1.86e-05  3.37e-09  1.99e-01
```
1383 free nodes, cond 1.25e+18: ladder = 1.07e-09 in 10 sweeps while GJ has diverged
to 3.9e+10 and GS is stalled at 2.6e-02. rho_lad ~0.2 on big feeders (0.02-0.06 on
small) => ~0.7 decades/sweep => 10 sweeps takes 4.4e-2 -> 1e-9.
=> The conditioning wall is a property of the SCHEME, not of the problem. A 12-step
network CAN solve pf if each step is a LADDER SWEEP (12 is ~exactly the right depth:
3->1e-5..1e-7, 10->1e-9..1e-13); it CANNOT with local relaxation at any width/depth/data.

NOTE Y_series depends on Y ALONE, not on V -- so its factorization is CONSTANT per
sample and can be precomputed in the dataloader; the ladder step
`V <- As^-1 (b - Ash V)` is then a triangular solve, O(n), exact, and differentiable
w.r.t. V (dV^{k+1}/dV^k = -As^-1 Ash). That is much cheaper than hand-rolling a tree
sweep, and it handles transformers/delta/center-tap automatically because they are
already IN Y_series.
DESIGN QUESTION FOR EMMANUEL: with Y and Icomp both visible, a ladder solver SOLVES
pf classically (that is what the 1e-9 above is). The foundation-model value is then
in the MASKED tasks (se / param / injection), where the ladder should act as the
PRECONDITIONER inside the learned iteration rather than as the whole answer.

**Build plan** (halves map onto what exists):
  backward = shunt currents from V (physics decode) + tree KCL -> branch currents
             **ALREADY EXACT** (dk_tree.reconstruct_full, 6e-8 / 7.9e-10 / 1.1e-6)
  forward  = V_child = V_parent - Z_branch @ I_branch, accumulated from the slack
             (the missing piece; Z from the same (A-B)/2 formula mesh_correct uses)
Open: V_sec across a TRANSFORMER without a stiff inverse -- the turns ratio lives in
null(YPrim), the same null space the current decoder already exploits.

## 7. CAVEAT — the ladder is NOT universal (measured on 648 samples, not 8 feeders)
`gridfm/test_ladder_all.py` (sparse splu, 120 feeders x 2 variants per corpus):
```
corpus              start(flat)  err30 median   reached 1e-6   median sweeps   DIVERGED
SMART-DS_1000        5.830e-02     1.962e-09     240/240            4            0
minimal_component    5.115e-01     2.292e-09     215/240            2           25  (err30 ~1e190)
dss_data             3.360e-01     3.431e-10     166/168            4            2  (+1 singular)
```
The "3 sweeps -> 1e-7" headline came from 8 hand-picked feeders. Broadly: it is
**perfect on SMART-DS (240/240, median 4 sweeps)** but **~10% of minimal_component
DIVERGES** (err30 up to 1e+190). Those all have base ~1.0-1.25 (large dv) = the
STIFF-SHUNT feeders (reactors, cf. [[dgfm-current-error-is-reactor]]): there
rho_lad > 1 and the Jacobi-style splitting blows up. Fixable with under-relaxation /
damping (V <- V + w(V_new - V)) or a stronger splitting, but **the honest claim is
"works on 96% of samples, needs damping for the stiff tail", NOT "solves everything"**.
`H__e0d8e0f2725c`: Y_series is exactly singular -> that feeder needs the full Ybus.

## 8. The V loss is SWAMPED — the model CAN learn V (partial rebuttal of §3)
V-ONLY probe (`--w-i 0 --w-kcl 0`) on minimal_component:
```
ep001  V skill unseen = 0.575      (dv=0 baseline = 68.77%)
ep002  V skill unseen = 0.503      <- 2x better than doing nothing
```
vs the mixed-loss probe on SMART-DS, which sits at v_skill ~1.0 (0.98-1.33 over 6
epochs). So the model is NOT incapable of learning V: with V as the sole objective and
real V signal, it gets skill 0.5 in 2 epochs. **`w_v*v_mse ~ 8e-3` vs `w_i*i_mse ~
O(1-7)` -> the current loss outweighs the V loss ~100-800x**, so the mixed-loss model
was largely ignoring V. Both effects are real (conditioning AND loss balance); the
earlier "message passing cannot solve pf by construction" was over-claimed.
TODO: SMART-DS V-only probe to finish the separation; then reweight (normalise each
loss term by its own scale) rather than hand-tuning w_v.

## 9. Ladder splitting variants — no clean winner (measured, 648 samples)
Folding the shunt's own DIAGONAL into the solve matrix (`M = Y_series + diag(Y_shunt)`;
a grounded reactor/cap is purely diagonal so its coupling vanishes; a diagonal add does
NOT break the tree structure, so it is still realizable as a sweep):
```
                     plain ladder            + diag(Y_shunt)
minimal_component    25/240 diverge          4/240        <- rescues the stiff tail
SMART-DS_1000         0/240, median 4 swps   0/240, median 3
dss_data              2/168 diverge          10/168 NOT converged (max 4.3e-2)
                      median 3.4e-10         median 1.7e-12, median 1 sweep
```
Better MEDIAN, worse TAIL on dss_data: 8 feeders that converged now creep instead.
**No clean winner — the right splitting is feeder-dependent.** Do not ship either as
"the" answer. Note `M = Ybus` itself converges in ONE step (it is the direct solve),
and Y_shunt's only off-diagonals are WITHIN a bus (delta/multi-terminal shunts), so a
BLOCK tree sweep over 4-conductor bus blocks would realize M = Ybus exactly. That is
probably the real design: block-tree sweep, not a scalar ladder. **Emmanuel should
decide** — it is an architecture choice, not a bug fix.

## 10. V-only probe, final: the cause is CORPUS-DEPENDENT
```
                            ep1     best      dv=0 baseline
SMART-DS,  mixed loss      1.129    ~1.0        4.22%
SMART-DS,  V-only          0.933    0.927       4.22%   <- loss fix barely helps
minimal_component, V-only  0.575    0.278/0.39  68.77%  <- learns well
```
So BOTH causes are real and they split by corpus. minimal_component: the V loss was
simply swamped -> fix the weighting (`--norm-loss`, implemented). SMART-DS: even with V
as the SOLE objective the model stays at skill ~0.93, consistent with dv being only 4%
of |V| and cond(Ybus)=1e18 -> that one needs the architecture (ladder/block sweep).

## 11. RETRACTION — `--norm-loss` is NOT a fix; the "swamped V loss" was wrong
The control I had not run (minimal_component, UNNORMALISED mixed loss) settles it.
minimal_component, 12 epochs, V skill unseen (best):
```
CONTROL (unnormalised mixed, w_v=10 w_i=1)   0.674 .. 0.387 0.400 0.402   -> ~0.39
norm-loss + family-MEAN                      0.713 .. 0.412 0.415         -> ~0.41
norm-loss + family-SUM                       0.905 .. 0.545 0.513         -> ~0.51
V-only (w_i=0)                               0.575 .. 0.376 0.391         -> ~0.37
```
**The ORIGINAL loss already reaches 0.40 — as good as V-only (0.37). Normalising made
it slightly WORSE.** So minimal_component was never swamped; the arithmetic said so
(mc dv~0.78 => w_v*v_mse ~ 6, comparable to i_mse ~1-7) and I should have checked
BEFORE claiming the imbalance explained the corpus difference. On SMART-DS the 100-400x
imbalance is real but is NOT the cause either: deleting the current loss entirely
(V-only) moved skill only 1.0 -> 0.91.

**=> Loss balance is a MINOR factor. Do not spend more time on weighting.** What
separates the corpora is the V SIGNAL SIZE + CONDITIONING, under every loss config:
```
                   dv=0 baseline   cond(Ybus)   best V skill (any loss config)
minimal_component      70.1%        1e6..1e10        ~0.39
SMART-DS_1000           4.4%        1e9..1e18        ~0.91
```
`--norm-loss` is kept as a flag (off by default) but it is NOT the answer. The
architecture (ladder / block-tree sweep) is the live lever, per sections 6-9.

## 12. Excluded `W` (de-energized) — dss_data now 0/83 feeders above 1e-6
Emmanuel recalled "a network in the dss-data that is not good". Found it by measuring
V on the BUILT corpus (`gridfm/scan_dead.py`):
```
  dead%   nodes    medV    maxV  feeder
   92.1     762  0.0000  1.0000  W__8d6d9ac2ecbe            <- 702/762 nodes at V=0
   18.2      11  0.9968  1.0000  case3_balanced_battery_3ph_en   (grounded neutrals, fine)
    2.7     111  0.9879  1.0000  H__e0d8e0f2725c                 (grounded neutrals, fine)
```
OpenDSS reports `converged=True, iterations=2` for W regardless — **a converged solve is
NOT evidence of a usable network**, which is how it passed curation. I had dismissed W's
2e-06 as a "metric artifact" because |I| = 7e-08; the currents are ~0 BECAUSE 92% of the
network is dead. That was a DATA defect explained away as a measurement quirk.
Moved to `data/excluded/de_energized/W` (+ W_REASON.md); removed from training_data.

**Result: dss_data 1.078e-06 -> 1.664e-07, feeders >1e-6: 2/84 -> 0/83.**

NOT excluded — `IEEE 30 Bus` compiles, converges in 2 iters, V pu 0.861..1.060, no NaN.
The segfault in arranged_validation belongs to a DIFFERENT copy (data/data-new/OpenDSS/
Distrib__IEEETestCases__IEEE_30_Bus__Master). It is a legitimate meshed transmission
network and the ONE feeder the decoder refuses, so deleting it on a name match would
have quietly erased our own known gap. Kept deliberately. If transmission is out of
scope that is a SCOPE call, not a data-quality one.

## FINAL DECODER STATE
| corpus | samples | WAPE | feeders >1e-6 |
|---|---|---|---|
| SMART-DS_1000 | 100,000 | 6.050e-08 | 0/1000 |
| minimal_component | 200,000 | 7.859e-10 | 0/2000 |
| dss_data | 8,300 | 1.664e-07 | 0/83 (+1 refused: IEEE 30 Bus) |

## 13. IEEE 30 Bus: the refusal is NOT "a loop through a transformer"

Carried that diagnosis for a while. It is WRONG, and measuring the null space killed it.

`dbg_null.py` dumps the null basis of the joint transformer/bridge system:

```
group: 86 unknowns | rows: kcl=18 cut=18 bridge=15 dirs=41 | cond=2.93e+16
  NULL space: 9 modes
    line         100.00% of null weight
    transformer    0.00% of null weight
    modes live on exactly 4 line components: c18 c19 c20 c21
```

**0.00% on any transformer winding.** The 9 undetermined DOF are pure LINE circulating
currents that close through BRIDGES; the transformer is merely the ROOT of the component
a bridge lands in, and is never in the loop. So this needs no transformer loop model, no
turns ratio in the KVL, and no impedance form for a winding -- which does not exist, since
YPrim is singular. Every hour spent on "how do I write KVL through a tap changer" was
spent on a non-problem.

Corollary: `transformer WAPE 1.07` was never indeterminacy. The transformers are
DETERMINED. That number was pinv least-squaring an inconsistent mid-Jacobi rhs and
smearing the error across the determined unknowns too. A rank-deficient system does not
politely confine its error to the null space.

### What fixed 6 of the 9: KVL rows INSIDE the system (`build_kvl_rows`)
One row per BRIDGE that is a chord of the mesh forest: `(mᵀZ) f = 0` around its
fundamental loop -- currents and impedances only, never V1-V2, so no Y@V stiffness.
Fed INTO the joint system (not bolted on after), so it becomes full rank, pinv is a true
inverse, and nothing smears. **rank 77 -> 83 of 86.**

Post-hoc `mesh_correct` cannot substitute for this: measured IEEE 30 Bus line 3.2e-01 /
xfmr 1.07, and it REGRESSED 37Bus 6.6e-11 -> 1.1e-07. Do not retry it.

### The remaining 3, and why they are exactly 3
The line graph is PER-PHASE DISCONNECTED (`_series_edges` pairs same-slot conductors
only), so each phase is its own component set with its own slack component -- and
`build_xfmr_system` SKIPS the cut-set of any component containing the slack (its vsource
is unknown until the end). That is one lost row per phase = **3**. Per phase the bridge
cycle space supplies only 2 loops (5 bridge conductors, mesh forest uses 3 as tree
edges), which is exactly the 6 rows found. 2/phase found + 1/phase missing = 9.

The 3 survivors live on c19/c20 -- bridges that are mesh-TREE edges, so they close no
loop against that forest and no bridge-chord row can reach them.

### The identified close (NOT yet done)
Two mechanisms currently own loop currents and they overlap: the joint system owns bridge
conductors, `mesh_correct` owns pure-line chords, and neither sees the other's unknowns.
Unify: treat EVERY non-tree line edge (bridge AND rooted-chord) as an unknown conductor
pair with its `I1+I2 = charging` row, add one KVL row per mesh chord, and retire
`mesh_correct`. Radial feeders have no non-tree edges, so they are untouched by
construction. This subsumes `_split_parallel_lines` (the L=1 case) too.

STATUS: IEEE 30 Bus is STILL REFUSED (3 DOF short). Refusing beats returning
silently-zero currents, which is how all seven earlier decoder bugs presented.

## 14. RETRACTION: the cut-set rows. And a harness bug that framed the decoder.

Re-validating dss_data showed 10/74 feeders > 1e-6, some at WAPE 0.98, against a baseline
of 0/83. Two independent causes, BOTH mine, neither one a decoder bug.

### (a) test_all.py fed the decoder zeros and then scored them
`test_all.py` built `cur` from `SHUNT_STORES` only and gave every other store ZEROS. But a
SHUNT-connected reactor is physics-decoded and `reconstruct_full` KEEPS what it is handed
-- so its current stayed at exactly 0 and scored `WAPE 1.000e+00`. `test_mc.py` always
decoded `SHUNT_STORES or AMBIG_STORES`; test_all did not. Invisible until now only because
the OLD dss_data had no shunt reactors -- the rebuilt corpus does.

  3ph_matrix_shunt   9.792e-01 -> 3.430e-13
  1ph_shunt_ground   9.014e-01 -> 1.847e-12
  5bus_shunt_reactor 5.486e-01 -> 8.471e-13
  3ph_delta_shunt    8.447e-01 -> 3.168e-12

An exact `1.000e+00` is the silently-zero signature. Read it as "this current is 0",
never as "the model is bad here".

### (b) the cut-set rows are WRONG -- retracted
Bisected trans_3w_center_tap across the three commits:

  17839ee (validated baseline)  TOTAL 6.508e-11
  1c278a3 (cut-sets restored)   TOTAL 6.611e-01   transformer 8.3e-01, vsource ZERO
  removing them restores:  center_tap 6.5e-11 | IEEE123 1.1e-09 | 13Bus 4.5e-09 | 37Bus 5.7e-11

Making them GREEDY (add only if rank-increasing) did NOT fix it => a rank-INCREASING
cut-set row is itself wrong, so the equation does not hold on networks with grounded /
center-tap windings. Suspected cause: the row sums KCL over a component's nodes, but node
0 (GROUND) belongs to NO component, so an element with one terminal in the component and
the other at ground never cancels and is silently counted as known.

They also bought nothing: IEEE 30 Bus was refused with AND without them. And the KVL rows
do not need them -- ALONE they take IEEE 30 Bus from 15 DOF short (rank 50<62 + 25<28) to
**3** (rank 53<56, one group), better than cut-sets ever managed.

Kept: never hand pinv a row that does not raise the rank. The rhs comes from a
half-converged Jacobi sweep, so a redundant row carries a DIFFERENT rhs mid-iteration and
pinv least-squares the disagreement into unknowns that were already exact.

### State after the retraction
| corpus | samples | WAPE | feeders >1e-6 |
|---|---|---|---|
| dss_data | 8,300 | 1.657e-07 | 0/83 (+1 refused: IEEE 30 Bus, 3 DOF short) |
| minimal_component | 200,000 | 7.859e-10 | 0/2000 |

Lesson: I restored 131 uncommitted lines from an attic snapshot and committed them as
"WIP -- not a fix". They were worse than not a fix; they were a regression, and only a
full re-validation caught it. Unvalidated work is not neutral.

## 15. new_dss_data (864 feeders): first full validation

Corpus totals now (training_data/aa.md): SMART-DS_1000 1000 feeders/100k, dss_data 84/8.4k,
minimal_component 2000/200k, new_dss_data 864/86.4k = **3948 feeders / 394,800 snapshots**.

First sweep of new_dss_data (849 feeders reduced, 75,530 variants):

```
  AGGREGATE WAPE = 2.538e-05      (line 2.076e-05, vsource 1.103e-04, xfmr 2.716e-07)
  feeders mean WAPE > 1e-6: 93 / 893      refused: 1
```

Bucketed (`|I|` tells the three apart, they are NOT the same problem):
```
  |I| ~ 0 (DE-ENERGIZED, like W): 27   <- data defect, exclude
  WAPE ~1.0 with real |I|:         0
  genuinely partial error:        66   <- real decoder gaps
```
Family incidence among the 66: vsource 66, line 39, transformer 30, reactor 4.
vsource is bad in ALL 66 -- it is computed LAST by KCL at the slack, so it is a SYMPTOM
(everything upstream lands there), never the cause. Do not chase it.

The genuine ones cluster by NAME: `TestAuto`, `AutoTrans`, `AutoHLT` = OpenDSS
AUTOTRANSFORMER cases, a winding topology the corpus never contained before.
  * AutoTrans/AutoHLT: ~1.5e-05 with |I| = 1.1e+04 (large-|I| transmission cases)
  * TestAuto: 3.99e-01 -- the worst genuine class

TestAuto__32ce4da47eca dissected:
```
  line 9.987e-01 (|I| 8.9e-03)   reactor 2.236e-04   transformer 9.613e-11   vsource 9.983e-01
  bridges: [('reactor', 0, 4, 8)]     <- the BRIDGE IS A REACTOR, not a line
  group unknowns: transformer(0,{0,1,4,5}) + reactor(0,{0,4})
  line c0: rec_zero=False    <- WRONG, not silently zero
```
The transformer is EXACT, so the autotransformer's Y is handled. The failure is the
SERIES REACTOR acting as a bridge: `build_kvl_rows` only ever considers `ltree["bridges"]`
whose loop it closes with LINE impedance, and `_bridge_inj`/bridge rows assume the bridge
carries line charging (`I1+I2 = Yh(V1+V2)`) -- a reactor has no `Yh` entry in that table.
NOT yet fixed. Note this is a DIFFERENT bug from IEEE 30 Bus (which is pure-line loops).

STATUS: full re-validation of all four corpora launched with the current decoder
(`gridfm/val_final.sbatch`, fresh `runs/final_<corpus>` dirs -- the older runs/ dirs
straddled the cut-set retraction and are untrustworthy).

## 16. EXCLUSIONS from new_dss_data, and the corpus totals

Decided on the DATA (node voltages + stored |I|), never on a WAPE score or a name match
(`gridfm/scan_dead.py` prints an auditable verdict per feeder). 49 moved, 815 remain:

  data/excluded/de_energized/  3   66.7% of nodes at V=0 with converged=True (the W signature)
  data/excluded/no_circuit/   46   |I| = 0.0 EXACTLY across every family

The `no_circuit` 46 are OpenDSS FEATURE-TEST scripts swept up by the collector because they
compile -- CableParameters, XYCurvetest, TextTsCable750MCM, and old IEEETestCases archive
copies. A source at 1.0 pu with nothing connected. They are invisible in a WAPE column
(0/0 scores 0.0 and PASSES), so they were never in the 93 failures -- they are 4,600
no-op training samples that teach nothing. Used |I| < 1e-12 (strict zero) not < 1e-3, to
stay conservative: a genuinely tiny-but-live feeder is kept.

Note the three failure shapes are distinguished ONLY by |I|, never by WAPE:
  WAPE 1.0, |I| ~ 1e-8   -> DEAD circuit (data)      -> exclude
  WAPE 1.0, |I| real     -> silently-zero current    -> decoder/harness bug
  WAPE 0.0, |I| == 0     -> empty circuit (data)     -> exclude, and it PASSES silently

## 17. OPEN, in priority order (handoff)

1. **PORT the decoder into the model** -- the biggest single win available, ~7 orders of
   magnitude. `dk_model._completed_currents` calls `reconstruct_vectorized`: MEASURED
   5.942e-01 on SMART-DS vs 4.588e-08 for `reconstruct_full` (`gridfm/dbg_model_decoder.py`).
   NOT charging (SMART-DS |charging|/|I_line| = 0.0%): it is the transformers (64% wrong,
   4.6% of |I|) propagating into the lines (92% of abs err). Needs the joint-transformer
   pinv machinery batched + differentiable, with xfmr maps rebuilt PER SAMPLE (variants
   retap; a stale ctx decodes at the wrong ratio -- `reconstruct_full` raises on `yref`).
   Topology half is reusable across variants via `build_recon_ctx(data, topo=...)`.
2. **AUTOTRANSFORMER gap** (66 feeders in new_dss_data): a SERIES REACTOR acting as a
   bridge. `build_kvl_rows` closes loops with LINE impedance only, and the bridge row
   assumes line charging `I1+I2 = Yh(V1+V2)` which a reactor has no entry for. The
   autotransformer Y itself is FINE (transformer 9.6e-11).
3. **IEEE 30 Bus**: 3 DOF short. Unify the two loop mechanisms (see 13).
4. **VOLTAGE**: untouched this session. The 4% is the dv=0 null baseline; ALWAYS report
   v_skill. Loss weighting is a MEASURED dead end. cond(Ybus)=1.25e18 caps local relaxation
   at skill 0.59; the ladder sweep converges on 96% of samples but diverges on the
   stiff-reactor tail. Architecture is the only live lever.

## 18. FIXED the autotransformer failures: reactor-as-bridge (new_dss_data 4 x 0.40 -> 2e-10)

The 4 genuinely-broken new_dss_data feeders (TestAuto, WAPE 0.40) were autotransformers
with a SERIES REACTOR bridging two rooted components -- a reactor BRIDGE, a topology no
prior corpus had. TWO bugs, both from code that silently assumed every bridge is a LINE:

1. Bridge-row RHS looked up the LINE charging table indexed by the bridge's own comp id
   -> garbage for a reactor. Fixed with `_series_ycm` (common-mode CM1=A+Bᵀ, CM2=B+D from
   the element's OWN primitive; line->Yh(V1+V2), series reactor->0, no type branch).
2. DOUBLE COUNT: `binj` injected the reactor bridge current, but `build_q` already injects
   it (reactor in line_stores) -> line current doubled (0.006 vs 0.003). `binj` now covers
   only bridges whose store is NOT in line_stores (i.e. LINE bridges).

Result (all four corpora re-validated, fresh dirs, NO regression):
| corpus | before | after | mean>1e-6 |
|---|---|---|---|
| new_dss_data      | 2.183e-05 | **2.964e-06** | 76/814 |
| dss_data          | 1.657e-07 | 1.657e-07 | 0/83 (+IEEE30 refused) |
| SMART-DS_1000     | 6.050e-08 | 6.050e-08 | 2/1000 |
| minimal_component | 9.293e-10 | 9.293e-10 | 0/2000 |

The remaining new_dss_data tail (76 feeders ~1e-5, vsource-dominated) is a BENIGN floor,
NOT a structural gap: identical at JACOBI 6/20/50 (so not under-convergence), every other
family machine-exact, and the corpus AGGREGATE is 2.96e-6. vsource is computed LAST by KCL
at the slack and carries little current on these PV-heavy feeders, so a tiny ABSOLUTE
mismatch inflates into ~1e-5 RELATIVE. Not worth chasing versus the port (5.9e-01).

Fast parallel probe added: `gridfm/dbg_many.py` (many feeders, one srun, worker pool).

## 19. VOLTAGE: diagnosis + the concrete architecture plan (grounded, not a fluke)

MEASURED state (probe logs, minimal_component): best unseen v_skill ~0.37-0.39; dv=0 null
= 68.77%. The model captures ~60% of voltage variation, then plateaus. v-only loss (0.37)
~= mixed v+i+kcl loss (0.39) -- so the broken decoder's gradient into V is NOT the cap
(it hurts the i metric, not v). The cap is ARCHITECTURAL.

Why pure message passing plateaus: `dk_model.forward` is a feedforward V-predictor -- N
GRU message-passing steps, then predict dv ONCE. Local MP is relaxation: one hop of
correction per step. With cond(Ybus)=1.25e18 the corrections propagate O(diameter x cond)
slowly, so no fixed step count converges. The code comment "no residual to feed back"
because tree-KCL satisfies KCL structurally is the TRAP: KCL is trivially satisfied, yes,
but the CONSTITUTIVE/KVL law is NOT, and that is the signal being thrown away.

THE PLAN (do in this order; each is a prerequisite for the next):

A. PORT the exact decoder into the model (`_completed_currents` -> reconstruct_full, not
   reconstruct_vectorized). MEASURED 5.9e-01 -> 4.6e-08. Mechanism: precompute per-feeder
   ctx = build_recon_ctx(base) in DKFeeder (alongside self.plan); rebuild the Y-dependent
   xmaps PER SAMPLE in collate (variants retap transformers -- a stale ctx decodes at the
   wrong ratio; reconstruct_full raises on a stale yref). Topology half is reused across
   variants. Batch the pinv-apply (it is per-group, small). Verify: model i_wape drops to
   ~1e-7 on a smoke batch. This alone fixes every current-based metric.

B. KVL RESIDUAL as a feedback feature (this is the voltage lever). Once currents come from
   V exactly, per node compute r = V_pred - (V_parent - Z @ I_decoded) -- the backward-
   forward-sweep residual. Z @ I is WELL-conditioned (not Y @ V), so r is a clean, finite
   error signal that local MP structurally lacks. Feed r into the next MP step's node
   input. This turns the model into a LEARNED ladder sweep, which (memory) converges on 96%
   of samples; the GNN's job shrinks to correcting the stiff-reactor tail where the bare
   sweep diverges. Expect v_skill well below the 0.59 relaxation cap.

C. Only if B still tails: the stiff-reactor 4% samples. Predict branch-current / v-drop as
   a first-class state (see [[dgfm-line-current-unrecoverable-from-V]]) so the stiff series
   drop is represented directly instead of inferred from V.

NOT started tonight -- A and B are multi-session builds needing training runs to verify,
and a half-built architecture that regresses training is worse than a written plan. The
decoder is now EXACT and complete across all four corpora, which is exactly the
precondition A/B need.

## 20. THE PORT: batched transformer solve proven, half done

Per-sample reconstruct_full in the model was benchmarked at 350ms/sample = ~700s/forward
(these SMART-DS feeders average 220 transformers). Non-starter. But MEASURED: SMART-DS
transformers are ALL ISOLATED (groups == xfmr count, sizes 6/7/8, zero bridges/kvl), so the
joint-pinv system is an independent small solve per transformer -> batches as a per-
size-class bmm.

DONE + VERIFIED:
* `_pack_isolated_xfmr` stacks same-signature isolated groups; `_apply_xfmr_batched` solves
  each class with one bmm, falls back to `_apply_xfmr_groups_loop` for non-isolated groups
  (IEEE30 bridges, autotransformer reactor-bridges, IEEE9500 mixed-dir-length coupled banks).
* Bit-identical to the scalar path: max|scalar-batched| = 5e-21.
* reconstruct_full now uses it (packed cached in ctx). Regression at scale UNCHANGED:
  dss_data 1.657e-07 (0/83), new_dss_data 3.026e-06, minimal_component 9.293e-10.

REMAINING (model integration -- now well-defined, de-risked):
1. `batch_xpacked(xpacked_list, node_off, comp_off)`: merge per-feeder packed structs with
   PyG's node/comp offsets (mirror `batch_plans`). KN/DND += node_off; CI += comp_off.
2. In a `reconstruct_model` (or extend reconstruct_vectorized): after the batched unified
   tree sweep places shunt+line currents, call `_apply_xfmr_batched` on the BATCHED graph
   with the merged packed (samples are node-disjoint, so the global _full_residual is
   per-sample-correct), then re-sweep lines. This is the Jacobi loop, batched.
3. Wire into `dk_model._completed_currents` behind a flag (default OFF so training never
   breaks); precompute ctx per DKFeeder (SMART-DS Y is static across variants, so one ctx
   per feeder -- confirmed 0/40). Verify: model i_wape on a smoke batch drops ~5.9e-01 ->
   ~1e-7, then the i_term / kcl gradient into V becomes physical (prerequisite for the
   voltage KVL-residual feedback, section 19).
Also add "reactor" to dk_model.PHYS_DECODE (it is in SERIES_STORES but not PHYS_DECODE, so
the model zeroes shunt reactors -- the same silently-zero bug fixed in test_all.py).

## 21. BATCHED RECON IS WIP AND **NOT CORRECT** -- do not wire into the model

`batch_recon_ctx` / `_offset_group` / `_merge_ltree` / `batch_xpacked` are committed but
**DO NOT reproduce per-feeder reconstruct_full yet**. Verified with
`gridfm/verify_batch_recon.py` (builds a real PyG Batch, runs both, diffs per store):

```
 4 small radial feeders   max|batched - per_feeder| = 0.000e+00     EXACT
24-30 mixed feeders       max 2.66e-02 , WAPE vs truth 6.7e-02      WRONG
```
Failures concentrate on HIGH-TRANSFORMER feeders (xfmr 130-470). Root cause NOT found.
Nothing calls it, so training is unaffected today -- but it must be exact before the port.

NOTE: commit 757f241 ("Reorganize probes and add perturbation screening tools") was made by
the **Codex agent**, not by this work, and it swallowed these changes under a message that
does NOT record this WIP status. Two agents are committing to this repo; do not read that
commit message as a claim that batched recon works.

REAL fixes found on the way (kept, per-feeder verified):
* `_full_residual` zeroed only GLOBAL node 0. A PyG batch has ONE GROUND PER SAMPLE at
  node_off[i]; the others' residuals leaked into the xfmr KCL rhs. Now takes `grounds`.
* `mesh_correct`'s live test `e[2] != 0` carries the same ground-sentinel assumption ->
  now takes `grounds`.
* `_merge_ltree` must merge the ROOTED-forest fields (chords / parent_edge / parent_node /
  depth) that mesh_correct actually reads -- NOT the mparent_* cycle-space ones.
* `_apply_xfmr_batched` can zero unknowns from the packed CI/SI (no group objects needed).

DESIGN (no silent skipping, per user): the batched path must cover isolated transformers
(batched bmm), coupled banks (offset leftover groups) AND line loops (merged mesh tree).
Transmission BRIDGES raise loudly instead of silently falling back.

PER-FEEDER DECODER VERIFIED UNAFFECTED by all of the above (defaults preserve old
behaviour): 37Bus 5.706e-11, trans_3w_center_tap 6.508e-11 -- both exactly at baseline.

gridfm/ tidied 61 -> 29 files; 32 one-off probes moved to gridfm/probes/ (README there).

### 21b. Narrowing the batched-recon bug (in progress)

Measured with `gridfm/probes/dbg_batch_iso.py` (explicit feeder dirs, no corpus scan):

```
batch of 1  (p1rdt10307: 472 xfmr, 471 groups, 1 COUPLED group, 0 chords)  -> EXACT
batch of 2  (idt740: 113 xfmr + 3 CHORDS)  +  (p1rdt10307)                 -> EXACT
batch of 6  (first 6 alphabetical)  -> ONLY feeder1 wrong:
     line        diff 1.07e-02  227/483 rows   (values wrong, NOT zero)
     transformer diff 4.55e-03   67/119 rows
     vsource     diff 1.07e-02    1/1
```
So: the merge + apply are CORRECT in isolation (batch-of-1 exact, including a coupled
group), and mesh + coupled merging work across 2 feeders. The bug is CROSS-FEEDER and hits
only SOME feeders -- not a uniform offset error.

LEAD (unverified, probe queued): `dk_physics.store_size()` falls back to `I_r_bus1_pu` when
`{prefix}_r_pu` is absent:
```
for f in (f"{prefix}_r_pu", "I_r_bus1_pu"):
    if f in st: return st[f].shape[0]
```
but PyG offsets a store by `st.num_nodes`, which both dk_data.__getitem__ and the probes set
from `{prefix}_r_pu` ONLY. If any store takes the fallback path, `soff` (from store_size)
advances while PyG's offset does not -> later feeders index the wrong rows. That matches the
symptom exactly (fine at N<=2, one feeder corrupted at N=6). Check: runs/p2.sbatch prints any
store where store_size != the {prefix}_r_pu row count.

If that is NOT the cause, the next probe is `gridfm/probes/dbg_offsets.py`, which diffs MY
merged `ctx["inj"]` against `_inj_index(batched_graph)` per store -- reconstruct_full uses
BOTH (build_q uses ctx["inj"]; _full_residual recomputes _inj_index), so any disagreement
between them corrupts the residual.

## 22. Batched-recon bug SOLVED: PyG mis-offsets edges for stores absent in some samples

Root cause of the cross-feeder corruption, and a SILENT PRE-EXISTING TRAINING BUG.

PyG's hetero collate accumulates edge_index offsets ONLY over samples that HAVE a relation.
A feeder without `pvsystem` is skipped in that cumsum, so LATER feeders' pvsystem edges are
offset too little and point INTO an earlier feeder's node range.

```
pvsystem   node: 494 differ   mine=[58,0,2992]  pyg=[58,0,1405]   <- feeder2's PV -> feeder1's nodes
storage    node: 182 differ   mine=[66,0,3888]  pyg=[66,0,1963]
load / line / transformer / vsource / capacitor: OK  (present in EVERY feeder)
```
Only the NODE index, only for stores ABSENT somewhere -- matching the bisect exactly
(batch-of-1 and -2 EXACT; N=6 corrupted feeder1, the store-less one).

FIX `dk_physics.ensure_batch_schema(samples)`: one schema for every sample (empty store +
empty edge_index where absent) so each relation's inc contributes (0, n_node). MUST be
batch-level (only there is the union of keys knowable; PyG raises KeyError if an empty store
lacks a peer's key). Refuses to zero-fill a POPULATED store missing a key. Wired into
make_dk_collate (fixes TRAINING) and the recon batching path.

| check | before | after |
|---|---|---|
| batched vs per-feeder (20 feeders) | 2.66e-02 | **1.084e-19** |
| batched WAPE vs truth | 6.7e-02 | **3.227e-08** |
| pvsystem/storage node offsets | 494/182 wrong | **all OK** |

## 23. Model SHUNT REACTOR silently-zero: BOTH fixes required
`PHYS_DECODE` lacked "reactor" AND `build_tree_plan` let shunt reactors become TREE EDGES
(so the sweep overwrote the decode). Either alone is a NO-OP -- measured. Together:
`reactor 1.000e+00 -> 4.318e-16`, feeder TOTAL `9.955e-01 -> 1.249e-01`.

## 24. PORT: reconstruct_full is now dtype/device aware -- 2 blockers remain
Was CPU/fp64-only (float64 allocations, no device=) so it could not run in a GPU forward.
Now every allocation follows the currents. `recon_ctx_to()` moves a ctx once per batch.
`_full_residual` takes cached `inj` (it re-ran terminal_slot EVERY Jacobi iteration).
VERIFIED a true no-op on the reference -- 6/6 exactly at baseline.

REMAINING BLOCKERS before wiring into the model:
1. **`mesh_correct` is NOT autograd-safe**: `_branch_f` does `f[k] = complex(0.5*(Or[..]-Or[..]))`
   -- python `complex()` on a tensor DETACHES the graph, so no gradient flows through the loop
   correction. Affects the 1.6% chord feeders. Needs a vectorized, tensor-native rewrite.
2. **Speed**: batched recon measured ~83ms/feeder on CPU fp64. The `inj` cache removes a big
   chunk; must re-measure on GPU/fp32 before it can sit in a training step.
Only then: wire `_completed_currents` -> reconstruct_full behind a flag, then the
KVL-residual feedback for voltage (section 19).

---

## 25. Decoder port + the plateau diagnosis (2026-07-17)

### Corpus is finally trustworthy
- 3190 distinct networks x 100 variants = 319,000 snapshots; SMART-DS 1000 (true 1000).
- 151 duplicate NETWORKS retired (IEEE123 x12, ckt5 x10, SMART-DS scenario twins). Identity
  is the unperturbed baseline SOLVE (dynamic row 0), NOT DSS text: text over-merged (5
  civanlar copies hashed identical, one had 2664 vs 2360 columns -- a different circuit)
  and under-merged (missed 51 dupes + all 5 cross-corpus groups).
- **PF determinacy 100% on all four corpora.** DG_FM_Training's headline blocker
  ("smartds_gso12 ~0%, grounding/neutral export") is GONE. |Ybus.Vtrue - Icomp| = 1e-09.

### The exact decoder is now IN the model  (6.7e-01 -> 3.4e-06)
End-to-end through DKFeeder -> collate -> DKSolver on truth V:

    store         reconstruct_vectorized   reconstruct_full
    line                       6.758e-01          3.346e-06
    transformer                8.430e-01          7.556e-06
    vsource                    1.000e+00          2.769e-06   <- exactly 1.0 = silently ZERO
    TOTAL                      6.731e-01          3.437e-06

Gradients flow (34280/34376 nonzero). Affordable via topology caching in DKFeeder
(2.09s -> 0.142s per variant on a 9710-node/517-xfmr feeder, identical WAPE).
Confirmed in training: unseen I now TRACKS V (3.6% vs 3.85%) across every family.

### The v_skill plateau is the ITERATION OPERATOR (three refuted assumptions)
1. NOT receptive field. Max BFS depth from slack = 4 on a 7092-node feeder; steps=12
   reaches 100%. But ground (node 0) is a hub of degree 985-5815 -- that 4 is a shortcut
   through one node's hidden vector (oversquashing). True electrical depth 49-113; tree
   depth 47-110.
   => attention / a global node adds NOTHING: ground already is one.
2. NOT conditioning. cond(Ybus) hits 2.74e18, but:

     feeder  cond      Gauss-Jacobi@60   Gauss-Seidel@60   LAD@3     LAD@10    rho_lad
     P1R     4.26e+09         3.03e+10          2.20e-02   2.39e-07  2.85e-09    0.049
     P31U    2.74e+18         2.82e+12          2.07e-02   1.23e-05  4.47e-09    0.146

   Gauss-Jacobi DIVERGES -- and a generic message-passing step IS a Jacobi relaxation on
   Ybus. That is the ceiling. My old note "cond=1e18 caps local relaxation at skill 0.59"
   measured the right thing for the WRONG operator.
3. The LADDER reaches ~1e-09 in 10 sweeps because rho(Yser^-1 Ysh) = 0.03-0.18,
   independent of cond. "Ladder diverges on the stiff-reactor tail" does not apply:
   reactors are 99% of minimal_component and 0% of SMART-DS.

=> A 12-step net CAN hit machine precision IF its step is a SWEEP, not a relaxation.

### Tree-only forward sweep: works, does not yet scale  (OPEN)
Backward half is already exact in the model. Forward half V2 = B^-1(I1 - A V1):
- case3_delta_gens (no transformer in path): sw1 2.9e-03 -> sw6 1.0e-08 -> **sw12 1.23e-12**
- 123Bus/H (transformers): frozen at the flat start, because TREE_STORES is
  (line,reactor,capacitor) -- transformers are NOT tree edges, so the tree is a forest
  rooted at the slack AND every xfmr secondary.
- Pushing V through transformers DIVERGES (6.6e+06 -> 2.6e+99). Two hypotheses tested and
  BOTH REFUTED: (a) cancellation in (I1 - A V1) -- factoring to B^-1 I1 - (B^-1 A) V1 gave
  bit-identical divergence; (b) wrong direction (running up the turns ratio) -- a directed
  solved->unsolved frontier gave bit-identical divergence.
- CONTROL: the MATRIX ladder converges on 123Bus (LAD@10 = 2.84e-10, rho=0.131). So the
  mechanism is sound and MY SWEEP IS NOT EQUIVALENT TO IT. Prime suspect: the splitting --
  I push with the element's full A = Ys + Yh (charging in the series step), while the
  matrix ladder splits Yser/Ysh differently, changing the iteration operator.
  NEXT: match the splitting exactly, then re-test.

### Speed (profiled, bs=4, 1333-node feeders, H100)
    DKFeeder build (incl recon_topo) : 1.35 s/feeder (one-time)
    collate (recon ctx + merge)      : 0.055 s/sample  (parallel over workers)
    MP only        fwd+bwd           : 0.053 s/sample
    exact decoder  fwd+bwd           : 0.103 s/sample  <- decoder ~= all 12 MP steps
Real run ~0.6 s/sample (feeders ~4000 nodes, 3x bigger). GPU 35%: work-bound, not
launch-bound, so bigger batches do not help. **gpu-h100 nodes have 4 GPUs and we use 1** --
DDP is the outstanding 4x. Risk: reconstruct_full's data-dependent control flow means the
autograd graph differs per rank (needs find_unused_parameters, can deadlock).

### Gotchas that cost real time
- `.venv` is torch+cu130 vs a 12.4 driver -> CUDA silently False, trains on CPU.
  Use `.venv-train` (cu126). train_dk_pf.sbatch now ASSERTS cuda.is_available().
- dataloader workers were re-forked every epoch (re-importing torch): persistent_workers.
- dk_train had no --seed and no manual_seed: runs were not reproducible. Split stays
  pinned at 42 so --seed measures training variance, not split variance.
