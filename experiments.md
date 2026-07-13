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
