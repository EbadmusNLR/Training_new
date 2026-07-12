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
