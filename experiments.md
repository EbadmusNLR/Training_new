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
