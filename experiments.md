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
