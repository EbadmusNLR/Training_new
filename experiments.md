# Experiments

## Promotion contract

1. No topology-specific baseline derived from solved voltage targets.
2. No solver-assisted model output.
3. Slack voltage and every-bus `V_init` are always known.
4. Train/seen/unseen feeder sets are asserted disjoint where required.
5. Metrics are split-level WAPE percentages; slack voltage is excluded.
6. Truth-physics and tiny-overfit gates precede scaling.

## 2026-07-12 — clean architecture start

- Initialized an independent Git repository.
- Reused only the validated scenario-store decoder, masks, and float64 physics helpers.
- Selected explicit component-terminal edge state, terminal voltage proposals, global graph
  state, and complex branch-drop supervision as the first architecture.
- Per-feeder solved-voltage means are forbidden; voltage is represented as `V_init + dV`.

