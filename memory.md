# Project memory

- Goal: a solver-free foundational grid model with strict unseen-topology evaluation.
- Data authority: `DG_FM_DK/docs/feature.tex`, `DG_FM_DK/docs/physics.tex`, and the validated
  `training_data/minimal_component_ifields` scenario stores.
- Always-known inputs: `V_init` for every node and solved real/imaginary voltage at the three
  non-ground slack phase nodes.
- Never combine `Ibus` and `Icomp` as one ambiguous target. Physics is
  `Ibus + Icomp = YV`; KCL sums `Ibus`.
- Never use a topology-specific mean computed from solved voltage targets.
- Never promote a seen-topology-only result as foundational generalization.

