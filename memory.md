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
- Initial implementation commit: `6a7a8ba`.
- Compute rule remains hard: PyG/corpus/training gates run through Slurm. The first submission
  attempt was blocked by the agent sandbox's Slurm stream-socket restriction, not a code error.
- `asinh` is invertible and not the physics bug. The bad conditioning came from a collapsed
  TriplexLine P95 current scale (`9.41e-10`) in `minimal_component_ifields`; use the validated,
  exactly re-encoded `minimal_component_v3f` (`4.05e-6` floor) for E7 onward.
- Current oracle with truth voltage is exact (`1.18e-5%` WAPE). Unseen line current, not loads or
  transformers, is the remaining generalization bottleneck; increasing feeder coverage helps most.
- `minimal_component_v3f` fixes scale conditioning only. The later `minimal_component_det2`
  fixes feeder wiring/determinacy but still has a collapsed TriplexLine scale; re-encode it exactly
  to `minimal_component_det2f`, validate, then use `det2f` for production comparisons.
- `det2f` must floor current scales only (`alpha_y=0`): flooring stiff Y coordinates exceeded the
  `1e-6` physics gate. Derive missing triplex flags from baseline JSON; the audit cache has only 40 feeders.
- Current solution: train direct device currents with physical WAPE and H256 on all 2,000 feeders,
  then reconstruct radial line series current by subtree KCL. This is solver-free and gives seen
  `0.845% V / 1.633% Ibus`, unseen `2.021% / 6.851%`; test remains sealed until final selection.
