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
  TriplexLine P95 current scale (`9.41e-10`) in `minimal_component_ifields`.
- Current oracle with truth voltage is exact (`1.18e-5%` WAPE). Unseen line current, not loads or
  transformers, is the remaining generalization bottleneck; increasing feeder coverage helps most.
- `minimal_component_v3f` fixes scale conditioning only. The later `minimal_component_det2`
  fixes feeder wiring/determinacy but still has a collapsed TriplexLine scale; re-encode it exactly
  to `minimal_component_det2f`, validate, then use `det2f` for production comparisons.
- `det2f` must floor current scales only (`alpha_y=0`): flooring stiff Y coordinates exceeded the
  `1e-6` physics gate. Derive missing triplex flags from baseline JSON; the audit cache has only 40 feeders.
- Current solution: train direct device currents with physical WAPE and H384 on all 2,000 feeders,
  then reconstruct radial line series current by subtree KCL. It is solver-free and gives seen
  `0.570% V / 1.368% Ibus`, unseen validation `1.767% / 6.732%`, and sealed test
  `2.124% / 6.888%`. The selected checkpoint is hash-pinned by `runs/final_selection.json`.
- Do not call legacy `topo`, `sysid`, or `ctrl` masks foundation capabilities: topology is visible,
  system identification is single-snapshot/underdetermined, and explicit control labels are absent.
- Masked-Y tasks must set `model.use_electrical_pe: false`; the final PE coordinate is computed from true
  variant-0 Y. Use `se_known`, `param_one`, and `injection` for identifiable inverse tasks.
- Generic random masking must preserve complex pairs, structural zeros, all-bus `V_init`, and solved slack
  voltage. Train it after identifiable pretraining; starting with all random tasks caused severe interference.
- WAPE is not dominated by tiny currents: truth above `0.1 pu` accounts for `88.6%` of unseen error.
- A reactor is not a simple paired line edge: reactor-first subtree decoding worsened Ibus to `10.43%`.
- The broad unseen-topology winner is E51: PF `1.691% / 9.753%` direct, SE `1.876% / 10.713%`, Y `0.843%`, Icomp `0.479%`; worst Y/Icomp family-scale fields are `2.628% / 1.200%` before external finalization.
- Keep `random_safe` distinct from simultaneous all-field `random`: the former samples only identifiable PF/SE/one-Y/Icomp tasks; the latter is an explicitly underdetermined stress test.
- Voltage WAPE does not certify current. Stiff line `Ys*dV` amplifies tiny drop/precision errors; use local physics only for non-stiff devices, tree KCL for series line flow, and exact `jYh(V1+V2)/2` for line common mode.
- Foundation checkpoint selection uses aggregate PF/SE/Y/Icomp plus worst scale-normalized Y/Icomp fields. Raw family WAPE with zero truth denominator is diagnostic-only.
- Promoted E51 is `runs/foundation_best` (SHA-256 `1c9a97b9183e0527c42439e8d052135bdaef83d3e9598f7ab35961b5a821ee17`). Fixed-test structural-hybrid PF is `2.106% V / 6.535% Ibus`; Y/Icomp are `0.918% / 0.481%`.
