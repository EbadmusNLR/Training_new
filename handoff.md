# Live handoff
- Mission: topology-general four-array GridFM that reconstructs any identifiable missing V, Y, Icomp, or terminal feature and obeys physics.
- First read `prompt.md`, then this file, then `training_experiments.md`.
- All nine element-definition-to-Y paths pass target-poisoned audits; data should reopen only with failing proof.
- Contract: stored `I_feat=I_bus+Icomp=YV`; only physical KCL uses `I_bus=I_feat-Icomp`.
- Structural-safe artifact `runs/foundation_best_structural_v10` passes full gate `15317136`: max/mean `2.677e-4/7.084e-5%`.
- Raw learned heads still fail Icomp near `99-101%`; seed17 is fallback/diagnostic, not success.
- `build_synthetic_corpus` is canonical; imports `datakit.core.solver` (`06d46b4`) and keeps analysis in package `analysis/`.
- `DG_FM_DK` is removed legacy; active references were removed from Training_new/build_synthetic.
- Cleanup: `datakit` `9e8fe42`; `Training_new` removed stale `wt/*`, dead probe launchers, runtime logs/cache.
- Commit each validated major change; preserve unrelated edits and keep generated artifacts/checkpoints/logs out of Git.
## Next actions
1. Build one-process multi-task evaluation; replace dense PF with sparse fp64; rerun structural/topology gates.
2. Resume speed-first training/debugging only against the topology-general identifiable scorecard.
