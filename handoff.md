# Live handoff
- Mission: topology-general four-array GridFM that reconstructs any identifiable missing V, Y, Icomp, or terminal feature and obeys physics.
- Read `prompt.md`, then `training_experiments.md`; keep this file short.
- All nine element-definition-to-Y paths pass exhaustive target-poisoned audits; do not reopen data without a failing proof.
- Contract: stored `I_feat=I_bus+Icomp=YV`; only physical KCL uses `I_bus=I_feat-Icomp`.
- Injection masks now hide only PC slots on valid KCL nodes, globally at most one hidden component per conductor node.
- Structural-safe is the deployed identifiable path: full gate `15317136` max/mean `2.677e-4/7.084e-5%`.
- Raw learned heads still fail Icomp around `99-101%`; seed17 is fallback/diagnostic only.
- Promoted artifact: `runs/foundation_best_structural_v10`, checkpoint SHA-256 `213670b9...18c0c4`.
- Speed work: exact-cache fingerprint stabilized; KCL decode vectorized (`852x` microbench); sparse PF/multi-task eval still next.
- `build_synthetic_corpus` is the canonical synthetic generator; cleanup commit `3e458d8`, scratch now belongs in `/tmp`/logs/data, not source.
- Commit each validated major change; preserve unrelated edits and keep generated artifacts/checkpoints/logs out of Git.
## Next actions
1. Build one-process multi-task evaluation so gates do not reload the model/dataset per lens.
2. Replace dense PF solve with sparse fp64 for large feeders, then rerun structural/topology gates.
3. Continue training/debugging only against the topology-general identifiable scorecard; raw heads are not success.
