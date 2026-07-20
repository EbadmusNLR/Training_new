# Live handoff
- Mission: topology-general four-array GridFM that reconstructs any identifiable missing V, Y, Icomp, or terminal feature and obeys physics.
- Read `prompt.md`, then `training_experiments.md`; the latter is the sole detailed experiment ledger.
- All nine element-definition-to-Y paths pass exhaustive target-poisoned corpus audits; do not reopen data without a failing proof.
- Current contract: stored `I_feat=I_bus+Icomp=YV`; only physical KCL uses `I_bus=I_feat-Icomp`.
- Key commits: DG_FM_Training `4802d3b`, `bf468d7`, `08f4978`; Training_new `b24a1bc`, `5efdc02`, `a806505`.
- Injection masks now hide only PC slots on valid KCL nodes, globally at most one hidden component per conductor node.
- Structural-safe passes; all four raw seeds fail (Icomp ~99-101%); unseen-only selection chose seed17 and packaging `15317056` passed.
- Promoted artifact: `runs/foundation_best_structural_v10`, checkpoint SHA-256 `213670b9...18c0c4`; structural max error `2.677e-4%`.
- Pin-memory root cause is fixed in `scripts/train.py`: only train uses workers; seen/unseen/task evaluation is synchronous.
- Foundation evaluation also defaults to zero workers; override only after measuring a larger split.
- Raw scorecards now label the stored terminal target as `Ifeat`; legacy `Ibus` receipts remain readable.
- Preserve unrelated edits; `handoff.md` itself was intentionally emptied before this rewrite, and every validated major change must be committed.
## Next actions
1. Keep structural-safe as the deployed identifiable path; raw seed17 is fail-closed fallback/diagnostics only.
2. Stabilize exact-cache fingerprints, then make multi-task evaluation reuse one model/dataset process.
3. KCL vectorization passed unit tests and benchmarked `852x` faster; monitor full structural regression `15317136`.
4. Promote structural-safe as the deployed identifiable path; retain raw heads only for fail-closed fallback/diagnostics.
5. Vectorize `kcl_decode_icomp` and replace the dense PF solve with sparse fp64 before large-feeder scaling.
6. Stabilize exact-cache fingerprints and make multi-task evaluation reuse one process/dataset.
7. Rerun unit, mask, structural, topology-general, and target-poison gates after speed changes.
8. Clean canceled/invalid runs, logs, stale cache temporaries, and proven dead probes without touching corpora.
9. Commit each validated cleanup/change separately and leave every nested repository clean.
