# Live handoff
- Mission: topology-general four-array GridFM that reconstructs any identifiable missing V, Y, Icomp, or terminal feature and obeys physics.
- Read `prompt.md`, then `training_experiments.md`; the latter is the sole detailed experiment ledger.
- All nine element-definition-to-Y paths pass exhaustive target-poisoned corpus audits; do not reopen data without a failing proof.
- Current contract: stored `I_feat=I_bus+Icomp=YV`; only physical KCL uses `I_bus=I_feat-Icomp`.
- Key commits: DG_FM_Training `4802d3b`, `bf468d7`, `08f4978`; Training_new `b24a1bc`, `5efdc02`, `a806505`.
- Injection masks now hide only PC slots on valid KCL nodes, globally at most one hidden component per conductor node.
- Structural-safe passes; raw seeds 73 and 31 fail (Icomp ~101%); seed42/17 eval + unseen-only selection are chained as `15316830/897/898`.
- Next: finish the live chain, separate hybrid/raw scorecards, stabilize exact-cache fingerprinting, and remove only proven dead/generated remnants.
- Pin-memory root cause is fixed in `scripts/train.py`: only train uses workers; seen/unseen/task evaluation is synchronous.
- Foundation evaluation also defaults to zero workers; override only after measuring a larger split.
- Raw scorecards now label the stored terminal target as `Ifeat`; legacy `Ibus` receipts remain readable.
- Preserve unrelated edits; `handoff.md` itself was intentionally emptied before this rewrite, and every validated major change must be committed.
## Next actions
1. Package the passing structural scorecard with the least-bad raw fallback after selection completes.
2. Monitor `15316830/897/898`; use only completed unseen scorecards and never select on test data.
3. Update T100 with all four final raw scorecards and an explicit verdict.
4. Promote structural-safe as the deployed identifiable path; retain raw heads only for fail-closed fallback/diagnostics.
5. Vectorize `kcl_decode_icomp` and replace the dense PF solve with sparse fp64 before large-feeder scaling.
6. Stabilize exact-cache fingerprints and make multi-task evaluation reuse one process/dataset.
7. Rerun unit, mask, structural, topology-general, and target-poison gates after speed changes.
8. Clean canceled/invalid runs, logs, stale cache temporaries, and proven dead probes without touching corpora.
9. Commit each validated cleanup/change separately and leave every nested repository clean.
