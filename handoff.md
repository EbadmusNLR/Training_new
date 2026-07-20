# Live handoff
- Mission: topology-general four-array GridFM that reconstructs any identifiable missing V, Y, Icomp, or terminal feature and obeys physics.
- Read `prompt.md`, then `training_experiments.md`; the latter is the sole detailed experiment ledger.
- All nine element-definition-to-Y paths pass exhaustive target-poisoned corpus audits; do not reopen data without a failing proof.
- Current contract: stored `I_feat=I_bus+Icomp=YV`; only physical KCL uses `I_bus=I_feat-Icomp`.
- Key commits: DG_FM_Training `4802d3b`, `bf468d7`, `08f4978`; Training_new `b24a1bc`, `5efdc02`, `a806505`.
- Injection masks now hide only PC slots on valid KCL nodes, globally at most one hidden component per conductor node.
- Structural-safe v2 passes; seed73 `15316282_3` plus retries `15316386` are live; eval/select `15316387/388` follow.
- Next: finish the live chain, separate hybrid/raw scorecards, stabilize exact-cache fingerprinting, and remove only proven dead/generated remnants.
- Pin-memory root cause is fixed in `scripts/train.py`: only train uses workers; seen/unseen/task evaluation is synchronous.
- Raw scorecards now label the stored terminal target as `Ifeat`; legacy `Ibus` receipts remain readable.
- Preserve unrelated edits; `handoff.md` itself was intentionally emptied before this rewrite, and every validated major change must be committed.
## Next actions
1. Package the passing structural scorecard with the winning raw fallback after `15316282/283/284` completes.
2. Monitor `15316282/283/284`; use only completed unseen scorecards and never select on test data.
3. Update T99 and the v10 ledger row with final measured values and an explicit verdict.
4. If structural-safe passes, make it the deployed identifiable reconstruction path while retaining raw heads as diagnostics/fallbacks.
5. Vectorize `kcl_decode_icomp` and replace the dense PF solve with a sparse fp64 implementation before large-feeder scaling.
6. Decouple exact-cache validity from wrapper-only source edits; retain explicit schema and decoder fingerprints.
7. Rerun unit, mask, structural, and topology-general gates after those speed changes; poison hidden targets again.
8. Clean canceled/invalid v10 run directories and logs, stale cache temporaries, and proven dead probes without touching training corpora.
9. Commit each validated cleanup/change separately and leave all nested repositories clean for handoff.
