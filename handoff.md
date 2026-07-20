# Live handoff
- Mission: topology-general four-array GridFM that reconstructs any identifiable missing V, Y, Icomp, or terminal feature and obeys physics.
- Read `prompt.md`, then `training_experiments.md`; the latter is the sole detailed experiment ledger.
- All nine element-definition-to-Y paths pass exhaustive target-poisoned corpus audits; do not reopen data without a failing proof.
- Current contract: stored `I_feat=I_bus+Icomp=YV`; only physical KCL uses `I_bus=I_feat-Icomp`.
- Key commits: DG_FM_Training `4802d3b`, `bf468d7`, `08f4978`; Training_new `b24a1bc`, `5efdc02`, `a806505`.
- Injection masks now hide only PC slots on valid KCL nodes, globally at most one hidden component per conductor node.
- Live: structural-safe v2 `15316280`; corrected four-seed train/eval/select `15316282/283/284` (some array tasks may wait for H100s).
- Next: record v2 Icomp/random-safe results, finish the live chain, then stabilize exact-cache fingerprinting and remove only proven dead/generated remnants.
- Preserve unrelated edits; `handoff.md` itself was intentionally emptied before this rewrite, and every validated major change must be committed.
