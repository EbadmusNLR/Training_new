# Architecture review: differentiable solver in the training loop?

2026-07-18. Review of the proposal to embed the sparse physics solve as a
differentiable layer (Architecture B) vs the current design (A: direct
supervision on hidden quantities, solver external at inference) vs pure GNN (C).
Standard: try to falsify before recommending. Verdict at the end.

## 0. The measured facts this review stands on (all from this repo's ledger)

- F1 (T15/T17b): fp64 solve of Ybus at truth Icomp = machine precision on every
  feeder, mesh or radial (pf med 4.1e-9 over 43 unseen feeders).
- F2 (T17): checkpoint's hidden-Icomp estimates -> solve beats the same
  checkpoint's V head ~100x on every feeder.
- F3 (T21): zero-prior joint0 == trained-model joint on EVERY sample. The
  estimator currently contributes nothing through the solve.
- F4 (T21): V stays near-exact at nullity>0 because V depends on hidden Icomps
  only through node sums; the sums are determined, the splits are not.
- F5 (T22): Y recovery from snapshots fails by algebra regardless of K --
  load-variant excitation spans ~1 mode of 6 (sv 7.0 -> 1.9e-12).
- F6 (ic saturation): ic_wape ~100-107% under every loss recipe (ic_only,
  ic_heavy, vonly all alike) at probe scale.

## 1. Mathematical validity of B: YES, cheaply

Implicit differentiation through `x = A^{-1} b` is textbook adjoint:
dL/db = A^{-T} g (one triangular solve reusing the LU factor),
dL/dA = -(A^{-T} g) x^T (sparse: only nonzero-pattern entries needed).
Factor once per (feeder, variant), cache; backward adds ~one solve. A custom
`torch.autograd.Function` around scipy splu (fp64, CPU) or cuDSS suffices.
Conditioning is a non-issue here (cond ~4e9, F1). So B is FEASIBLE. Feasible is
not the question. The question is what the gradients would carry.

## 2. The falsification: the solve is a projection, and projections kill
##    exactly the gradients the model needs

Decompose the hidden-Icomp space at a sample into:
- the DETERMINED subspace (node sums + measurement-pinned directions), and
- the NULLSPACE (splits among co-located injections, unexcited directions).

Backprop of a V-loss through the solve gives dL/d(ic) = (dV/d(ic))^T dL/dV.

- In the determined subspace: the joint projection already corrects any
  estimate error, so V error ~ 0 there NO MATTER what the estimator emits
  (F3, F4). Loss ~ 0 => gradient ~ 0. B teaches nothing where physics wins.
- In the nullspace: dV/d(ic) = 0 EXACTLY (that is what nullspace means -- F4's
  mechanism). Zero sensitivity => zero gradient. B teaches nothing where
  physics loses, either.

So for the clean linear corpus, an end-to-end V loss through the solver is
gradient-dead precisely on the model's actual estate (the nullspace prior) and
redundant everywhere else. Direct supervision (A) is strictly more informative
for the estimator: it is the ONLY signal that reaches nullspace directions.
The same argument covers Y-through-the-solve: dV/dY vanishes on unexcited
directions (F5), which are exactly the directions needing the learned prior.

This is not a hypothetical: F3 is the empirical shadow of this argument -- the
V-through-solve objective is already implicitly optimized (any estimate gives
the same V), and correspondingly the estimator learned nothing of value.

## 3. Where B genuinely wins (and should be adopted LATER)

The projection argument holds only while data are CLEAN and constraints EXACT.
It breaks -- in B's favor -- when:

1. NOISE: with noisy visible entries, the solve becomes weighted least squares;
   the output depends on estimates and learned confidence weights everywhere
   (no exact-projection subspace). Training through the WLS layer teaches
   calibrated weighting/bad-data rejection (learned-GLS). This is the "state
   estimation with bad measurements" capability and is B's real home.
2. NONLINEAR INJECTIONS: Icomp = f(V; theta) closes a fixed-point loop;
   implicit-layer (DEQ-style) training through the solve is then the natural
   formulation.
3. LEARNED ITERATION over approximate operators (not our case; F1 says the
   exact operator is affordable).

## 4. Novelty check

Differentiable physics/optimization layers are established SciML (OptNet, DC3,
DEQ, unrolled AC-PF / DeepOPF variants). Claiming B as the contribution would
WEAKEN the paper. The defensible novelty of GridFM is the other half:
the four-array masking universe with the determinate subspace delegated to
exact algebra and the NN posed -- and directly supervised -- as the prior over
the identifiable-complement. Architecture A *is* the novel claim, stated
honestly.

## 5. Architecture C (pure GNN): already falsified

V-head plateau 0.77-1.13 across depth (s32), width (T07), target frame (vabs),
feature codec (no-feat), LR, PE dose -- vs 1e-8 through physics. Closed.

## 6. The deeper finding this review surfaced (from the mask-taxonomy doc)

The taxonomy's Class B row states: with (Y, V) visible and one snapshot, the
split of YV into (I_bus, I_comp) is NOT identifiable without constitutive
information. That is EXACTLY our measured ic saturation (F6): ic_wape ~100%
is not an optimization failure -- the task as posed carries no signal. The
estimator cannot learn what the sample does not determine.

Consequence (the real architecture change): MULTI-OPERATING-POINT SAMPLES.
The corpus has 100 variants per feeder sharing components and Y. Bundling K
variants of the same feeder into one sample (taxonomy task T6) makes the
response map V -> Icomp(V) learnable (constitutive-law learning) and
simultaneously provides the excitation aggregation that Y estimation needs
(F5). One structural change serves both unsolved heads.

## 7. Recommendation

KEEP A NOW; ADOPT B ONLY WITH NOISE/NONLINEARITY; DO THE TAXONOMY + T6 FIRST.

1. Keep direct supervision + external solve (A) for the clean-linear stage:
   B is measurably gradient-dead here (Sec. 2) and adds solver cost per step.
2. Adopt the mask taxonomy NOW (cheap, high-information):
   - class-conditional metrics (A/B/C/D per the four-class table) instead of
     one aggregate -- our aggregate currently mixes ~31% trivial class-A slots;
   - passive stores excluded from ic metrics (already done via PC_STORES);
   - track the I_bus/I_comp overlap mask under four-array masking
     (complementary visibility makes slots trivially exact);
   - add contiguous-region masks (T5) as an eval lens.
3. Build MULTI-SNAPSHOT SAMPLES (T6) as the next architecture step -- the
   measured route to an estimator that finally beats the zero prior, and the
   prerequisite for identifiable Y training.
4. Revisit B (differentiable WLS layer) when noise enters the masks; the
   adjoint implementation in Sec. 1 is the plan of record for that stage.
