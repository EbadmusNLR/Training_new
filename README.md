# Training_new

Topology-held-out training code for a distribution-grid foundation model.

The non-negotiable inference contract is:

- `V_init` is known at every bus.
- Solved slack-phase voltage is known and hard-clamped.
- Real and imaginary channels remain separate.
- `Ibus` and `Icomp` remain separate and satisfy `Ibus + Icomp = YV` in pu.
- No OpenDSS or assembled power-flow solve is used by the learned model.
- No baseline computed from solved target voltages of an evaluation topology is allowed.
- Content-derived topology groups, not names or samples, define unseen/test splits;
  structurally equivalent vendored copies can never cross split boundaries.

The first architecture is `EdgeStateGridFM`: recurrent bipartite message passing over
component-terminal incidences, explicit terminal voltage proposals, a graph-global state,
and direct complex line-drop supervision. It reuses the already validated scenario-store
decoder and float64 physics functions from `DG_FM_Training`; learned model code and strict
evaluation live here.

## Quick gates

```bash
python -m unittest discover -s tests -v
mkdir -p logs
sbatch smoke.sbatch
```

Full fractional run: `sbatch run.sbatch`. The production corpus is
`minimal_component_det2f`; its exact-current re-encoding and clean-validator evidence are
recorded in `experiments.md`.

Evaluate every identifiable task plus the explicit all-field stress mask, then select only
from unseen-topology scorecards:

```bash
sbatch --export=ALL,RUN_DIR=runs/<candidate>,CKPT=runs/<candidate>/best_foundation.pt \
  scripts/evaluate_foundation.sbatch
python scripts/select_foundation.py \
  --scorecard runs/<candidate-a>/task_reports_unseen/scorecard.json \
  --scorecard runs/<candidate-b>/task_reports_unseen/scorecard.json \
  --output runs/foundation_selection.json
```

Only after that command writes the selection receipt may the fixed held-out test be read.
`random_safe` randomly samples identifiable PF/SE/Y/Icomp tasks; `random` is a deliberately
underdetermined simultaneous-mask stress test.

```bash
sbatch --export=ALL,RUN_DIR=runs/<selected>,CKPT=runs/<selected>/best_foundation.pt,\
SPLIT=seen,OUTPUT_DIR=runs/<selected>/task_reports_seen scripts/evaluate_foundation.sbatch
sbatch --export=ALL,RUN_DIR=runs/<selected>,CKPT=runs/<selected>/best_foundation.pt,\
SPLIT=test,OUTPUT_DIR=runs/<selected>/task_reports_test scripts/evaluate_foundation.sbatch
```

The selector rejects non-unseen reports, and evaluators refuse to overwrite reports unless
`FORCE=1` is explicit. Package the hash-verified checkpoint with
`scripts/promote_foundation.py` after both final splits complete.

Nontrivial training must run on an allocated compute node through Slurm. Every promoted
checkpoint must report both held operating points on known feeders and entirely held-out
feeders using split-level WAPE percentages.

## Selected foundation result

E51 is selected on unseen feeders. It reaches PF `1.691% V / 9.753% Ibus` direct
(`6.557%` structural), known-injection SE `1.876% / 10.713%`, Y `0.843%`, and Icomp
`0.479%`. Worst scale-normalized Y/Icomp fields are `2.628% / 1.200%`. The safe random
mixture is `1.745% / 9.899% / 0.832% / 0.490%` for V/Ibus/Y/Icomp. It does not pass the
1% all-task gate; the scorecard records every failure rather than promoting a false claim.
On the fixed held-out test it reaches `2.106% V / 6.535% Ibus` with the stable hybrid
branch-current policy, Y `0.918%`, and Icomp `0.481%`. The packaged artifact is
`runs/foundation_best`; checkpoint SHA-256 is
`1c9a97b9183e0527c42439e8d052135bdaef83d3e9598f7ab35961b5a821ee17`.
