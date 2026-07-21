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

The current path is the datakit-backed four-array GridFM: datakit owns the source
export and device-definition metadata, while Training_new owns strict splits,
exact metadata integration, learned heads, and evaluation. Training_new is
self-contained: the corpus schema, masking and physics helpers live in
`gridfm/core/` (re-exported via `gridfm.legacy` for the existing import sites),
so nothing depends on the old `DG_FM_Training` tree any more.

## Quick gates

```bash
PYTHONPATH=/kfs2/projects/gogpt/Ebadmus:$PWD python -m unittest discover -s tests -v
mkdir -p logs
# (datakit is a sibling repo on PYTHONPATH, not pip-installed)
bash -n run.sbatch scripts/*.sbatch gridfm/*.sbatch gridfm/probes/*.sbatch gridfm/tools/*.sbatch
```

Full fractional run: `sbatch run.sbatch` (GPU; pass a config name, e.g.
`sbatch run.sbatch foundation_v8_fractional`, which is also the default). The feature
corpus is built from `training_data/` into `training_data_foundation_v8_fractional/`;
the nine-family exact-Y decode and clean-validator evidence are recorded in
`training_experiments.md` (T88-T94).

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

## Current foundation result

The deployed identifiable path is structural-safe: exact device-definition Y,
fp64 voltage solve, `I_feat=YV`, and KCL recovery for uniquely identifiable PC
injections. `runs/foundation_best_structural_v10` is the promoted artifact;
checkpoint SHA-256 starts `213670b9`. Raw learned heads remain fallback
diagnostics because Icomp is still around `99-101%` WAPE.
