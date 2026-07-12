# Training_new

Topology-held-out training code for a distribution-grid foundation model.

The non-negotiable inference contract is:

- `V_init` is known at every bus.
- Solved slack-phase voltage is known and hard-clamped.
- Real and imaginary channels remain separate.
- `Ibus` and `Icomp` remain separate and satisfy `Ibus + Icomp = YV` in pu.
- No OpenDSS or assembled power-flow solve is used by the learned model.
- No baseline computed from solved target voltages of an evaluation topology is allowed.
- Feeder identities, not samples, define unseen-topology splits.

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

Select a final checkpoint only from unseen-topology reports. Each report must have been
produced with `--tree-line --kcl-vsource`:

```bash
python scripts/select_champion.py \
  --report runs/<candidate-a>/unseen_tree.json \
  --report runs/<candidate-b>/unseen_tree.json \
  --output runs/final_selection.json
```

Only after that command writes the selection receipt may the sealed test split be opened:

```bash
sbatch --export=ALL,SELECTION=runs/final_selection.json scripts/evaluate_final.sbatch
```

The selector rejects non-unseen or non-structural-current reports. The final evaluator
refuses to overwrite an existing test report unless `FORCE=1` is explicitly supplied.

Nontrivial training must run on an allocated compute node through Slurm. Every promoted
checkpoint must report both held operating points on known feeders and entirely held-out
feeders using split-level WAPE percentages.
