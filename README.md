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

Full fractional run: `sbatch run.sbatch`.

For an already-open GPU allocation, run the complete corrected-corpus promotion gate from
the compute-node shell:

```bash
cd /kfs2/projects/gogpt/Ebadmus/Training_new
MAX_PARALLEL=5 bash scripts/run_e7_in_allocation.sh
```

It creates and validates `minimal_component_det2f`, runs the matched v3f/det2f ablations,
selects only from unseen-topology validation metrics, and evaluates the winner once on test.

Nontrivial training must run on an allocated compute node through Slurm. Every promoted
checkpoint must report both held operating points on known feeders and entirely held-out
feeders using split-level WAPE percentages.
