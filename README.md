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
python scripts/audit_contract.py --config configs/pf_fraction.yaml
python scripts/train.py --config configs/pf_fraction.yaml --epochs 2 --device cpu
python scripts/evaluate.py --config configs/pf_fraction.yaml --ckpt runs/pf_fraction/best_voltage.pt
```

Nontrivial training must run on an allocated compute node through Slurm. Every promoted
checkpoint must report both held operating points on known feeders and entirely held-out
feeders using split-level WAPE percentages.
