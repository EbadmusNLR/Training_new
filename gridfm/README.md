# gridfm

Two model paths live here. They do not share data code, and mixing them up is the main
way to get lost:

## `dk_*` — the datakit path (current)

Trains on the datakit corpus (`training_data/<corpus>/<feeder>/static.pt` + `dynamic.npy`,
full Y matrices, pu). This is the live foundation-model path.

| module | role |
|---|---|
| `dk_physics.py` | store schema, terminal/slot indexing, element currents, nodal KCL residual, `ensure_batch_schema` |
| `dk_tree.py` | the current decoder: tree/mesh topology, transformer null-space systems, `reconstruct_full` (exact) and `build_recon_ctx`/`batch_recon_ctx` |
| `dk_data.py` | `DKFeeder` (per-feeder caches), `DKDataset`, task masks, `make_dk_collate` |
| `dk_model.py` | `DKSolver` — the weight-tied recurrent hetero solver |

Entry point: `scripts/dk_train.py`. Gate: `scripts/check_pf_determinacy.py`.

**Currents are decoded, never free-headed.** Shunts are physics-decoded from V
(`I = Y@V - Icomp`, well-conditioned); series flows come from the subtree-KCL
reconstruction. Series `Y·(V1-V2)` is never used — it amplifies V error ~2e7x on lines.

## legacy — the E51 foundation path (previous champion)

`model.py`, `data.py`, `legacy.py`, `featurizing.py`, `tree_current.py`,
`current_projection.py`, `hybrid_current.py`, `kcl_feedback.py`, `kcl_series.py`,
`voltage_refinement.py`. Trains on the pre-datakit format via `scripts/train.py`, and is
still what ~10 evaluation scripts import. Kept because `runs/foundation_best` (E51) came
from it. Do not extend it; new work goes in the `dk_*` path.

## Shared

`losses.py` (`balanced_reconstruction_loss` — per-NODE averaging over the batch, so a
4000-node feeder outweighs a 67-node one ~58:1), `config.py`.

## Subdirectories

- `tests/` — assertions on physics/decoder identities (`test_all.py`, `test_physics.py`,
  `verify_batch_recon.py`, …). Run these after touching `dk_tree.py`.
- `tools/` — corpus tooling (dedupe, scans, perturbation screening). Mutates or inspects
  `data/`/`training_data/`; run on a compute node.
- `probes/` — one-off diagnostics kept for their answers, not part of any pipeline.
  `verify_model_decoder.py` and `time_topo_reuse.py` are the ones worth rerunning.

Nothing in `tests/`, `tools/` or `probes/` is imported by the model or training path.
