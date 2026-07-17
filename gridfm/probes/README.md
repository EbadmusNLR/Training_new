# Diagnostic probes

Scripts that answer one measured question each; each is the evidence behind a claim in
`../../training_experiments.md` or `../../experiments.md`. Nothing here is imported by the
training pipeline.

## Active (evidence for current architecture decisions)
- `ladder_solver_probe.py` — machine-precision test: ladder(truth Icomp) convergence +
  Icomp→V error gain, all 4 corpora (T15)
- `ladder_tree_sweep.py` — directed-frontier tree sweep (1.2e-12 transformer-free;
  transformer B-blocks singular = the open thread)
- `overfit_one_batch.py` — fixed-batch optimization ceiling + per-term grad norms (T02)
- `verify_model_decoder.py` — in-model exact decoder WAPE at truth V
- `profile_step.py`, `time_recon_ctx.py`, `time_topo_reuse.py` — where a training step's
  time goes; topology-cache speedup
- `measure_graph_depth.py`, `measure_receptive_field.py` — hop depth (113) vs PE cap /
  MP steps
- `test_*.py` — small invariant checks (batching, bridges, ground, vbase, xfmr maps, ...)
- `exclude_outliers.py` — the loud-skip exclusion list builder

## archive/
One-shot forensic scripts from **finished** investigations (decoder null-space, IEEE30
rank, reactor families, SMART-DS replicas). Kept for provenance only; most hardcode
corpus paths and may need repointing after a rebuild.
