# One-off diagnostic probes

Throwaway scripts written to answer a single question during decoder debugging. Kept for
provenance (each one is the evidence behind a claim in ../../experiments.md), NOT part of
any pipeline -- nothing imports them and no sbatch references them.

Reusable validators live in ../ (test_all.py, test_mc.py, scan_dead.py, scan_perturb.py,
screen_perturb.py, dbg_many.py, dbg_model_decoder.py, verify_batch_recon.py).

Most hardcode `training_data/<corpus>` paths and may need repointing after a corpus rebuild.
