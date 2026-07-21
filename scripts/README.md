# Training_new scripts map

Keep runnable code here, generated logs/checkpoints in `../logs` or `../runs`, and old one-off scratch outside the repo.

## Active foundation path

- `dk_train.py`, `train.py`: training entrypoints.
- `evaluate_foundation.sbatch`, `foundation_scorecard.py`, `select_foundation.py`, `promote_foundation.py`: unseen-only evaluation, selection, and packaging.
- `evaluate_structural_safe.sbatch`, `structural_scorecard.py`: structural-safe identifiable gate.
- `foundation_v10_*`: current structural contract/evaluation sweep launchers.

## Historical sweep launchers

- `foundation_v4_*` through `foundation_v9_*`: retained for reproducibility of earlier documented rows in `training_experiments.md`.
- `train_dk_full.sbatch`, `train_dk_pf.sbatch`: older baseline training launchers used by prior scorecards.

## Diagnostics

- `check_*`, `current_*`, `e2e_solve.sbatch`, `ladder_*`, `noisy_se.sbatch`, `y_recover.sbatch`: targeted physics/model probes; use only when debugging a named failure.

## Removed dead clutter

- `probe.sbatch`, `probe2.sbatch`: generic one-off wrappers replaced by named launchers.
- `promote.py`: old promotion helper superseded by `promote_foundation.py`.
