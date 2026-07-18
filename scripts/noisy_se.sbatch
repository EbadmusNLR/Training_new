#!/bin/bash
#SBATCH --job-name=noisy_se
#SBATCH --account=gogpt
#SBATCH --partition=shared
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/Training_new/logs/noisy_se-%j.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/Training_new/logs/noisy_se-%j.err
# Noisy-SE probe: naive joint vs weighted (WLS) vs robust solve under corrupted
# visible V (see gridfm/probes/noisy_se_probe.py).
set -uo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=32
CKPT="${CKPT:-runs/dk_foundation_15261953/last.pt}"
/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python -u gridfm/probes/noisy_se_probe.py \
  --ckpt "$CKPT" --n-feeders "${NF:-16}" --sigma "${SIGMA:-0.01}" \
  --gross-frac "${GROSS:-0.03}" --subset-seed "${SEED:-401}" \
  --task "${TASK:-random}" --wp "${WP:-1e-3}"
