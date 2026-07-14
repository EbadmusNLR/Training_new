#!/bin/bash
#SBATCH --job-name=smartds_pe
#SBATCH --account=gogpt
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=104
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/DG_FM_DK/logs/smartds_pe-%j.out
set -euo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKERS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
ROOT="${ROOT:-/kfs2/projects/gogpt/Ebadmus/training_data/smartds1000}"

/kfs2/projects/gogpt/Ebadmus/.venv/bin/python -u /kfs2/projects/gogpt/Ebadmus/DG_FM_DK/scripts/generate_training_data/featurize_scenario_stores.py \
  --root "$ROOT" --workers "$WORKERS"

/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python -u scripts/precompute_pe.py \
  --root "$ROOT" --workers "$WORKERS"
