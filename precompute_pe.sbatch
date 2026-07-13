#!/bin/bash
#SBATCH --job-name=smartds_pe
#SBATCH --account=gogpt
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=104
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/Training_new/runs/smartds_pe-%j.out
set -euo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python -u scripts/precompute_pe.py \
  --root /kfs2/projects/gogpt/Ebadmus/training_data/smartds1000 --workers 104
