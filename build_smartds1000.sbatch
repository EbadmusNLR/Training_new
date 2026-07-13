#!/bin/bash
#SBATCH --job-name=smartds1000_build
#SBATCH --account=view
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/Training_new/runs/smartds1000_build-%j.out
set -euo pipefail
cd /kfs2/projects/gogpt/Ebadmus
export PYTHONPATH=/kfs2/projects/gogpt/Ebadmus/DG_FM_DK/scripts:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Fresh full build from source: fits scaler + perturbation ranges, then generates
# 30 variants/feeder across all 104 cores. tasks-per-child recycles OpenDSS workers
# so per-worker memory cannot grow unbounded (the earlier 64-worker/200G OOM).
.venv/bin/python -u DG_FM_DK/scripts/generate_training_data/make_training_pt.py \
  --source data/SMART-DS_1000 --count "${COUNT:-30}" \
  --out training_data/smartds1000 --workers 104 --tasks-per-child 8
