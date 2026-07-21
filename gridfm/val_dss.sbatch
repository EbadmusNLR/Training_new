#!/bin/bash
#SBATCH -J dk_valdss
#SBATCH -A gogpt
#SBATCH -p short
#SBATCH -N 1
#SBATCH --array=0-9
#SBATCH --mem=0
#SBATCH -t 2:00:00
#SBATCH -o /kfs2/projects/gogpt/Ebadmus/logs/valdss-%A_%a.out
set -euo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
source /kfs2/projects/gogpt/Ebadmus/.venv-train/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CORPUS_ROOT=/kfs2/projects/gogpt/Ebadmus/training_data/dss_data
python gridfm/test_all.py --shard "${SLURM_ARRAY_TASK_ID}/10" --workers 64 \
    --variants 0 --out runs/kvl_dss
