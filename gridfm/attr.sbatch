#!/bin/bash
#SBATCH -J dk_attr
#SBATCH -A gogpt
#SBATCH -p short
#SBATCH -N 1
#SBATCH --array=0-19
#SBATCH --mem=0
#SBATCH -t 1:00:00
#SBATCH -o /kfs2/projects/gogpt/Ebadmus/logs/attr-%A_%a.out
set -euo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
# datakit is a sibling repo on PYTHONPATH, not pip-installed.
export PYTHONPATH=/kfs2/projects/gogpt/Ebadmus:/kfs2/projects/gogpt/Ebadmus/Training_new${PYTHONPATH:+:${PYTHONPATH}}
source /kfs2/projects/gogpt/Ebadmus/.venv-train/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python gridfm/test_attr.py --corpus minimal_component --shard "${SLURM_ARRAY_TASK_ID}/20" \
    --workers 64 --variants 0 --out runs/attr
