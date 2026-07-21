#!/bin/bash
# Validate ONE corpus of training_data with the current decoder. CORPUS is passed in:
#   sbatch --export=ALL,CORPUS=new_dss_data gridfm/val_final.sbatch
# Fresh --out per corpus: reusing a runs/ dir mixes shards from different code versions,
# which is how a straddled sweep produced an untrustworthy number earlier today.
#SBATCH -J dk_valfin
#SBATCH -A gogpt
#SBATCH -p short
#SBATCH -N 1
#SBATCH --array=0-19
#SBATCH --mem=0
#SBATCH -t 3:00:00
#SBATCH -o /kfs2/projects/gogpt/Ebadmus/logs/valfin-%A_%a.out
set -euo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
# datakit is a sibling repo on PYTHONPATH, not pip-installed.
export PYTHONPATH=/kfs2/projects/gogpt/Ebadmus:/kfs2/projects/gogpt/Ebadmus/Training_new${PYTHONPATH:+:${PYTHONPATH}}
source /kfs2/projects/gogpt/Ebadmus/.venv-train/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CORPUS_ROOT=/kfs2/projects/gogpt/Ebadmus/training_data/${CORPUS}
python gridfm/test_all.py --shard "${SLURM_ARRAY_TASK_ID}/20" --workers 64 \
    --variants 0 --out runs/${OUTTAG:-final}_${CORPUS}
