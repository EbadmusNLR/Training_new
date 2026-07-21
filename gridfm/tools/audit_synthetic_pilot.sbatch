#!/bin/bash
#SBATCH -J synth_diversity
#SBATCH -A gogpt
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -o /kfs2/projects/gogpt/Ebadmus/logs/synthetic-diversity-%j.out
#SBATCH -e /kfs2/projects/gogpt/Ebadmus/logs/synthetic-diversity-%j.err

set -euo pipefail

PROJECT=/kfs2/projects/gogpt/Ebadmus
ROOT=${ROOT:-$PROJECT/Training_new/runs/synthetic_v2_pilot_15299610}
CORPUS=${CORPUS:-minimal_component_v3_r2}
PY=${PY:-$PROJECT/.venv/bin/python}
WORKERS=${SLURM_CPUS_PER_TASK:-32}
WORKERS=${WORKERS%%(*}

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "$PROJECT/Training_new"
for mode in all loading orthogonal; do
  "$PY" -u gridfm/tools/scan_perturb.py \
    --root "$ROOT/$mode/$CORPUS" \
    --workers "$WORKERS" \
    --json-out "$ROOT/${mode}_diversity.json" \
    --fail-on-errors \
    --require-no-electrical-duplicates
done

echo "SYNTHETIC_DIVERSITY_OK root=$ROOT corpus=$CORPUS"
