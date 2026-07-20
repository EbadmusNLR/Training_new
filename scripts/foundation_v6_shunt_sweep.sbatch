#!/bin/bash
#SBATCH --job-name=foundation_v6_shunt
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s,gpu-h100l
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/Training_new/logs/foundation-v6-shunt-%A_%a.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/Training_new/logs/foundation-v6-shunt-%A_%a.err

set -euo pipefail
PROJECT=/kfs2/projects/gogpt/Ebadmus
ROOT=$PROJECT/Training_new
seeds=(31 42 17 73)
i=$SLURM_ARRAY_TASK_ID
label="all_seed${seeds[$i]}"

export PYTHONPATH="$PROJECT:$ROOT:$PROJECT/DG_FM_Training"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PROJECT/.venv-cu124/bin/python "$ROOT/scripts/train.py" \
  --config "$ROOT/configs/foundation_v6_fractional.yaml" --scratch \
  --exact-metadata all --batch-size 8 --num-workers 6 --lr 1e-4 \
  --epochs 36 --seed "${seeds[$i]}" \
  --out "$ROOT/runs/foundation_v6_shunt_${SLURM_ARRAY_JOB_ID}/$label"
