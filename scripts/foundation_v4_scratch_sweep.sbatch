#!/bin/bash
#SBATCH --job-name=foundation_v4_scratch
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s,gpu-h100l
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/Training_new/logs/foundation-v4-scratch-%A_%a.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/Training_new/logs/foundation-v4-scratch-%A_%a.err

set -euo pipefail

PROJECT=/kfs2/projects/gogpt/Ebadmus
ROOT=$PROJECT/Training_new
labels=(lr5e5_e24 lr1e4_e24 lr2e4_e24 lr4e4_e24 seed0_e36 seed17_e36 seed31_e36 seed73_e36)
lrs=(5e-5 1e-4 2e-4 4e-4 1e-4 1e-4 1e-4 1e-4)
epochs=(24 24 24 24 36 36 36 36)
seeds=(42 42 42 42 0 17 31 73)
i=$SLURM_ARRAY_TASK_ID

export PYTHONPATH="$PROJECT:$ROOT:$PROJECT/DG_FM_Training"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
$PROJECT/.venv-cu124/bin/python "$ROOT/scripts/train.py" \
  --config "$ROOT/configs/foundation_v4_fractional.yaml" \
  --scratch \
  --exact-metadata both \
  --batch-size 8 \
  --num-workers 6 \
  --lr "${lrs[$i]}" \
  --epochs "${epochs[$i]}" \
  --seed "${seeds[$i]}" \
  --out "$ROOT/runs/foundation_v4_scratch_${SLURM_ARRAY_JOB_ID}/${labels[$i]}"
