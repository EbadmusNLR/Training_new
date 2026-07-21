#!/bin/bash
#SBATCH --job-name=foundation_v5_gen
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s,gpu-h100l
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/foundation-v5-gen-%A_%a.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/logs/foundation-v5-gen-%A_%a.err

set -euo pipefail

PROJECT=/kfs2/projects/gogpt/Ebadmus
ROOT=$PROJECT/Training_new
labels=(both_seed31 all_seed31 all_seed17 all_seed0 all_seed73 all_seed42)
metadata=(both all all all all all)
seeds=(31 31 17 0 73 42)
i=$SLURM_ARRAY_TASK_ID

export PYTHONPATH="$PROJECT:$ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
$PROJECT/.venv-cu124/bin/python "$ROOT/scripts/train.py" \
  --config "$ROOT/configs/foundation_v5_fractional.yaml" \
  --scratch \
  --exact-metadata "${metadata[$i]}" \
  --batch-size 8 \
  --num-workers 6 \
  --lr 1e-4 \
  --epochs 36 \
  --seed "${seeds[$i]}" \
  --out "$ROOT/runs/foundation_v5_generator_${SLURM_ARRAY_JOB_ID}/${labels[$i]}"
