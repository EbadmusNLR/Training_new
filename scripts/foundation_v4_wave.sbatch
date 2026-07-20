#!/bin/bash
#SBATCH --job-name=foundation_v4_wave
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s,gpu-h100l
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=180G
#SBATCH --time=03:30:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/Training_new/logs/foundation-v4-wave-%A_%a.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/Training_new/logs/foundation-v4-wave-%A_%a.err

set -euo pipefail

PROJECT=/kfs2/projects/gogpt/Ebadmus
ROOT=$PROJECT/Training_new
CONFIG=$ROOT/configs/foundation_v4_fractional.yaml
labels=(warm_exact scratch_exact warm_generic scratch_generic)
label=${labels[$SLURM_ARRAY_TASK_ID]}
extra=()
case "$label" in
  warm_exact) extra=(--exact-metadata both --lr 2.5e-6) ;;
  scratch_exact) extra=(--exact-metadata both --scratch --lr 1e-4) ;;
  warm_generic) extra=(--exact-metadata none --lr 2.5e-6) ;;
  scratch_generic) extra=(--exact-metadata none --scratch --lr 1e-4) ;;
esac

export PYTHONPATH="$PROJECT:$ROOT:$PROJECT/DG_FM_Training"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
srun --ntasks=1 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PROJECT/.venv-cu124/bin/python "$ROOT/scripts/train.py" \
  --config "$CONFIG" \
  --batch-size 8 \
  --num-workers 6 \
  --out "$ROOT/runs/foundation_v4_${SLURM_ARRAY_JOB_ID}/$label" \
  "${extra[@]}"
