#!/bin/bash
#SBATCH --job-name=foundation_v10_contract
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s,gpu-h100,gpu-h100l
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/foundation-v10-contract-%A_%a.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/logs/foundation-v10-contract-%A_%a.err

set -euo pipefail
PROJECT=/kfs2/projects/gogpt/Ebadmus
ROOT=$PROJECT/Training_new
seeds=(31 42 17 73)
i=$SLURM_ARRAY_TASK_ID
run_group_job="${RUN_GROUP_JOB:-$SLURM_ARRAY_JOB_ID}"
num_workers="${NUM_WORKERS:-6}"
export PYTHONPATH="$PROJECT:$ROOT:$PROJECT/DG_FM_Training"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PROJECT/.venv-cu124/bin/python "$ROOT/scripts/train.py" \
  --config "$ROOT/configs/foundation_v8_fractional.yaml" --scratch \
  --exact-metadata all --batch-size 8 --num-workers "$num_workers" --lr 1e-4 \
  --epochs 36 --seed "${seeds[$i]}" \
  --out "$ROOT/runs/foundation_v10_contract_${run_group_job}/all_seed${seeds[$i]}"
