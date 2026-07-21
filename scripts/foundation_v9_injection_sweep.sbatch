#!/bin/bash
#SBATCH --job-name=foundation_v9_injection
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s,gpu-h100l
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/foundation-v9-injection-%A_%a.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/logs/foundation-v9-injection-%A_%a.err

set -euo pipefail
PROJECT=/kfs2/projects/gogpt/Ebadmus
ROOT=$PROJECT/Training_new
mixtures=(
  '{"pf":0.35,"se_known":0.25,"injection":0.40}'
  '{"pf":0.25,"se_known":0.15,"injection":0.60}'
  '{"pf":0.35,"se_known":0.25,"injection":0.40}'
  '{"pf":0.35,"se_known":0.25,"injection":0.40}'
)
icomp_weights=(5 10 10 20)
ibus_weights=(1 1 0 1)
i=$SLURM_ARRAY_TASK_ID
export PYTHONPATH="$PROJECT:$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PROJECT/.venv-cu124/bin/python "$ROOT/scripts/train.py" \
  --config "$ROOT/configs/foundation_v8_fractional.yaml" --scratch \
  --exact-metadata all --batch-size 8 --num-workers 6 --lr 1e-4 \
  --epochs 36 --seed 73 --task-mixture "${mixtures[$i]}" \
  --recon-icomp-weight "${icomp_weights[$i]}" \
  --recon-ibus-weight "${ibus_weights[$i]}" \
  --out "$ROOT/runs/foundation_v9_injection_${SLURM_ARRAY_JOB_ID}/arm${i}"
