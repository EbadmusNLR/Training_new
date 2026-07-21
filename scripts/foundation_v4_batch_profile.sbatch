#!/bin/bash
#SBATCH --job-name=foundation_v4_batch
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s,gpu-h100l
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
# Kestrel allocates whole nodes; a partial --mem just caps you below the node's real
# memory (gpu-h100s = 360G) and OOMs. Take the node.
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/foundation-v4-batch-%A_%a.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/logs/foundation-v4-batch-%A_%a.err

set -euo pipefail

PROJECT=/kfs2/projects/gogpt/Ebadmus
ROOT=$PROJECT/Training_new
batches=(4 16 24 32)
batch=${batches[$SLURM_ARRAY_TASK_ID]}
export PYTHONPATH="$PROJECT:$ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
srun --ntasks=1 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PROJECT/.venv-cu124/bin/python "$ROOT/scripts/train.py" \
  --config "$ROOT/configs/foundation_v4_fractional.yaml" \
  --epochs 3 \
  --batch-size "$batch" \
  --num-workers 6 \
  --exact-metadata both \
  --lr 2.5e-6 \
  --out "$ROOT/runs/foundation_v4_batch_${SLURM_ARRAY_JOB_ID}/batch${batch}"
