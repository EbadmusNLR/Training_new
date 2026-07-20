#!/bin/bash
#SBATCH --job-name=foundation_v8_eval
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s,gpu-h100l
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/Training_new/logs/foundation-v8-eval-%A_%a.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/Training_new/logs/foundation-v8-eval-%A_%a.err

set -euo pipefail
PROJECT=/kfs2/projects/gogpt/Ebadmus
ROOT=$PROJECT/Training_new
seeds=(31 42 17 73)
TRAIN_JOB="${TRAIN_JOB:?set TRAIN_JOB to the completed v8 training array id}"
i=$SLURM_ARRAY_TASK_ID
RUN_DIR="$ROOT/runs/foundation_v8_storage_${TRAIN_JOB}/all_seed${seeds[$i]}"
export RUN_DIR CONFIG="$ROOT/configs/foundation_v8_fractional.yaml"
export OUTPUT_DIR="$RUN_DIR/task_reports_corrected"
exec bash "$ROOT/scripts/evaluate_foundation.sbatch"
