#!/bin/bash
#SBATCH --job-name=foundation_v10_eval
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s,gpu-h100,gpu-h100l
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
# Kestrel allocates whole nodes; a partial --mem just caps you below the node's real
# memory (gpu-h100s = 360G) and OOMs. Take the node.
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/foundation-v10-eval-%A_%a.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/logs/foundation-v10-eval-%A_%a.err

set -euo pipefail
PROJECT=/kfs2/projects/gogpt/Ebadmus
ROOT=$PROJECT/Training_new
seeds=(31 42 17 73)
TRAIN_JOB="${TRAIN_JOB:?set TRAIN_JOB to the completed v10 training array id}"
i=$SLURM_ARRAY_TASK_ID
RUN_DIR="$ROOT/runs/foundation_v10_contract_${TRAIN_JOB}/all_seed${seeds[$i]}"
export RUN_DIR CONFIG="$ROOT/configs/foundation_v8_fractional.yaml"
export OUTPUT_DIR="$RUN_DIR/task_reports_corrected"
exec bash "$ROOT/scripts/evaluate_foundation.sbatch"
