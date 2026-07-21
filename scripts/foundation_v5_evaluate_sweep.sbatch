#!/bin/bash
#SBATCH --job-name=foundation_v5_eval
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s,gpu-h100l
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
# Kestrel allocates whole nodes; a partial --mem just caps you below the node's real
# memory (gpu-h100s = 360G) and OOMs. Take the node.
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/foundation-v5-eval-%A_%a.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/logs/foundation-v5-eval-%A_%a.err

set -euo pipefail
# datakit is a sibling repo on PYTHONPATH, not pip-installed.
export PYTHONPATH=/kfs2/projects/gogpt/Ebadmus:/kfs2/projects/gogpt/Ebadmus/Training_new${PYTHONPATH:+:${PYTHONPATH}}

ROOT=/kfs2/projects/gogpt/Ebadmus/Training_new
TRAIN_JOB_ID="${TRAIN_JOB_ID:?set TRAIN_JOB_ID to the v5 training array ID}"
labels=(both_seed31 all_seed31 all_seed17 all_seed0 all_seed73 all_seed42)
label=${labels[$SLURM_ARRAY_TASK_ID]}
export RUN_DIR="$ROOT/runs/foundation_v5_generator_${TRAIN_JOB_ID}/$label"
export CKPT="$RUN_DIR/best_foundation.pt"
export CONFIG="$ROOT/configs/foundation_v5_fractional.yaml"
export OUTPUT_DIR="$RUN_DIR/task_reports_corrected"
export PYTHON_BIN=/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python
bash "$ROOT/scripts/evaluate_foundation.sbatch"
