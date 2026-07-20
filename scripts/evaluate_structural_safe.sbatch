#!/bin/bash
#SBATCH --job-name=gridfm_structural_safe
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/Training_new/logs/structural-safe-%j.out

set -euo pipefail
ROOT=/kfs2/projects/gogpt/Ebadmus/Training_new
PYTHON_BIN=/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
CONFIG="${CONFIG:-$ROOT/configs/foundation_v8_fractional.yaml}"
OUT="${OUTPUT_DIR:-$RUN_DIR/task_reports_structural_safe}"
mkdir -p "$OUT"
cd "$ROOT"
for task in pf se_known param_one injection random_safe; do
  "$PYTHON_BIN" scripts/evaluate.py --config "$CONFIG" \
    --ckpt "$RUN_DIR/best_foundation.pt" --split unseen --task "$task" \
    --structural-safe --output "$OUT/$task.json"
done
