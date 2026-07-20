#!/bin/bash
#SBATCH --job-name=gridfm_foundation_eval
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
PYTHON_BIN="${PYTHON_BIN:-/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python}"
RUN_DIR="${RUN_DIR:?set RUN_DIR to the candidate run directory}"
CKPT="${CKPT:-$RUN_DIR/best_foundation.pt}"
SPLIT="${SPLIT:-unseen}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/task_reports_$SPLIT}"
mkdir -p "$OUTPUT_DIR"

outputs=(pf.json pf_tree.json se_known.json param_one.json injection.json random_safe.json random.json)
if [[ "${FORCE:-0}" != 1 ]]; then
  for name in "${outputs[@]}"; do
    if [[ -e "$OUTPUT_DIR/$name" ]]; then
      echo "ERROR: refusing to overwrite $OUTPUT_DIR/$name" >&2
      exit 2
    fi
  done
fi

run_task() {
  local task="$1"
  "$PYTHON_BIN" scripts/evaluate.py \
    --ckpt "$CKPT" --split "$SPLIT" --task "$task" \
    --output "$OUTPUT_DIR/$task.json"
}

run_task pf
"$PYTHON_BIN" scripts/evaluate.py \
  --ckpt "$CKPT" --split "$SPLIT" --task pf --tree-line --kcl-vsource \
  --output "$OUTPUT_DIR/pf_tree.json"
run_task se_known
run_task param_one
run_task injection
run_task random_safe
run_task random

if [[ "$SPLIT" == unseen ]]; then
  # The scorecard intentionally exits 2 when any 1% gate fails. Preserve the
  # receipt while allowing the evaluation job to finish honestly.
  "$PYTHON_BIN" scripts/foundation_scorecard.py \
    --reports-dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/scorecard.json" || [[ $? == 2 ]]
fi
