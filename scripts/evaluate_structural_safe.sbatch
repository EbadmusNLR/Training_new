#!/bin/bash
#SBATCH --job-name=gridfm_structural_safe
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s
#SBATCH --gres=gpu:h100:1
# Kestrel allocates whole nodes; a partial --mem just caps you below the node's real
# memory (gpu-h100s = 360G) and OOMs. Take the node.
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/structural-safe-%j.out

set -euo pipefail
# datakit is a sibling repo on PYTHONPATH, not pip-installed.
export PYTHONPATH=/kfs2/projects/gogpt/Ebadmus:/kfs2/projects/gogpt/Ebadmus/Training_new${PYTHONPATH:+:${PYTHONPATH}}
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
"$PYTHON_BIN" scripts/structural_scorecard.py \
  --reports-dir "$OUT" --output "$OUT/scorecard.json"
