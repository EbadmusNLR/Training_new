#!/bin/bash
#SBATCH --job-name=foundation_promote
#SBATCH --account=gogpt
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/foundation-promote-%j.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/logs/foundation-promote-%j.err

set -euo pipefail
# datakit is a sibling repo on PYTHONPATH, not pip-installed.
export PYTHONPATH=/kfs2/projects/gogpt/Ebadmus:/kfs2/projects/gogpt/Ebadmus/Training_new${PYTHONPATH:+:${PYTHONPATH}}
ROOT=/kfs2/projects/gogpt/Ebadmus/Training_new
PYTHON_BIN="${PYTHON_BIN:-/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python}"
SELECTION="${SELECTION:?set SELECTION to an unseen-only selection receipt}"
STRUCTURAL_REPORTS="${STRUCTURAL_REPORTS:?set STRUCTURAL_REPORTS to passing reports}"
PROMOTED_DIR="${PROMOTED_DIR:?set PROMOTED_DIR to a fresh output directory}"
cd "$ROOT"

CKPT="$($PYTHON_BIN - "$SELECTION" <<'PY'
import json
import sys
from pathlib import Path
print(Path(json.loads(Path(sys.argv[1]).read_text())["selected"]["checkpoint"]).resolve())
PY
)"
RUN_DIR="$(dirname "$CKPT")"

for split in seen test; do
  reports="$RUN_DIR/task_reports_${split}_corrected"
  complete=1
  for name in pf pf_tree se_known param_one injection random_safe random; do
    [[ -f "$reports/$name.json" ]] || complete=0
  done
  if [[ "$complete" == 0 ]]; then
    RUN_DIR="$RUN_DIR" CKPT="$CKPT" SPLIT="$split" OUTPUT_DIR="$reports" \
      EVAL_WORKERS=0 FORCE=1 bash scripts/evaluate_foundation.sbatch
  fi
done

"$PYTHON_BIN" scripts/promote_foundation.py \
  --selection "$SELECTION" \
  --seen-reports "$RUN_DIR/task_reports_seen_corrected" \
  --test-reports "$RUN_DIR/task_reports_test_corrected" \
  --structural-reports "$STRUCTURAL_REPORTS" \
  --output "$PROMOTED_DIR"
