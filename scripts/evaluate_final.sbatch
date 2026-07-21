#!/bin/bash
#SBATCH --job-name=gridfm_final
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/slurm-%j.out
set -euo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
# datakit is a sibling repo on PYTHONPATH, not pip-installed.
export PYTHONPATH=/kfs2/projects/gogpt/Ebadmus:/kfs2/projects/gogpt/Ebadmus/Training_new${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python}"
SELECTION="${SELECTION:-runs/final_selection.json}"

selection_values=$("$PYTHON_BIN" - "$SELECTION" <<'PY'
import hashlib
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
if receipt.get("test_metrics_read") is not False:
    raise SystemExit("invalid selection receipt")
row = receipt["selected"]
checkpoint = pathlib.Path(row["checkpoint"])
if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row["checkpoint_sha256"]:
    raise SystemExit("selected checkpoint hash no longer matches receipt")
print(checkpoint)
print(checkpoint.parent)
PY
)
readarray -t selected <<<"$selection_values"
ckpt="${selected[0]}"
out_dir="${selected[1]}"

outputs=(
  "$out_dir/test_direct.json"
  "$out_dir/test_tree.json"
  "$out_dir/seen_tree_final.json"
  "$out_dir/current_diagnostics_test.json"
)
if [[ "${FORCE:-0}" != 1 ]]; then
  for output in "${outputs[@]}"; do
    if [[ -e "$output" ]]; then
      echo "ERROR: refusing to overwrite final evaluation: $output" >&2
      exit 2
    fi
  done
fi

"$PYTHON_BIN" scripts/evaluate.py --ckpt "$ckpt" --split test \
  --output "$out_dir/test_direct.json"
"$PYTHON_BIN" scripts/evaluate.py --ckpt "$ckpt" --split test \
  --tree-line --kcl-vsource --output "$out_dir/test_tree.json"
"$PYTHON_BIN" scripts/evaluate.py --ckpt "$ckpt" --split seen \
  --tree-line --kcl-vsource --output "$out_dir/seen_tree_final.json"
"$PYTHON_BIN" scripts/current_diagnostics.py --ckpt "$ckpt" --split test \
  --output "$out_dir/current_diagnostics_test.json"
