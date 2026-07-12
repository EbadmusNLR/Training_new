#!/bin/bash
#SBATCH --job-name=e7_promote
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python}"
candidates=(
  e7_det2f_norm400 e7_det2f_norm1000 e7_det2f_norm2000
  e7_det2f_wape400 e7_det2f_wape1000 e7_det2f_wape2000
  e7_det2f_h256_400 e7_det2f_h256_1000 e7_det2f_h256_2000
)

for name in "${candidates[@]}"; do
  "$PYTHON_BIN" scripts/evaluate.py \
    --ckpt "runs/$name/best_current.pt" --split unseen --kcl-vsource \
    --output "runs/$name/unseen_source_kcl.json"
done

winner=$("$PYTHON_BIN" scripts/summarize_e7.py --select)
echo "selected=$winner"
ckpt="runs/$winner/best_current.pt"
"$PYTHON_BIN" scripts/evaluate.py --ckpt "$ckpt" --split test \
  --output "runs/$winner/test_direct.json"
"$PYTHON_BIN" scripts/evaluate.py --ckpt "$ckpt" --split test --kcl-vsource \
  --output "runs/$winner/test_source_kcl.json"
"$PYTHON_BIN" scripts/evaluate.py --ckpt "$ckpt" --split test \
  --kcl-project line --kcl-vsource --output "runs/$winner/test_line_kcl.json"
"$PYTHON_BIN" scripts/current_diagnostics.py --ckpt "$ckpt" --split test \
  --output "runs/$winner/current_diagnostics_test.json"
"$PYTHON_BIN" scripts/summarize_e7.py | tee runs/e7_summary.csv

