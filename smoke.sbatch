#!/bin/bash
#SBATCH --job-name=tn_smoke
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/kfs2/projects/gogpt/Ebadmus/Training_new}"
mkdir -p logs
PY=/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python
export PYTHONUNBUFFERED=1
"$PY" -m unittest discover -s tests -v
"$PY" scripts/audit_contract.py --config configs/smoke.yaml
"$PY" scripts/train.py --config configs/smoke.yaml
"$PY" scripts/evaluate.py --config configs/smoke.yaml --baseline v_init --split unseen \
  --output runs/smoke/baseline_unseen.json
"$PY" scripts/evaluate.py --ckpt runs/smoke/best_voltage.pt --split seen \
  --output runs/smoke/seen.json
"$PY" scripts/evaluate.py --ckpt runs/smoke/best_voltage.pt --split unseen \
  --output runs/smoke/unseen.json

