#!/bin/bash
#SBATCH --job-name=tn_edge_pf
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
"$PYTHON_BIN" scripts/train.py --config "${CONFIG:-configs/pf_fraction.yaml}" ${EXTRA_ARGS:-}
