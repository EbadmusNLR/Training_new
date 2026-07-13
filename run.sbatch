#!/bin/bash
#SBATCH --job-name=tn_train
#SBATCH --account=view
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/kfs2/projects/gogpt/Ebadmus/.venv/bin/python3.14}"
CONFIG="${1:-configs/pf_fraction.yaml}"
"$PYTHON_BIN" scripts/train.py --config "$CONFIG" ${EXTRA_ARGS:-}
