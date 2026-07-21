#!/bin/bash
#SBATCH --job-name=tn_train
#SBATCH --account=view
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/slurm-%j.out
set -euo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/kfs2/projects/gogpt/Ebadmus/.venv/bin/python3.14}"
# Handle config argument: if no path, prepend configs/ and .yaml
if [[ -n "${1:-}" ]]; then
    if [[ "$1" == *.yaml ]]; then
        CONFIG="$1"
    else
        CONFIG="configs/$1.yaml"
    fi
else
    CONFIG="configs/pf_fraction.yaml"
fi
"$PYTHON_BIN" scripts/train.py --config "$CONFIG" ${EXTRA_ARGS:-}
