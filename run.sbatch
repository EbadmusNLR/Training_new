#!/bin/bash
#SBATCH --job-name=tn_train
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/slurm-%j.out
# THE training launcher. Consolidated from the old run.sbatch + train_smartds.sbatch,
# which disagreed on account (view vs gogpt), partition (standard = NO GPU vs
# gpu-h100s) and interpreter (.venv cu130 vs .venv-train cu126). GPU + gogpt +
# .venv-train is what every launcher in scripts/ uses.
#
# 64G (not 180G) per T85: measured peak RSS across scientific arms is <43GB and the
# smaller request schedules faster.
#
# Usage:
#   sbatch run.sbatch                            # default config below
#   sbatch run.sbatch foundation_v8_fractional   # -> configs/<name>.yaml
#   sbatch run.sbatch configs/other.yaml         # explicit path
#   RESUME=1 sbatch run.sbatch <cfg>             # resume from checkpoint
#   EXTRA_ARGS="--epochs 12" sbatch run.sbatch <cfg>
set -euo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
export PYTHONUNBUFFERED=1
# datakit is a sibling repo on PYTHONPATH, not pip-installed. Without this every run
# dies at import with ModuleNotFoundError: No module named 'datakit'.
export PYTHONPATH=/kfs2/projects/gogpt/Ebadmus:/kfs2/projects/gogpt/Ebadmus/Training_new${PYTHONPATH:+:${PYTHONPATH}}
PYTHON_BIN="${PYTHON_BIN:-/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python}"

# Config argument: bare name gets configs/ prefix and .yaml suffix.
if [[ -n "${1:-}" ]]; then
    if [[ "$1" == *.yaml ]]; then
        CONFIG="$1"
    else
        CONFIG="configs/$1.yaml"
    fi
else
    CONFIG="configs/foundation_v8_fractional.yaml"
fi
[[ -f "$CONFIG" ]] || { echo "no such config: $CONFIG" >&2; exit 2; }
echo "[$(date '+%F %T')] config=$CONFIG python=$PYTHON_BIN"

"$PYTHON_BIN" scripts/train.py --config "$CONFIG" ${RESUME:+--resume} ${EXTRA_ARGS:-}
