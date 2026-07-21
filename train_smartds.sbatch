#!/bin/bash
#SBATCH --job-name=smartds_kcl_train
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/smartds_train-%j.out
set -euo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
export PYTHONUNBUFFERED=1
PY=/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python
"$PY" scripts/train.py --config configs/smartds_kcl.yaml ${RESUME:+--resume}
