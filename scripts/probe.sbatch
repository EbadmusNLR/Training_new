#!/bin/bash
#SBATCH -J dk_probe
#SBATCH -A gogpt
#SBATCH -p gpu-h100s
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=0
#SBATCH -t 1:00:00
#SBATCH -o /kfs2/projects/gogpt/Ebadmus/Training_new/logs/probe-%j.out
set -euo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
source /kfs2/projects/gogpt/Ebadmus/.venv-train/bin/activate
export PYTHONUNBUFFERED=1
python scripts/dk_train.py "$@"
