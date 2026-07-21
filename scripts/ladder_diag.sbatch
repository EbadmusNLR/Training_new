#!/bin/bash
#SBATCH --job-name=ladder_diag
#SBATCH --account=gogpt
#SBATCH --partition=shared
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/ladder_diag-%j.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/logs/ladder_diag-%j.err
set -uo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=32
/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python -u gridfm/probes/ladder_anomaly_diag.py
