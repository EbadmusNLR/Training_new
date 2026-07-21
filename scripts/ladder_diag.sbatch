#!/bin/bash
#SBATCH --job-name=ladder_diag
#SBATCH --account=gogpt
#SBATCH --partition=shared
# Kestrel allocates whole nodes; a partial --mem just caps you below the node's real
# memory (gpu-h100s = 360G) and OOMs. Take the node.
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/ladder_diag-%j.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/logs/ladder_diag-%j.err
set -uo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
# datakit is a sibling repo on PYTHONPATH, not pip-installed.
export PYTHONPATH=/kfs2/projects/gogpt/Ebadmus:/kfs2/projects/gogpt/Ebadmus/Training_new${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=32
/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python -u gridfm/probes/ladder_anomaly_diag.py
