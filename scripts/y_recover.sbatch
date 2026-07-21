#!/bin/bash
#SBATCH --job-name=y_recover
#SBATCH --account=gogpt
#SBATCH --partition=shared
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/y_recover-%j.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/logs/y_recover-%j.err
# Y-recovery: missing line/xfmr Y from K snapshots of (V, Icomp) via KCL + lstsq --
# the parameter-estimation corner is LINEAR when V is observable (user scenario).
set -uo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=32
/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python -u gridfm/probes/y_recovery_probe.py --per-corpus 3
