#!/bin/bash
#SBATCH --job-name=y_recover
#SBATCH --account=gogpt
#SBATCH --partition=shared
# Kestrel allocates whole nodes; a partial --mem just caps you below the node's real
# memory (gpu-h100s = 360G) and OOMs. Take the node.
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/y_recover-%j.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/logs/y_recover-%j.err
# Y-recovery: missing line/xfmr Y from K snapshots of (V, Icomp) via KCL + lstsq --
# the parameter-estimation corner is LINEAR when V is observable (user scenario).
set -uo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
# datakit is a sibling repo on PYTHONPATH, not pip-installed.
export PYTHONPATH=/kfs2/projects/gogpt/Ebadmus:/kfs2/projects/gogpt/Ebadmus/Training_new${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=32
/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python -u gridfm/probes/y_recovery_probe.py --per-corpus 3
