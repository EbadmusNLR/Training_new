#!/bin/bash
#SBATCH --job-name=e2e_solve
#SBATCH --account=gogpt
#SBATCH --partition=shared
# Kestrel allocates whole nodes; a partial --mem just caps you below the node's real
# memory (gpu-h100s = 360G) and OOMs. Take the node.
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/e2e_solve-%j.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/logs/e2e_solve-%j.err
# E2E: checkpoint's hidden-Icomp estimates -> ONE direct fp64 Ybus solve -> V,
# scored against the same checkpoint's V head (see gridfm/probes/direct_solve_e2e.py).
set -uo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
# datakit is a sibling repo on PYTHONPATH, not pip-installed.
export PYTHONPATH=/kfs2/projects/gogpt/Ebadmus:/kfs2/projects/gogpt/Ebadmus/Training_new${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=32
CKPT="${CKPT:-runs/fix_lr2e4/last.pt}"
TASKS="${TASKS:-}"    # e.g. "random pf sw80 sw50 sw20 sw0"
/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python -u gridfm/probes/direct_solve_e2e.py \
  --ckpt "$CKPT" --n-feeders "${NF:-8}" ${TASKS:+--tasks $TASKS}
