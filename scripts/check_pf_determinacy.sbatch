#!/bin/bash
#SBATCH -J dk_determinacy
#SBATCH -A gogpt
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 03:30:00
#SBATCH -o /kfs2/projects/gogpt/Ebadmus/logs/determinacy-%j.out
#SBATCH -e /kfs2/projects/gogpt/Ebadmus/logs/determinacy-%j.err
set -uo pipefail
R=/kfs2/projects/gogpt/Ebadmus; PY=$R/.venv/bin/python
export PYTHONUNBUFFERED=1
for c in dss_data minimal_component new_dss_data SMART-DS_1000; do
  echo "################################################################ $c"
  $PY -u $R/Training_new/scripts/check_pf_determinacy.py --corpus "$c" --feeders 12 --variants 2
done
