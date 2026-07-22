#!/bin/bash
#SBATCH --job-name=det2f_reencode
#SBATCH --account=gogpt
#SBATCH --partition=short
#SBATCH --nodes=1
# Kestrel allocates whole nodes; a partial --mem just caps you below the node's real
# memory (gpu-h100s = 360G) and OOMs. Take the node.
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=det2f-%j.out

set -euo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
# datakit is a sibling repo on PYTHONPATH, not pip-installed.
export PYTHONPATH=/kfs2/projects/gogpt/Ebadmus:/kfs2/projects/gogpt/Ebadmus/Training_new${PYTHONPATH:+:${PYTHONPATH}}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/reencode_corpus.py \
  --src /kfs2/projects/gogpt/Ebadmus/training_data/minimal_component \
  --out /kfs2/projects/gogpt/Ebadmus/training_data/minimal_component \
  --flags /kfs2/projects/gogpt/Ebadmus/training_data/minimal_component/_audit_cache/line_triplex.pt \
  --alpha-y 0 \
  --workers 32
python scripts/validation.py scenario \
  --search-root /kfs2/projects/gogpt/Ebadmus/training_data/minimal_component \
  --scenario-max-rows 1 --quiet-info
