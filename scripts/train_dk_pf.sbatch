#!/bin/bash
#SBATCH --job-name=dk_pf_probe
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/Training_new/logs/dk_pf-%j.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/Training_new/logs/dk_pf-%j.err

# PF probe on the datakit corpus with the EXACT decoder wired in.
#
# Baseline question: where does v_skill actually sit on the NEW corpus? The ~0.37 plateau
# was measured on the old data and blamed on cond(Ybus)=1.25e18 -- but this corpus
# measures cond 4e9 and is 100% PF-determinate, so that diagnosis needs re-testing before
# it is trusted.
#
# Env:
#   FEEDERS  feeders to use (default 200; the corpus is 1000 SMART-DS)
#   EPOCHS, HIDDEN, STEPS, BS, LR
#   OUT      run directory

set -uo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
export PYTHONUNBUFFERED=1
# .venv is torch+cu130 and the node driver is 12.4 -> CUDA unavailable, silently CPU-only.
# .venv-train is cu126, which runs on any 12.x driver (CUDA minor-version compatibility)
# and gets the full H100. Do not "fix" this by switching back.
PY=/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python

FEEDERS="${FEEDERS:-200}"
EPOCHS="${EPOCHS:-20}"
HIDDEN="${HIDDEN:-256}"
STEPS="${STEPS:-12}"
BS="${BS:-4}"
LR="${LR:-4e-4}"
ROOT_CORPUS="${ROOT_CORPUS:-/kfs2/projects/gogpt/Ebadmus/training_data/SMART-DS_1000}"
OUT="${OUT:-/kfs2/projects/gogpt/Ebadmus/Training_new/runs/dk_pf_probe_${SLURM_JOB_ID}}"
mkdir -p "$OUT" /kfs2/projects/gogpt/Ebadmus/Training_new/logs

echo "[$(date '+%F %T')] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
"$PY" -c "import torch;assert torch.cuda.is_available(), 'CUDA UNAVAILABLE -- wrong venv or driver';print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "[$(date '+%F %T')] feeders=$FEEDERS epochs=$EPOCHS hidden=$HIDDEN steps=$STEPS bs=$BS lr=$LR"
echo "[$(date '+%F %T')] out=$OUT"

"$PY" -u scripts/dk_train.py \
  --root "$ROOT_CORPUS" \
  --limit-feeders "$FEEDERS" \
  --hidden "$HIDDEN" --steps "$STEPS" \
  --epochs "$EPOCHS" --batch-size "$BS" --lr "$LR" \
  --workers 16 \
  --out "$OUT"
echo "[$(date '+%F %T')] exit=$?"
