#!/bin/bash
#SBATCH --job-name=dk_pf_probe
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
# WALL env sets walltime. Ask the SHORTEST honest time: short requests backfill
# into scheduler gaps, a 4h request waits for a 4h hole. Gates run ~35 min.
#SBATCH --time=01:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/Training_new/logs/dk_pf-%j.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/Training_new/logs/dk_pf-%j.err

# PF training over ALL FOUR corpora (stratified splits; see dk_train.py --roots).
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
ulimit -n 65536 2>/dev/null || true
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
SPE="${SPE:-4000}"      # samples per epoch
SEED="${SEED:-0}"
TASK="${TASK:-pf}"
W_I="${W_I:-1.0}"
W_KCL="${W_KCL:-0.1}"
WARM="${WARM:-0.03}"
FB="${FB:-0}"
NORM="${NORM:-1}"
WORKERS="${WORKERS:-16}"
OUT="${OUT:-/kfs2/projects/gogpt/Ebadmus/Training_new/runs/dk_pf_probe_${SLURM_JOB_ID}}"
mkdir -p "$OUT" /kfs2/projects/gogpt/Ebadmus/Training_new/logs

echo "[$(date '+%F %T')] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
"$PY" -c "import torch;assert torch.cuda.is_available(), 'CUDA UNAVAILABLE -- wrong venv or driver';print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "[$(date '+%F %T')] feeders=$FEEDERS epochs=$EPOCHS hidden=$HIDDEN steps=$STEPS bs=$BS lr=$LR spe=$SPE seed=$SEED task=$TASK norm=$NORM"
echo "[$(date '+%F %T')] out=$OUT"

"$PY" -u scripts/dk_train.py \
  --limit-feeders "$FEEDERS" \
  --hidden "$HIDDEN" --steps "$STEPS" \
  --epochs "$EPOCHS" --batch-size "$BS" --lr "$LR" \
  --samples-per-epoch "$SPE" --seed "$SEED" --task "$TASK" \
  --w-i "$W_I" --w-kcl "$W_KCL" --warmup "$WARM" --fb-points "$FB" \
  ${NORM:+--norm-loss} --small-first \
  --workers "$WORKERS" \
  --out "$OUT"
echo "[$(date '+%F %T')] exit=$?"
