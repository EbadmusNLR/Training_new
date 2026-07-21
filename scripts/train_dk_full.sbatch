#!/bin/bash
#SBATCH --job-name=dk_full
#SBATCH --account=gogpt
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=440G
#SBATCH --time=40:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/dk_full-%j.out
#SBATCH --error=/kfs2/projects/gogpt/Ebadmus/logs/dk_full-%j.err

# FULL foundation pretraining: all 4 H100s on the node via torchrun DDP.
# DO NOT LAUNCH until the 3-seed gates pass their criteria (scripts/launch_gates.sh):
# every seed's unseen v_skill clearly under 1.0 and trending down, seeds within ~0.1.
#
# Effective batch = BS * 4 ranks. Each rank draws an independent random sample subset;
# DDP averages gradients. find_unused_parameters is required (data-dependent decoder
# control flow + stores absent from some batches).

set -uo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
# .venv is cu130 vs the 12.4 driver -> CPU-only. .venv-train (cu126) gets the H100s.
PY=/kfs2/projects/gogpt/Ebadmus/.venv-train/bin/python

FEEDERS="${FEEDERS:-}"            # empty = every feeder in every corpus
EPOCHS="${EPOCHS:-120}"
HIDDEN="${HIDDEN:-256}"
STEPS="${STEPS:-12}"
BS="${BS:-8}"
LR="${LR:-4e-4}"
SPE="${SPE:-32000}"
SEED="${SEED:-0}"
TASK="${TASK:-random}"
# Loss weights default to the vonly winner (T11): the exact decoder makes I=f(V) by
# construction, so an explicit current loss is redundant AND injects stiff reactor/xfmr
# WAPE gradients (reactor 1e14%, xfmr 2666%) that fight V. Supervise V; get currents
# machine-exact from the decoder. Override W_I/W_KCL to re-enable if a run shows otherwise.
W_I="${W_I:-0.0}"
W_KCL="${W_KCL:-0.0}"
WARM="${WARM:-0.03}"
OUT="${OUT:-/kfs2/projects/gogpt/Ebadmus/Training_new/runs/dk_foundation_${SLURM_JOB_ID}}"
mkdir -p "$OUT" logs

"$PY" -c "import torch;assert torch.cuda.is_available();print('cuda OK,', torch.cuda.device_count(), 'GPUs')"
echo "[$(date '+%F %T')] task=$TASK epochs=$EPOCHS hidden=$HIDDEN steps=$STEPS bs=${BS}x4 lr=$LR spe=$SPE seed=$SEED out=$OUT"

"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/dk_train.py \
  ${FEEDERS:+--limit-feeders "$FEEDERS"} \
  --hidden "$HIDDEN" --steps "$STEPS" \
  --epochs "$EPOCHS" --batch-size "$BS" --lr "$LR" \
  --samples-per-epoch "$SPE" --seed "$SEED" --task "$TASK" \
  --w-i "$W_I" --w-kcl "$W_KCL" --warmup "$WARM" \
  --norm-loss \
  --workers 12 \
  --out "$OUT"
echo "[$(date '+%F %T')] exit=$?"
