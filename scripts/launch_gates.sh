#!/bin/bash
# The 3-seed gate before any full launch (the agreed sequence: small runs must LEARN
# first). Balanced multi-corpus splits, small-first feeders, the reference recipe
# (residual gauge + norm-loss + anneal), foundation objective by default.
#
#   bash scripts/launch_gates.sh              # task=random_safe
#   TASK=pf bash scripts/launch_gates.sh      # pf-only control
#
# Pass criteria (all three seeds):
#   - unseen v_skill clearly < 1.0 and TRENDING DOWN across epochs
#   - seeds agree within ~0.1 of each other (one diverging seed = instability, no launch)
set -uo pipefail
cd /kfs2/projects/gogpt/Ebadmus/Training_new
TASK="${TASK:-random_safe}"
for s in 0 1 2; do
  J=$(FEEDERS=60 EPOCHS=40 BS=8 SPE=1600 SEED=$s TASK=$TASK NORM=1 WORKERS=16 \
      OUT=/kfs2/projects/gogpt/Ebadmus/Training_new/runs/gate_${TASK}_s$s \
      sbatch --parsable scripts/train_dk_pf.sbatch)
  echo "gate seed=$s task=$TASK -> job $J"
done
squeue -u ebadmus -h -o "%.11i %.10P %.2t %R" | head -5
