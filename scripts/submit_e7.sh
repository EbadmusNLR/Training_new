#!/bin/bash
set -euo pipefail

cd /kfs2/projects/gogpt/Ebadmus/Training_new

submit_train() {
  local name="$1" config="$2" dependency="${3:-}"
  local args=(--parsable --job-name="$name" --export="ALL,CONFIG=$config")
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="afterok:$dependency")
  fi
  sbatch "${args[@]}" run.sbatch
}

# Scale-only controls can start immediately.
v3f_norm400=$(submit_train e7_v3f_n400 configs/e7_v3f_norm400.yaml)
v3f_raw400=$(submit_train e7_v3f_r400 configs/e7_v3f_raw400.yaml)
v3f_norm1000=$(submit_train e7_v3f_n1000 configs/e7_v3f_norm1000.yaml)

# Production candidates wait for exact re-encoding plus the physics validator.
det2f=$(sbatch --parsable scripts/reencode_det2f.sbatch)
det2f_norm400=$(submit_train e7_det2f_n400 configs/e7_det2f_norm400.yaml "$det2f")
det2f_norm1000=$(submit_train e7_det2f_n1000 configs/e7_det2f_norm1000.yaml "$det2f")

printf '%s\n' \
  "v3f_norm400=$v3f_norm400" \
  "v3f_raw400=$v3f_raw400" \
  "v3f_norm1000=$v3f_norm1000" \
  "det2f_reencode_validate=$det2f" \
  "det2f_norm400=$det2f_norm400" \
  "det2f_norm1000=$det2f_norm1000"

