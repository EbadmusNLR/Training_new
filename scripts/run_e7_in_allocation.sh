#!/bin/bash
set -euo pipefail

ROOT=/kfs2/projects/gogpt/Ebadmus
HERE="$ROOT/Training_new"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-train/bin/python}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
mkdir -p "$HERE/allocation_logs/e7"

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "ERROR: run this inside allocation 15101070 on a GPU compute node" >&2
  exit 2
fi
ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( ngpu < 1 )); then
  echo "ERROR: no visible GPUs" >&2
  exit 2
fi
if (( MAX_PARALLEL < 1 )); then
  MAX_PARALLEL=$((ngpu * 2))
  (( MAX_PARALLEL > 5 )) && MAX_PARALLEL=5
fi
echo "host=$(hostname) ngpu=$ngpu max_parallel=$MAX_PARALLEL"

src="$ROOT/training_data/minimal_component_det2"
det2f="$ROOT/training_data/minimal_component_det2f"
if [[ ! -f "$det2f/feature_scaler.json" ]]; then
  workers="${REENCODE_WORKERS:-$(nproc)}"
  (( workers > 32 )) && workers=32
  cd "$HERE"
  "$PYTHON_BIN" scripts/reencode_corpus.py \
    --src "$src" --out "$det2f" \
    --flags "$src/_audit_cache/line_triplex.pt" --workers "$workers" \
    2>&1 | tee "$HERE/allocation_logs/e7/det2f_reencode.log"
fi

cd "$ROOT/DG_FM_DK"
"$PYTHON_BIN" scripts/validation.py scenario --search-root "$det2f" \
  --scenario-max-rows 1 --quiet-info \
  2>&1 | tee "$HERE/allocation_logs/e7/det2f_validation.log"

cd "$HERE"
configs=(
  configs/e7_v3f_norm400.yaml
  configs/e7_v3f_raw400.yaml
  configs/e7_v3f_norm1000.yaml
  configs/e7_det2f_norm400.yaml
  configs/e7_det2f_norm1000.yaml
)
pids=()
names=()
launch_idx=0

reap_one() {
  local pid="${pids[0]}" name="${names[0]}"
  if ! wait "$pid"; then
    echo "ERROR: training failed: $name" >&2
    exit 1
  fi
  pids=("${pids[@]:1}")
  names=("${names[@]:1}")
}

for config in "${configs[@]}"; do
  while (( ${#pids[@]} >= MAX_PARALLEL )); do reap_one; done
  name=$(basename "$config" .yaml)
  gpu=$((launch_idx % ngpu))
  echo "launch $name on cuda:$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/train.py --config "$config" \
    >"allocation_logs/e7/${name}.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
  launch_idx=$((launch_idx + 1))
done
while (( ${#pids[@]} )); do reap_one; done

# Evaluate every candidate on validation topologies. Test remains untouched
# until summarize_e7.py selects one det2f checkpoint from unseen metrics.
for name in e7_v3f_norm400 e7_v3f_raw400 e7_v3f_norm1000 \
            e7_det2f_norm400 e7_det2f_norm1000; do
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/evaluate.py \
    --ckpt "runs/$name/best_current.pt" --split unseen --kcl-vsource \
    --output "runs/$name/unseen_source_kcl.json" \
    >"allocation_logs/e7/${name}_unseen_eval.log" 2>&1
done

winner=$("$PYTHON_BIN" scripts/summarize_e7.py --select)
echo "selected=$winner"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/evaluate.py \
  --ckpt "runs/$winner/best_current.pt" --split test --kcl-vsource \
  --output "runs/$winner/test_source_kcl.json" \
  2>&1 | tee "allocation_logs/e7/${winner}_test_eval.log"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/current_diagnostics.py \
  --ckpt "runs/$winner/best_current.pt" --split test \
  --output "runs/$winner/current_diagnostics_test.json" \
  >"allocation_logs/e7/${winner}_current_diagnostics.log" 2>&1
"$PYTHON_BIN" scripts/summarize_e7.py \
  2>&1 | tee "allocation_logs/e7/summary.txt"
