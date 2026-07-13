#!/bin/bash
# Run all three encoding ablations in parallel
# Results will appear on wandb.ai/project/GridFM_Ablation

set -euo pipefail

OUTDIR="/kfs2/projects/gogpt/Ebadmus/Training_new"
CONFIGS=(
    "ablate_encoding_asinh"
    "ablate_encoding_signed_log"
    "ablate_encoding_standardized_pu"
)

echo "=== Submitting 3-way encoding ablation ==="
echo "Dataset: minimal_component (200 feeders)"
echo "Epochs: 5"
echo "Track results: https://wandb.ai/project/GridFM_Ablation"
echo ""

for cfg in "${CONFIGS[@]}"; do
    echo "Submitting $cfg..."
    sbatch \
        -J "ablate_${cfg##*_}" \
        -A view \
        -p standard \
        --ntasks=1 --cpus-per-task=32 --mem=128G \
        -t 02:00:00 \
        --output="${OUTDIR}/runs/ablate_${cfg##*_}_%j.out" \
        "${OUTDIR}/run.sbatch" \
        "${cfg}"
    sleep 2  # stagger submissions
done

echo ""
echo "✓ All 3 ablation configs submitted"
echo "View results in ~10-15 min on wandb.ai:"
echo "  - encoding_asinh (baseline)"
echo "  - encoding_signed_log"
echo "  - encoding_standardized_pu"
echo ""
echo "Compare metrics: V_mae, Ibus_wape, total_loss @ epoch 5"
