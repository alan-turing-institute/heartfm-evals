#!/usr/bin/env bash
# Run all logreg and frozen-finetune classification experiments across all datasets.
# SAM uses gap only (no CLS token); CineMA and DINOv3 run both cls and gap.

set -euo pipefail

SCRIPT="scripts/classification/run_classification.py"
DATASETS=(acdc)

run() {
    echo "=== $* ==="
    python "$SCRIPT" "$@"
}

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "========== Dataset: $dataset =========="

    # ── CineMA ──
    for pooling in cls gap; do
        run --dataset "$dataset" --backbone cinema --eval-mode logreg   --pooling "$pooling"
        # run --dataset "$dataset" --backbone cinema --eval-mode finetune --pooling "$pooling"
    done

    # ── DINOv3 ──
    for model in dinov3_vits16 dinov3_vitb16; do
        for pooling in cls gap; do
            run --dataset "$dataset" --backbone dinov3 --dinov3-model-name "$model" --eval-mode logreg   --pooling "$pooling"
            # run --dataset "$dataset" --backbone dinov3 --dinov3-model-name "$model" --eval-mode finetune --pooling "$pooling"
        done
    done

    # # ── SAM (gap only — no CLS token) ──
    # for model in facebook/sam-vit-base facebook/sam-vit-large facebook/sam-vit-huge; do
    #     run --dataset "$dataset" --backbone sam --sam-model-id "$model" --eval-mode logreg   --pooling gap
    #     # run --dataset "$dataset" --backbone sam --sam-model-id "$model" --eval-mode finetune --pooling gap
    # done
done

echo "All logreg experiments complete."
