#!/usr/bin/env bash
# Run all logreg and frozen-finetune classification experiments across all datasets.
# Both SAM families use gap only (no CLS token); CineMA and DINOv3 run cls and gap.

set -euo pipefail

SCRIPT="scripts/classification/run_classification.py"
DATASETS=(acdc mnm mnm2)

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
        run --dataset "$dataset" --backbone cinema --eval-mode finetune --pooling "$pooling"
    done

    # ── DINOv3 ──
    for model in dinov3_vits16 dinov3_vitb16 dinov3_vitl16; do
        for pooling in cls gap; do
            run --dataset "$dataset" --backbone dinov3 --dinov3-model-name "$model" --eval-mode logreg   --pooling "$pooling"
            run --dataset "$dataset" --backbone dinov3 --dinov3-model-name "$model" --eval-mode finetune --pooling "$pooling"
        done
    done

    # ── SAM v1 (gap only — no CLS token) ──
    for model in facebook/sam-vit-base facebook/sam-vit-large facebook/sam-vit-huge; do
        run --dataset "$dataset" --backbone sam --sam-model-id "$model" --eval-mode logreg   --pooling gap
        run --dataset "$dataset" --backbone sam --sam-model-id "$model" --eval-mode finetune --pooling gap
    done

    # ── SAM2 (gap only — Hiera has no CLS token) ──
    for model in facebook/sam2.1-hiera-small facebook/sam2.1-hiera-base-plus facebook/sam2.1-hiera-large; do
        run --dataset "$dataset" --backbone sam2 --sam2-model-id "$model" --eval-mode logreg   --pooling gap
        run --dataset "$dataset" --backbone sam2 --sam2-model-id "$model" --eval-mode finetune --pooling gap
    done
done

echo "All experiments complete."
