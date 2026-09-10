#!/usr/bin/env bash
# Run all segmentation experiments across all datasets, backbones, and decoders.
#
# Usage:
#   bash scripts/segmentation/run_all_segmentation.sh

set -euo pipefail

SCRIPT="scripts/segmentation/run_segmentation.py"
DATASETS=(acdc mnm mnm2)
DECODERS=(linear_probe conv_decoder unetr)

run() {
    echo "=== $* ==="
    python "$SCRIPT" "$@"
}

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "========== Dataset: $dataset =========="

    # ── DINOv3 ──
    for model in dinov3_vits16 dinov3_vitb16 dinov3_vitl16; do
        for decoder in "${DECODERS[@]}"; do
            run --dataset "$dataset" --backbone dinov3 --dinov3-model-name "$model" --decoder "$decoder"
        done
    done

    # ── CineMA ──
    for decoder in "${DECODERS[@]}"; do
        run --dataset "$dataset" --backbone cinema --decoder "$decoder"
    done

    # ── SAM v1 ──
    for model_id in facebook/sam-vit-base facebook/sam-vit-large facebook/sam-vit-huge; do
        for decoder in "${DECODERS[@]}"; do
            run --dataset "$dataset" --backbone sam --sam-model-id "$model_id" --decoder "$decoder"
        done
    done
done

echo ""
echo "All segmentation experiments complete."
