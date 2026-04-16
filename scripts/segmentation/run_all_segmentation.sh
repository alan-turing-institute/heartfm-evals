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

    # ── SAM2 ──
    for model_id in facebook/sam2.1-hiera-tiny facebook/sam2.1-hiera-small facebook/sam2.1-hiera-base-plus facebook/sam2.1-hiera-large; do
        for decoder in "${DECODERS[@]}"; do
            run --dataset "$dataset" --backbone sam2 --sam2-model-id "$model_id" --decoder "$decoder"
        done
    done
done

echo ""
echo "All segmentation experiments complete."
