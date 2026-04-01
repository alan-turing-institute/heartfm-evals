#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DINO_VARIANTS=(
    "dinov3_vits16"
    "dinov3_vitb16"
    "dinov3_vitl16"
)

echo "=============================="
echo " Running DINOv3 UNetR variants"
echo "=============================="
for model in "${DINO_VARIANTS[@]}"; do
    echo ""
    echo ">>> DINOv3 UNetR: $model"
    echo "------------------------------"
    python "$SCRIPT_DIR/acdc_dino_unetr_segmentation.py" --model "$model"
    echo "Done: $model"
done

echo ""
echo "============================== All DINOv3 UNetR variants complete =============================="
