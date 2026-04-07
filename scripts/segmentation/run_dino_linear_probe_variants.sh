#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DINO_VARIANTS=(
    "dinov3_vits16"
    "dinov3_vitb16"
    "dinov3_vitl16"
)

echo "=============================="
echo " Running DINOv3 linear probe variants"
echo "=============================="
for model in "${DINO_VARIANTS[@]}"; do
    echo ""
    echo ">>> DINOv3: $model"
    echo "------------------------------"
    python "$SCRIPT_DIR/acdc_dino_dense_linear_probe_segmentation.py" --model "$model"
    echo "Done: $model"
done

echo ""
echo "============================== All DINOv3 linear probe variants complete =============================="
