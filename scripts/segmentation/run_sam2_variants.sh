#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SAM2_VARIANTS=(
    "facebook/sam2.1-hiera-tiny"
    "facebook/sam2.1-hiera-small"
    "facebook/sam2.1-hiera-base-plus"
    "facebook/sam2.1-hiera-large"
)

echo "=============================="
echo " Running SAM 2.1 variants"
echo "=============================="
for model in "${SAM2_VARIANTS[@]}"; do
    echo ""
    echo ">>> SAM 2.1: $model"
    echo "------------------------------"
    python "$SCRIPT_DIR/acdc_sam2_dense_segmentation.py" --model "$model"
    echo "Done: $model"
done

echo ""
echo "============================== All SAM 2.1 variants complete =============================="
