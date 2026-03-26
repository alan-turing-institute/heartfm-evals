#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SAM_VARIANTS=(
    "facebook/sam-vit-base"
    "facebook/sam-vit-large"
    "facebook/sam-vit-huge"
)

echo "=============================="
echo " Running SAM UNetR variants"
echo "=============================="
for model in "${SAM_VARIANTS[@]}"; do
    echo ""
    echo ">>> SAM UNetR: $model"
    echo "------------------------------"
    python "$SCRIPT_DIR/acdc_sam_unetr_segmentation.py" --model "$model"
    echo "Done: $model"
done

echo ""
echo "============================== All SAM UNetR variants complete =============================="
