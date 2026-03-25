#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================="
echo " Running SAM 3"
echo "=============================="
python "$SCRIPT_DIR/acdc_sam3_dense_segmentation.py"
echo ""
echo "============================== SAM 3 complete =============================="
