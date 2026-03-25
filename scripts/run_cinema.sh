#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================="
echo " Running CineMA"
echo "=============================="
python "$SCRIPT_DIR/acdc_cinema_dense_segmentation.py"
echo ""
echo "============================== CineMA complete =============================="
