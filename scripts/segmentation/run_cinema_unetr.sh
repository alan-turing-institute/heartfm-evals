#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================="
echo " Running CineMA UNetR"
echo "=============================="
python "$SCRIPT_DIR/acdc_cinema_unetr_segmentation.py"
echo ""
echo "============================== CineMA UNetR complete =============================="
