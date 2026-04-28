#!/usr/bin/env bash
# Submit all SAM2 classification jobs (logreg + finetune, all 4 model variants,
# all 3 datasets).
#
# Array layout per script:
#   3 datasets × 4 models = 12 tasks (indices 0–11)
#
#   Within-dataset layout (4 configs each):
#     0  sam2.1-hiera-tiny
#     1  sam2.1-hiera-small
#     2  sam2.1-hiera-base-plus
#     3  sam2.1-hiera-large
#
# Run from the repo root:
#   bash scripts/submit_classification_sam2.sh

set -euo pipefail

CLS="scripts/classification"

echo "=== SAM2 Classification — logreg ==="
sbatch --array=0-11 "$CLS/batch_run_sam2_logreg_classification.sh"

echo ""
echo "=== SAM2 Classification — finetune ==="
sbatch --array=0-11 "$CLS/batch_run_sam2_finetune_classification.sh"

echo ""
echo "All SAM2 classification jobs submitted."
