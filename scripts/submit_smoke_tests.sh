#!/usr/bin/env bash
# Submit smoke-test runs for all classification and segmentation batch scripts.
# Only the smallest model variant is submitted for each backbone:
#
#   CineMA  — only model, all tasks
#   DINOv3  — dinov3_vits16 only
#   SAM     — facebook/sam-vit-base only   (classification)
#   SAM2    — facebook/sam2.1-hiera-tiny   (segmentation)
#
# Run from the repo root:
#   bash scripts/submit_smoke_tests.sh

set -euo pipefail

CLS="scripts/classification"
SEG="scripts/segmentation"

echo "=== Classification ==="

# CineMA: 3 datasets × 2 poolings = 6 tasks, all are CineMA
# sbatch --array=0-5   "$CLS/batch_run_cinema_finetune_classification.sh"
sbatch --array=0-5   "$CLS/batch_run_cinema_logreg_classification.sh"

# DINOv3: vits16 = config_idx 0–1 per dataset (cls/gap)
# acdc→0,1  mnm→6,7  mnm2→12,13
# sbatch --array=0,1,6,7,12,13 "$CLS/batch_run_dino_finetune_classification.sh"
sbatch --array=0,1,6,7,12,13 "$CLS/batch_run_dino_logreg_classification.sh"

# SAM: sam-vit-base = config_idx 0 per dataset
# acdc→0  mnm→3  mnm2→6
# sbatch --array=0,3,6 "$CLS/batch_run_sam_finetune_classification.sh"
sbatch --array=0,3,6 "$CLS/batch_run_sam_logreg_classification.sh"

echo ""
echo "=== Segmentation ==="

# CineMA: 3 datasets, no sub-models — submit all
sbatch --array=0-2 "$SEG/batch_run_cinema_linear_probe_segmentation.sh"
sbatch --array=0-2 "$SEG/batch_run_cinema_conv_decoder_segmentation.sh"
sbatch --array=0-2 "$SEG/batch_run_cinema_unetr_segmentation.sh"

# DINOv3: vits16 = config_idx 0 per dataset
# acdc→0  mnm→3  mnm2→6
sbatch --array=0,3,6 "$SEG/batch_run_dino_linear_probe_segmentation.sh"
sbatch --array=0,3,6 "$SEG/batch_run_dino_conv_decoder_segmentation.sh"
sbatch --array=0,3,6 "$SEG/batch_run_dino_unetr_segmentation.sh"

# SAM2: hiera-tiny = config_idx 0 per dataset
# acdc→0  mnm→4  mnm2→8
sbatch --array=0,4,8 "$SEG/batch_run_sam2_linear_probe_segmentation.sh"
sbatch --array=0,4,8 "$SEG/batch_run_sam2_conv_decoder_segmentation.sh"
sbatch --array=0,4,8 "$SEG/batch_run_sam2_unetr_segmentation.sh"

echo ""
echo "All smoke-test jobs submitted."
