#!/usr/bin/env bash
#SBATCH --job-name=heartfm-cls
#SBATCH --array=0-65
#SBATCH --output=logs/classification/%A_%a.out
#SBATCH --error=logs/classification/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# SLURM array equivalent of run_all_classification.sh.
# 3 datasets × 22 configs = 66 tasks (indices 0–65).
# Tasks whose output JSON already exists are skipped automatically by the
# Python script, so completed experiments cost only a few seconds.
#
# Within-dataset layout (22 configs each):
#   0– 3  CineMA        (cls/gap × logreg/finetune)
#   4–15  DINOv3 vits16 (cls/gap × logreg/finetune)
#   8–11  DINOv3 vitb16 (cls/gap × logreg/finetune)   [offset +8 from 4]
#  12–15  DINOv3 vitl16 (cls/gap × logreg/finetune)   [offset +12 from 4]
#  16–21  SAM gap-only  (3 models × logreg/finetune)

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

source .venv/bin/activate

SCRIPT="scripts/classification/run_classification.py"
DATASETS=(acdc mnm mnm2)

# ── one entry per within-dataset config (indices 0–21) ──
BACKBONES=(
    cinema cinema cinema cinema
    dinov3 dinov3 dinov3 dinov3
    dinov3 dinov3 dinov3 dinov3
    dinov3 dinov3 dinov3 dinov3
    sam    sam    sam    sam    sam    sam
)
EVAL_MODES=(
    logreg finetune logreg finetune
    logreg finetune logreg finetune
    logreg finetune logreg finetune
    logreg finetune logreg finetune
    logreg finetune logreg finetune logreg finetune
)
POOLINGS=(
    cls cls gap gap
    cls cls gap gap
    cls cls gap gap
    cls cls gap gap
    gap gap gap gap gap gap
)
EXTRA_FLAGS=(
    "" "" "" ""
    "--dinov3-model-name dinov3_vits16" "--dinov3-model-name dinov3_vits16"
    "--dinov3-model-name dinov3_vits16" "--dinov3-model-name dinov3_vits16"
    "--dinov3-model-name dinov3_vitb16" "--dinov3-model-name dinov3_vitb16"
    "--dinov3-model-name dinov3_vitb16" "--dinov3-model-name dinov3_vitb16"
    "--dinov3-model-name dinov3_vitl16" "--dinov3-model-name dinov3_vitl16"
    "--dinov3-model-name dinov3_vitl16" "--dinov3-model-name dinov3_vitl16"
    "--sam-model-id facebook/sam-vit-base"  "--sam-model-id facebook/sam-vit-base"
    "--sam-model-id facebook/sam-vit-large" "--sam-model-id facebook/sam-vit-large"
    "--sam-model-id facebook/sam-vit-huge"  "--sam-model-id facebook/sam-vit-huge"
)

N_CONFIGS=22

dataset_idx=$(( SLURM_ARRAY_TASK_ID / N_CONFIGS ))
config_idx=$(( SLURM_ARRAY_TASK_ID % N_CONFIGS ))

DATASET="${DATASETS[$dataset_idx]}"
BACKBONE="${BACKBONES[$config_idx]}"
EVAL_MODE="${EVAL_MODES[$config_idx]}"
POOLING="${POOLINGS[$config_idx]}"
EXTRA="${EXTRA_FLAGS[$config_idx]}"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} backbone=${BACKBONE} eval=${EVAL_MODE} pooling=${POOLING} ${EXTRA} ==="

# shellcheck disable=SC2086
python "$SCRIPT" \
    --dataset   "$DATASET" \
    --backbone  "$BACKBONE" \
    --eval-mode "$EVAL_MODE" \
    --pooling   "$POOLING" \
    $EXTRA
