#!/usr/bin/env bash
#SBATCH --job-name=heartfm-cls-dino-logreg
#SBATCH --array=0-17
#SBATCH --output=logs/classification/%A_%a.out
#SBATCH --error=logs/classification/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# DINOv3 logreg classification array job.
# 3 datasets × 6 configs = 18 tasks (indices 0–17).
#
# Within-dataset layout (6 configs each):
#   0–1  dinov3_vits16 (cls / gap)
#   2–3  dinov3_vitb16 (cls / gap)
#   4–5  dinov3_vitl16 (cls / gap)

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

source .venv/bin/activate

SCRIPT="scripts/classification/run_classification.py"
DATASETS=(acdc mnm mnm2)

MODELS=(
    dinov3_vits16 dinov3_vits16
    dinov3_vitb16 dinov3_vitb16
    dinov3_vitl16 dinov3_vitl16
)
POOLINGS=(cls gap cls gap cls gap)

N_CONFIGS=6

dataset_idx=$(( SLURM_ARRAY_TASK_ID / N_CONFIGS ))
config_idx=$(( SLURM_ARRAY_TASK_ID % N_CONFIGS ))

DATASET="${DATASETS[$dataset_idx]}"
MODEL="${MODELS[$config_idx]}"
POOLING="${POOLINGS[$config_idx]}"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} backbone=dinov3 model=${MODEL} eval=logreg pooling=${POOLING} ==="

python "$SCRIPT" \
    --dataset           "$DATASET" \
    --backbone          dinov3 \
    --dinov3-model-name "$MODEL" \
    --eval-mode         logreg \
    --pooling           "$POOLING"
