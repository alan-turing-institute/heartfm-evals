#!/usr/bin/env bash
#SBATCH --job-name=heartfm-seg-dino-linear-probe
#SBATCH --array=0-8
#SBATCH --output=logs/segmentation/%A_%a.out
#SBATCH --error=logs/segmentation/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# DINOv3 linear_probe segmentation array job.
# 3 datasets × 3 models = 9 tasks (indices 0–8).
#
# Within-dataset layout (3 configs each):
#   0  dinov3_vits16
#   1  dinov3_vitb16
#   2  dinov3_vitl16

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

source .venv/bin/activate

SCRIPT="scripts/segmentation/run_segmentation.py"
DATASETS=(acdc mnm mnm2)
MODELS=(dinov3_vits16 dinov3_vitb16 dinov3_vitl16)

N_CONFIGS=3

dataset_idx=$(( SLURM_ARRAY_TASK_ID / N_CONFIGS ))
config_idx=$(( SLURM_ARRAY_TASK_ID % N_CONFIGS ))

DATASET="${DATASETS[$dataset_idx]}"
MODEL="${MODELS[$config_idx]}"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} backbone=dinov3 model=${MODEL} decoder=linear_probe ==="

python "$SCRIPT" \
    --dataset           "$DATASET" \
    --backbone          dinov3 \
    --dinov3-model-name "$MODEL" \
    --decoder           linear_probe
