#!/usr/bin/env bash
#SBATCH --job-name=heartfm-cls-cinema-finetune
#SBATCH --array=0-5
#SBATCH --output=logs/classification/%A_%a.out
#SBATCH --error=logs/classification/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# CineMA finetune classification array job.
# 3 datasets × 2 poolings = 6 tasks (indices 0–5).
#
# Within-dataset layout (2 configs each):
#   0  cls
#   1  gap

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

source .venv/bin/activate

SCRIPT="scripts/classification/run_classification.py"
DATASETS=(acdc mnm mnm2)
POOLINGS=(cls gap)

N_CONFIGS=2

dataset_idx=$(( SLURM_ARRAY_TASK_ID / N_CONFIGS ))
config_idx=$(( SLURM_ARRAY_TASK_ID % N_CONFIGS ))

DATASET="${DATASETS[$dataset_idx]}"
POOLING="${POOLINGS[$config_idx]}"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} backbone=cinema eval=finetune pooling=${POOLING} ==="

python "$SCRIPT" \
    --dataset   "$DATASET" \
    --backbone  cinema \
    --eval-mode finetune \
    --pooling   "$POOLING"
