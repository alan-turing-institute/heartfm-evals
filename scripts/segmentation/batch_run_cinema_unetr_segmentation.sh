#!/usr/bin/env bash
#SBATCH --job-name=heartfm-seg-cinema-unetr
#SBATCH --array=0-2
#SBATCH --output=logs/segmentation/%A_%a.out
#SBATCH --error=logs/segmentation/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# CineMA unetr segmentation array job.
# 3 datasets × 1 model = 3 tasks (indices 0–2).

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

source .venv/bin/activate

SCRIPT="scripts/segmentation/run_segmentation.py"
DATASETS=(acdc mnm mnm2)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} backbone=cinema decoder=unetr ==="

python "$SCRIPT" \
    --dataset  "$DATASET" \
    --backbone cinema \
    --decoder  unetr
