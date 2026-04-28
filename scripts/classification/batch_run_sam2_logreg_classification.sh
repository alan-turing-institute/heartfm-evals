#!/usr/bin/env bash
#SBATCH --job-name=heartfm-cls-sam2-logreg
#SBATCH --array=0-11
#SBATCH --output=logs/classification/%A_%a.out
#SBATCH --error=logs/classification/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# SAM2 logreg classification array job — gap pooling only (no CLS token).
# 3 datasets × 4 models = 12 tasks (indices 0–11).
#
# Within-dataset layout (4 configs each):
#   0  sam2.1-hiera-tiny
#   1  sam2.1-hiera-small
#   2  sam2.1-hiera-base-plus
#   3  sam2.1-hiera-large

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

source .venv/bin/activate

SCRIPT="scripts/classification/run_classification.py"
DATASETS=(acdc mnm mnm2)
MODELS=(
    facebook/sam2.1-hiera-tiny
    facebook/sam2.1-hiera-small
    facebook/sam2.1-hiera-base-plus
    facebook/sam2.1-hiera-large
)

N_CONFIGS=4

dataset_idx=$(( SLURM_ARRAY_TASK_ID / N_CONFIGS ))
config_idx=$(( SLURM_ARRAY_TASK_ID % N_CONFIGS ))

DATASET="${DATASETS[$dataset_idx]}"
MODEL="${MODELS[$config_idx]}"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} backbone=sam2 model=${MODEL} eval=logreg pooling=gap ==="

python "$SCRIPT" \
    --dataset       "$DATASET" \
    --backbone      sam2 \
    --sam2-model-id "$MODEL" \
    --eval-mode     logreg \
    --pooling       gap
