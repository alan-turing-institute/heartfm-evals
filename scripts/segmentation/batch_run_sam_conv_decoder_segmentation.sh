#!/usr/bin/env bash
#SBATCH --job-name=heartfm-seg-sam-conv-decoder
#SBATCH --array=0-8
#SBATCH --output=logs/segmentation/%A_%a.out
#SBATCH --error=logs/segmentation/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# SAM v1 conv_decoder segmentation array job.
# 3 datasets × 3 models = 9 tasks (indices 0–8).
#
# Within-dataset layout (3 configs each):
#   0  sam-vit-base
#   1  sam-vit-large
#   2  sam-vit-huge

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

source .venv/bin/activate

SCRIPT="scripts/segmentation/run_segmentation.py"
DATASETS=(acdc mnm mnm2)
MODELS=(facebook/sam-vit-base facebook/sam-vit-large facebook/sam-vit-huge)

N_CONFIGS=3

dataset_idx=$(( SLURM_ARRAY_TASK_ID / N_CONFIGS ))
config_idx=$(( SLURM_ARRAY_TASK_ID % N_CONFIGS ))

DATASET="${DATASETS[$dataset_idx]}"
MODEL="${MODELS[$config_idx]}"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} backbone=sam model=${MODEL} decoder=conv_decoder ==="

python "$SCRIPT" \
    --dataset      "$DATASET" \
    --backbone     sam \
    --sam-model-id "$MODEL" \
    --decoder      conv_decoder
