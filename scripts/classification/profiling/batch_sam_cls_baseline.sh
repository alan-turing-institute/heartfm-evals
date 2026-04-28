#!/usr/bin/env bash
#SBATCH --job-name=heartfm-sam-cls-baseline
#SBATCH --output=logs/classification/profile_%j_sam_cls_baseline.out
#SBATCH --error=logs/classification/profile_%j_sam_cls_baseline.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Baseline SAM ViT-Large classification run — no OMP_NUM_THREADS set.
# Mirrors batch_run_sam_finetune_classification.sh task 1 (acdc + sam-vit-large).
# Pair with batch_sam_cls_omp8.sh to measure the thread-count effect.
# Note: classification_feature_cache/acdc/sam_vit_large is already populated,
# so this run goes straight to training (warm cache).

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

source .venv/bin/activate

echo "=== Baseline SAM ViT-Large classification (no OMP_NUM_THREADS) ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CPUs allocated: ${SLURM_CPUS_ON_NODE:-unknown}"

python scripts/classification/run_classification.py \
    --dataset      acdc \
    --backbone     sam \
    --sam-model-id facebook/sam-vit-large \
    --eval-mode    finetune \
    --pooling      gap
