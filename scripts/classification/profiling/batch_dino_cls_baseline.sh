#!/usr/bin/env bash
#SBATCH --job-name=heartfm-dino-cls-baseline
#SBATCH --output=logs/classification/profile_%j_dino_cls_baseline.out
#SBATCH --error=logs/classification/profile_%j_dino_cls_baseline.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Baseline DINOv3 ViT-B/16 classification run — no OMP_NUM_THREADS set.
# Mirrors batch_run_dino_finetune_classification.sh task 2 (acdc + vitb16 + gap).
# Pair with batch_dino_cls_omp8.sh to measure the thread-count effect.

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

source .venv/bin/activate

echo "=== Baseline DINOv3 ViT-B/16 classification (no OMP_NUM_THREADS) ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CPUs allocated: ${SLURM_CPUS_ON_NODE:-unknown}"

python scripts/classification/run_classification.py \
    --dataset           acdc \
    --backbone          dinov3 \
    --dinov3-model-name dinov3_vitb16 \
    --eval-mode         finetune \
    --pooling           gap
