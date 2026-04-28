#!/usr/bin/env bash
#SBATCH --job-name=heartfm-dino-cls-omp8
#SBATCH --output=logs/classification/profile_%j_dino_cls_omp8.out
#SBATCH --error=logs/classification/profile_%j_dino_cls_omp8.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# OMP_NUM_THREADS=8 DINOv3 ViT-B/16 classification run.
# Mirrors batch_run_dino_finetune_classification.sh task 2 (acdc + vitb16 + gap).
# Pair with batch_dino_cls_baseline.sh to measure the thread-count effect.

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

source .venv/bin/activate

echo "=== OMP_NUM_THREADS=8 DINOv3 ViT-B/16 classification ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CPUs allocated: ${SLURM_CPUS_ON_NODE:-unknown}"

python scripts/classification/run_classification.py \
    --dataset           acdc \
    --backbone          dinov3 \
    --dinov3-model-name dinov3_vitb16 \
    --eval-mode         finetune \
    --pooling           gap
