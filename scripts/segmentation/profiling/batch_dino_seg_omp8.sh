#!/usr/bin/env bash
#SBATCH --job-name=heartfm-dino-seg-omp8
#SBATCH --output=logs/segmentation/profile_%j_dino_seg_omp8.out
#SBATCH --error=logs/segmentation/profile_%j_dino_seg_omp8.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# OMP_NUM_THREADS=8 DINOv3 ViT-B/16 segmentation run.
# Mirrors batch_run_dino_linear_probe_segmentation.sh task 1 (acdc + vitb16).
# Pair with batch_dino_seg_baseline.sh to measure the thread-count effect.

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

source .venv/bin/activate

echo "=== OMP_NUM_THREADS=8 DINOv3 ViT-B/16 segmentation ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CPUs allocated: ${SLURM_CPUS_ON_NODE:-unknown}"

python scripts/segmentation/run_segmentation.py \
    --dataset           acdc \
    --backbone          dinov3 \
    --dinov3-model-name dinov3_vitb16 \
    --decoder           linear_probe
