#!/usr/bin/env bash
#SBATCH --job-name=heartfm-dino-seg-baseline
#SBATCH --output=logs/segmentation/profile_%j_dino_seg_baseline.out
#SBATCH --error=logs/segmentation/profile_%j_dino_seg_baseline.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Baseline DINOv3 ViT-B/16 segmentation run — no OMP_NUM_THREADS set.
# Mirrors batch_run_dino_linear_probe_segmentation.sh task 1 (acdc + vitb16).
# Pair with batch_dino_seg_omp8.sh to measure the thread-count effect.

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

source .venv/bin/activate

echo "=== Baseline DINOv3 ViT-B/16 segmentation (no OMP_NUM_THREADS) ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CPUs allocated: ${SLURM_CPUS_ON_NODE:-unknown}"

python scripts/segmentation/run_segmentation.py \
    --dataset           acdc \
    --backbone          dinov3 \
    --dinov3-model-name dinov3_vitb16 \
    --decoder           linear_probe
