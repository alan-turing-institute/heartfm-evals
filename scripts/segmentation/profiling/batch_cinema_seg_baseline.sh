#!/usr/bin/env bash
#SBATCH --job-name=heartfm-cinema-seg-baseline
#SBATCH --output=logs/segmentation/profile_%j_cinema_seg_baseline.out
#SBATCH --error=logs/segmentation/profile_%j_cinema_seg_baseline.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Baseline CineMA segmentation run — no OMP_NUM_THREADS set.
# Mirrors batch_run_cinema_linear_probe_segmentation.sh for acdc.
# Pair with batch_cinema_seg_omp8.sh to measure the thread-count effect.

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

source .venv/bin/activate

echo "=== Baseline CineMA segmentation (no OMP_NUM_THREADS) ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CPUs allocated: ${SLURM_CPUS_ON_NODE:-unknown}"

python scripts/segmentation/run_segmentation.py \
    --dataset  acdc \
    --backbone cinema \
    --decoder  linear_probe
