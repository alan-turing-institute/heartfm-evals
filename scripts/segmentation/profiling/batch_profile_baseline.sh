#!/usr/bin/env bash
#SBATCH --job-name=heartfm-profile-baseline
#SBATCH --output=logs/segmentation/profile_%j_baseline.out
#SBATCH --error=logs/segmentation/profile_%j_baseline.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Baseline profiling run — no OMP_NUM_THREADS set (default: all available cores).
# Pair with batch_profile_omp8.sh to measure the thread-count effect.
# Runs: acdc + sam2.1-hiera-small (second smallest config).

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

source .venv/bin/activate

echo "=== Baseline profile (no OMP_NUM_THREADS) ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CPUs allocated: ${SLURM_CPUS_ON_NODE:-unknown}"

python scripts/segmentation/profiling/profile_feature_extraction.py \
    --backbone      sam2 \
    --sam2-model-id facebook/sam2.1-hiera-small \
    --dataset       acdc \
    --n-samples     20
