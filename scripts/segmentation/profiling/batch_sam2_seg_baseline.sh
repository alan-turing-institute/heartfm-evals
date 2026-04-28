#!/usr/bin/env bash
#SBATCH --job-name=heartfm-sam2-seg-baseline
#SBATCH --output=logs/segmentation/profile_%j_sam2_seg_baseline.out
#SBATCH --error=logs/segmentation/profile_%j_sam2_seg_baseline.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Baseline SAM2 segmentation run — no OMP_NUM_THREADS set.
# Mirrors batch_run_sam2_linear_probe_segmentation.sh for acdc + sam2.1-hiera-small.
# Pair with batch_sam2_seg_omp8.sh to measure the thread-count effect.
# Note: feature_cache/acdc/sam2_1_hiera_small is already populated,
# so this run goes straight to training (warm cache).

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

source .venv/bin/activate

echo "=== Baseline SAM2 hiera-small segmentation (no OMP_NUM_THREADS) ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CPUs allocated: ${SLURM_CPUS_ON_NODE:-unknown}"

python scripts/segmentation/run_segmentation.py \
    --dataset       acdc \
    --backbone      sam2 \
    --sam2-model-id facebook/sam2.1-hiera-small \
    --decoder       linear_probe
