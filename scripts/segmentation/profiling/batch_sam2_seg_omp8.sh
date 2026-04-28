#!/usr/bin/env bash
#SBATCH --job-name=heartfm-sam2-seg-omp8
#SBATCH --output=logs/segmentation/profile_%j_sam2_seg_omp8.out
#SBATCH --error=logs/segmentation/profile_%j_sam2_seg_omp8.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# OMP_NUM_THREADS=8 SAM2 segmentation run.
# Mirrors batch_run_sam2_linear_probe_segmentation.sh for acdc + sam2.1-hiera-small.
# Pair with batch_sam2_seg_baseline.sh to measure the thread-count effect.
# Note: feature_cache/acdc/sam2_1_hiera_small is already populated,
# so this run goes straight to training (warm cache).

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

source .venv/bin/activate

echo "=== OMP_NUM_THREADS=8 SAM2 hiera-small segmentation ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CPUs allocated: ${SLURM_CPUS_ON_NODE:-unknown}"

python scripts/segmentation/run_segmentation.py \
    --dataset       acdc \
    --backbone      sam2 \
    --sam2-model-id facebook/sam2.1-hiera-small \
    --decoder       linear_probe
