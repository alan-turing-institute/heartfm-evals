#!/usr/bin/env bash
#SBATCH --job-name=heartfm-cinema-cls-omp8
#SBATCH --output=logs/classification/profile_%j_cinema_cls_omp8.out
#SBATCH --error=logs/classification/profile_%j_cinema_cls_omp8.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# OMP_NUM_THREADS=8 CineMA classification run.
# Mirrors batch_run_cinema_finetune_classification.sh for acdc + gap pooling.
# Pair with batch_cinema_cls_baseline.sh to measure the thread-count effect.

set -euo pipefail

module load cuda/12.6
module load gcc-native/12.3

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

source .venv/bin/activate

echo "=== OMP_NUM_THREADS=8 CineMA classification ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CPUs allocated: ${SLURM_CPUS_ON_NODE:-unknown}"

python scripts/classification/run_classification.py \
    --dataset   acdc \
    --backbone  cinema \
    --eval-mode finetune \
    --pooling   gap
