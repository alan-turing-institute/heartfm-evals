#!/usr/bin/env bash
#SBATCH --job-name=heartfm-install
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

module load cuda/12.6
module load gcc-native/12.3

uv venv --python=3.12
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv sync

echo "Done!"
