#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

#SBATCH --qos turing
#SBATCH --account vjgo8416-heartfm
#SBATCH --time 03:00:0
#SBATCH --nodes 1 
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-gpu 36
#SBATCH --mem 16G
#SBATCH --ntasks-per-node 1
#SBATCH --job-name cinema_unetr_segmentation
#SBATCH --output logs_slurm/cinema_unetr_segmentation_%j.log
# #SBATCH --reservation vjgo8416-heartfm

echo "--------------------------------------"
echo
echo
echo "New job: ${SLURM_JOB_ID}"
echo "--------------------------------------"

module purge
module load baskerville
module load Python CUDA

# for hpc training, set cache dirs as part of project folder
export PIP_CACHE_DIR=/bask/projects/v/vjgo8416-heartfm/.cache/pip
export HF_HOME=/bask/projects/v/vjgo8416-heartfm/.cache/huggingface

python -m .venv
source .venv/bin/activate
echo $(which python)

python -m pip install -e .

# Run ViT pretraining
python heartfm/vit_pretrain_hydra.py

echo "--------------------------------------"
echo "Job completed: ${SLURM_JOB_ID}"
echo "--------------------------------------"
