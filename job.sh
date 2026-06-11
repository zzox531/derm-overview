#!/bin/bash
#SBATCH --partition=hpc
#SBATCH --mem=32G
#SBATCH --gres=gpu:titanx:1
#SBATCH --job-name=dermlip
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# This is just an example 
# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh   # adjust path if needed


conda activate dermlip_entropy

python3 inference_entropy.py