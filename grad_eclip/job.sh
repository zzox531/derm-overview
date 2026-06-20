#!/bin/bash
#SBATCH --partition=hpc
#SBATCH --mem=32G
#SBATCH --gres=gpu:titanx:1
#SBATCH --time=00:15:00
#SBATCH --job-name=dermlip_grad
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# This is just an example 
# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh   # adjust path if needed


conda activate dermlip_entropy

# python3 mult_map.py ham --model hf-hub:redlessone/DermLIP_ViT-B-16
python3 gen_map_dermlip.py pad