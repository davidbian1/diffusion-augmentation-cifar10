#!/bin/bash

#SBATCH --job-name=diffusion_aug
#SBATCH --output=logs/diffusion_%j.out
#SBATCH --error=logs/diffusion_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00


echo "Job started"
echo "Running on: $(hostname)"


# Load CUDA
module load cuda


# Activate environment
source .venv/bin/activate


# Check GPU
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"


# Run experiment
python -m src.preprocess