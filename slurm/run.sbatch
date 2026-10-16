#!/bin/bash
#SBATCH --job-name=run
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
# compute nodes have no internet, download.py fills the cache from the login node
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun uv run --frozen python run.py "$1"
