#!/bin/bash
# Qwen3-30B-A3B is 61 GB in bf16, so this needs a full H100
#SBATCH --job-name=judge
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
# pre installed copy
export HF_HUB_CACHE=/projects/2/managed_datasets/hf_cache_dir HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun uv run --frozen python judge.py "$1"
