#!/bin/bash
#SBATCH --job-name=qwen3       # Tên job
#SBATCH --output=log_%j.txt         # File log (chứa %j = job ID)
#SBATCH --gres=gpu:1           # Yêu cầu 1 GPU A100
#SBATCH --mem=8G                    # RAM 8GB

# Nếu cluster cần module, load CUDA hoặc Python
source /data2/shared/apps/conda/etc/profile.d/conda.sh
conda activate env_llm

# Chạy file Python
python train_sft.py