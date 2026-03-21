#!/bin/bash
#SBATCH --job-name=qwen3
#SBATCH --output=log_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0

source /data2/shared/apps/conda/etc/profile.d/conda.sh
conda activate env_llm

python "$1"