#!/bin/bash

#SBATCH --job-name=t5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1:00:00
#SBATCH -p 3090-gcondo --gres=gpu:1

module load miniconda
module load python/3.9.0
source /gpfs/runtime/opt/miniconda/4.10/etc/profile.d/conda.sh
conda activate ~/anaconda/lang2ltl
nvidia-smi
python llama_example.py