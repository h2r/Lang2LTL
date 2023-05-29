#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=1:00:00
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --mem-per-gpu=32G

#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

conda deactivate
module load anaconda/2022.05
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate lang2ltl

python $HOME/lang2ltl/llama_example.py
