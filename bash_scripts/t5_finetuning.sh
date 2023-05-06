#!/bin/bash

#SBATCH --job-name=t5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-gpu=24G
#SBATCH --time=10:00:00
#SBATCH -p 3090-gcondo --gres=gpu:1

module load miniconda
conda activate ~/anaconda/lang2ltl
/users/zyang157/anaconda/lang2ltl/bin/python s2s_hf_transformers.py --data data/holdout_split_batch12_perm/symbolic_batch12_perm_ltl_formula_9_42_fold0.pkl --model t5-base
