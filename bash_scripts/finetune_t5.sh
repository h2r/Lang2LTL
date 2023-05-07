#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=99:00:00
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --mem-per-gpu=32G

#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

module load anaconda/2022.05
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate lang2ltl

data_fpath = "$HOME/data/shared/lang2ltl/data/composed_formula/composed_formula_symbolic_batch12_noperm_900_42_fold0.pkl"
model_dpath = "$HOME/data/shared/lang2ltl/model"

python $HOME/lang2ltl/s2s_hf_transformers.py --data $data_fpath --model_dpath $model_dapth --model t5-base
