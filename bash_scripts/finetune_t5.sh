#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=99:00:00
#SBATCH -p gpu-he
#SBATCH --gpus-per-node=v100:2
#SBATCH --mem-per-gpu=64G

#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

module load anaconda/2022.05
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate lang2ltl

python s2s_hf_transformers.py --data data/holdout_split_batch12_perm/composed_formula/composed_formula_symbolic_batch12_noperm_900_42_fold0.pkl --model t5-base
