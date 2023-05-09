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

DATA_FPATH="$HOME/data/shared/lang2ltl/data/composed/split_raito0.6_seed42_composed_nsamples10000_seed42_base_symbolic_batch12_perm.pkl"
MODEL_FPATH="$HOME/data/shared/lang2ltl/model"

python $HOME/lang2ltl/s2s_hf_transformers.py --data_fpath $DATA_FPATH --model_dpath $MODEL_FPATH --model t5-base
