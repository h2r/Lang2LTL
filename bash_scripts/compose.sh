#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=199G
#SBATCH --time=99:00:00

#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

module load anaconda/2022.05
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate lang2ltl

BASE_FPATH="$HOME/data/shared/lang2ltl/data/symbolic_batch12_perm.csv"
COMPOSED_FPATH="$HOME/data/shared/lang2ltl/data/composed"
NSAMPLES=1000000
RATIO=0.6
SAVE_EVERY=100

python $HOME/lang2ltl/dataset_composed_new.py --base_fpath $BASE_FPATH --composed_dpath $COMPOSED_FPATH --nsamples $NSAMPLES --split_ratio $RATIO --save_every $SAVE_EVERY
