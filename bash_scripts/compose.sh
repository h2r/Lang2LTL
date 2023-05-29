#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=99G
#SBATCH --time=99:00:00

#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

conda deactivate
module load anaconda/2022.05
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate lang2ltl

BASE_FPATH="$HOME/data/shared/lang2ltl/data/symbolic_batch12_perm.csv"
COMPOSED_FPATH="$HOME/data/shared/lang2ltl/data/composed"
NSAMPLES=1250000
UTT_SPLIT_RATIO=0.3
TEST_SPLIT_RATIO=0.6

python $HOME/lang2ltl/dataset_composed_new.py --base_fpath $BASE_FPATH --composed_dpath $COMPOSED_FPATH --nsamples $NSAMPLES --utt_split_ratio $UTT_SPLIT_RATIO  --test_split_ratio $TEST_SPLIT_RATIO
