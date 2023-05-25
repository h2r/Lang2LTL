#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=399:00:00
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --array=0-6

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-%A_%a.err
#SBATCH -o sbatch_out/arrayjob-%A_%a.out

module load anaconda/2022.05
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate lang2ltl

runs=( 0 )
end_indices=( 25000 50000 75000 100000 200000 300000 400000 )

i=`expr $SLURM_ARRAY_TASK_ID % ${#runs[@]}`
j=`expr $SLURM_ARRAY_TASK_ID / ${#runs[@]}`
k=`expr $j % ${#end_indices[@]}`

run=${runs[$i]}
end_idx=${end_indices[$k]}

nsamples="1000000"
data_fpath="${HOME}/data/shared/lang2ltl/data/composed/split-sample_nsamples${nsamples}_raito0.3-0.6_seed42_symbolic_batch12_perm.pkl"
model_dpath="${HOME}/data/shared/lang2ltl/models/model_${nsamples}_run${run}_endidx${end_idx}"
python $HOME/lang2ltl/s2s_hf_transformers.py --data_fpath $data_fpath --end_idx $end_idx --model_dpath $model_dpath --model t5-base
