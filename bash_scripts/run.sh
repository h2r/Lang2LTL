#!/usr/bin/bash
#SBATCH -n 365
#SBATCH --mem=199G
#SBATCH -t 99:00:00

#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

module load anaconda/2022.05
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate my_env

python
