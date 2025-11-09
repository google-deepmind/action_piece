#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=46:00:00
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=zl4789@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate TIGER
cd /scratch/zl4789/action_piece_google

python build_vocab.py   \
    --category=Beauty             \
    --multimodal.enable=false             \
    --dataset=AmazonReviews2014    \
    --n_hash_buckets=64
conda deactivate