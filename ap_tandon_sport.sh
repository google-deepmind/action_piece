#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --account=pr_119_tandon_priority
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=zl4789@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate TIGER
cd /scratch/zl4789/action_piece_google

CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Sports_and_Outdoors \
    --weight_decay=0.18 \
    --lr=0.001 \
    --n_hash_buckets=64

conda deactivate