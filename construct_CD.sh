#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=80:00:00
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=zl4789@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate TIGER
cd /scratch/zl4789/action_piece_google
python build_vocab.py   \
    --category=CDs_and_Vinyl             \
    --multimodal.enable=true             \
    --multimodal.image_pca_dim=128       \
    --multimodal.final_pca_dim=128       \
    --dataset=AmazonReviews2018

conda deactivate