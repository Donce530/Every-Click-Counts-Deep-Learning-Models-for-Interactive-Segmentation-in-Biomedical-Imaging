#!/bin/bash
#SBATCH -J bcv_donatas_masks
#SBATCH --partition=main
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=20000

# TODO 2
# load your virtual environment here
module load any/python/3.8.3-conda
conda activate bcv_clickseg

python precompute_masks.py