#!/bin/bash
#SBATCH -J iter_seg_train
#SBATCH -A revvity
#SBATCH --partition=main
#SBATCH -t 96:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64000

# TODO 2
# load your virtual environment here
module load any/python/3.8.3-conda
conda activate bcv_clickseg

python classical_methods_param_search_flood.py