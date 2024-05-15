#!/bin/bash
#SBATCH -J iter_eval
#SBATCH -A revvity
#SBATCH --partition=gpu
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=20000
#SBATCH --gres=gpu:tesla

# TODO 2
# load your virtual environment here
module load any/python/3.8.3-conda cuda/11.7.0
conda activate bcv_clickseg

export HYDRA_FULL_ERROR=1

# Focalclick, Focalclick dynamic

# LIDC_2D_VAL,
python scripts/evaluate_model.py --multirun \
  datasets=['LIDC_2D_VAL'] \
  start_progressive_merge=-1,1,10 \
  model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/FocalClick-LIDC_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/FocalClick-LIDC_2D-Dynamic disks'] \
  exp_name='Progressive merge' \
  hydra.job.chdir=False\