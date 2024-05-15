#!/bin/bash
#SBATCH -J iter_eval
#SBATCH -A revvity
#SBATCH --partition=gpu
#SBATCH -t 96:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=20000
#SBATCH --gres=gpu:tesla

# TODO 2
# load your virtual environment here
module load any/python/3.8.3-conda cuda/11.7.0
conda activate bcv_clickseg

export HYDRA_FULL_ERROR=1

# UNET++, RITM, Focalclick, Ritm dynamic, focalclick dynamic

# LIDC_2D_VAL, LIDC_2D_TEST
python scripts/evaluate_model.py \
  datasets=['LIDC_2D_VAL','LIDC_2D_TEST'] \
  model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/UnetPlusPlus-LIDC_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/FocalClick-LIDC_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/FocalClick-LIDC_2D-Dynamic disks'] \
  exp_name='LIDC_SMALL' \
  hydra.job.chdir=False\

# # KITS23_2D_TUMOURS_VAL, KITS23_2D_TUMOURS_TEST
python scripts/evaluate_model.py \
  datasets=['KITS23_2D_TUMOURS_VAL','KITS23_2D_TUMOURS_TEST'] \
  model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/UnetPlusPlus-KITS23_2D_TUMOURS-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/FocalClick-KITS23_2D_TUMOURS-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/FocalClick-KITS23_2D_TUMOURS-Dynamic disks'] \
  exp_name='KITS23_SMALL' \
  hydra.job.chdir=False\

python scripts/evaluate_model.py \
  datasets=['KITS23_2D_TUMOURS_VAL','KITS23_2D_TUMOURS_TEST'] \
  predictor.ensure_minimum_focus_crop_size= True \
  model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/UnetPlusPlus-KITS23_2D_TUMOURS-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/FocalClick-KITS23_2D_TUMOURS-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/FocalClick-KITS23_2D_TUMOURS-Dynamic disks'] \
  exp_name='KITS23_SMALL_FORCED_ZOOM' \
  hydra.job.chdir=False\

# # LITS_2D_VAL, LITS_2D_TEST
python scripts/evaluate_model.py \
  datasets=['LITS_2D_VAL','LITS_2D_TEST'] \
  model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/UnetPlusPlus-LITS_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LITS_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/FocalClick-LITS_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LITS_2D-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/FocalClick-LITS_2D-Dynamic disks'] \
  exp_name='LITS_SMALL' \
  hydra.job.chdir=False\

# # MD_PANC_2D_VAL, MD_PANC_2D_TEST
python scripts/evaluate_model.py \
  datasets=['MD_PANC_2D_VAL','MD_PANC_2D_TEST'] \
  model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/UnetPlusPlus-MD_PANC_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-MD_PANC_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/FocalClick-MD_PANC_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-MD_PANC_2D-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/FocalClick-MD_PANC_2D-Dynamic disks'] \
  exp_name='MD_PANC_SMALL' \
  hydra.job.chdir=False\

# # COMBINED_2D_VAL, COMBINED_2D_TEST
python scripts/evaluate_model.py \
  datasets=['COMBINED_2D_VAL','COMBINED_2D_TEST'] \
  model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/UnetPlusPlus-COMBINED_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-COMBINED_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/FocalClick-COMBINED_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-COMBINED_2D-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/FocalClick-COMBINED_2D-Dynamic disks'] \
  exp_name='COMBINED_SMALL' \
  hydra.job.chdir=False\
