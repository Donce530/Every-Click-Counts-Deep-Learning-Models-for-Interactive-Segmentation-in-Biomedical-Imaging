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


python scripts/evaluate_model.py \
  datasets=['KITS23_2D_TUMOURS_TEST'] \
  model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D-Static disks soft_tissue_window','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D-Dynamic disks soft_tissue_window','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS-Dynamic disks'] \
  exp_name='KITS23_TRANSFER_SMALL' \
  hydra.job.chdir=False\

python scripts/evaluate_model.py \
  datasets=['MD_PANC_2D_TEST'] \
  model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D-Static disks soft_tissue_window','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-MD_PANC_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D-Dynamic disks soft_tissue_window','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-MD_PANC_2D-Dynamic disks'] \
  exp_name='MD_PANC_TRANSFER_SMALL' \
  hydra.job.chdir=False\

python scripts/evaluate_model.py \
  datasets=['LIDC_2D_TEST'] \
  model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-COMBINED_2D-Static disks lidc_window','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-COMBINED_2D-Dynamic disks lidc_window'] \
  exp_name='COMBINED_TO_LIDC_TRANSFER_SMALL' \

