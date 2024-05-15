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

  # datasets=['LIDC_2D_FULL_TEST','KITS23_2D_TUMOURS_FULL_TEST',LITS_2D_FULL_TEST','MD_PANC_2D_FULL_TEST','COMBINED_2D_FULL_TEST'] \
python scripts/evaluate_model.py \
  datasets=['LIDC_2D_FULL_TEST'] \
  n_clicks=1 \
  max_clicks=1 \
  preprocessing.force_windowing=True \
  preprocessing.window_min=-900 \
  preprocessing.window_max=600 \
  model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LITS_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-MD_PANC_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-COMBINED_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LITS_2D_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-MD_PANC_2D_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-COMBINED_2D_FULL-Dynamic disks'] \
  exp_name='TRANSFER_LIDC' \
  hydra.job.chdir=False\

python scripts/evaluate_model.py \
  datasets=['KITS23_2D_TUMOURS_FULL_TEST'] \
  n_clicks=1 \
  max_clicks=1 \
  preprocessing.force_windowing=True \
  preprocessing.window_min=-150 \
  preprocessing.window_max=250 \
  model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LITS_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-MD_PANC_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-COMBINED_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LITS_2D_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-MD_PANC_2D_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-COMBINED_2D_FULL-Dynamic disks'] \
  exp_name='TRANSFER_KITS' \
  hydra.job.chdir=False\

python scripts/evaluate_model.py \
  datasets=['LITS_2D_FULL_TEST'] \
  n_clicks=1 \
  max_clicks=1 \
  preprocessing.force_windowing=True \
  preprocessing.window_min=-45 \
  preprocessing.window_max=105 \
  model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LITS_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-MD_PANC_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-COMBINED_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LITS_2D_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-MD_PANC_2D_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-COMBINED_2D_FULL-Dynamic disks'] \
  exp_name='TRANSFER_LITS' \
  hydra.job.chdir=False\

python scripts/evaluate_model.py \
  datasets=['MD_PANC_2D_FULL_TEST'] \
  n_clicks=1 \
  max_clicks=1 \
  preprocessing.force_windowing=True \
  preprocessing.window_min=-150 \
  preprocessing.window_max=250 \
  model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LITS_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-MD_PANC_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-COMBINED_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LITS_2D_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-MD_PANC_2D_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-COMBINED_2D_FULL-Dynamic disks'] \
  exp_name='TRANSFER_MD_PANC' \
  hydra.job.chdir=False\

python scripts/evaluate_model.py \
  datasets=['COMBINED_2D_FULL_TEST'] \
  n_clicks=1 \
  max_clicks=1 \
  preprocessing.force_windowing=True \
  preprocessing.window_min=-150 \
  preprocessing.window_max=250 \
  model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LITS_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-MD_PANC_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-COMBINED_2D_FULL-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LIDC_2D_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-KITS23_2D_TUMOURS_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-LITS_2D_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-MD_PANC_2D_FULL-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/final_models/RITM-COMBINED_2D_FULL-Dynamic disks'] \
  exp_name='TRANSFER_COMBINED' \
  hydra.job.chdir=False\