#!/bin/bash
#SBATCH -J iter_seg_train
#SBATCH -A revvity
#SBATCH --partition=gpu
#SBATCH -t 190:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH --gres=gpu:a100-80g
#SBATCH --exclude=falcon1,falcon2,falcon3

# TODO 2
# load your virtual environment here
module load any/python/3.8.3-conda cuda/11.7.0
conda activate bcv_clickseg

# Dynamic big LIDC
# python train.py \
#     exp_name='Dynamic disks' \
#     model_type='FocalClick' \
#     clicker.mode='distributed' \
#     iterative_trainer=True \
#     dist_map_mode='disk' \
#     dataset.train='LIDC_2D_FULL' \
#     dataset.val='LIDC_2D_FULL_VAL' \
#     preprocessing/windowing=lidc \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='target_crop' \
#     total_epochs=300 \
#     batch_size=80 \
#     iterative_evaluation_interval=-1 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

# Dynamic SMALL KITS
# python train.py \
#     exp_name='Dynamic disks' \
#     model_type='FocalClick' \
#     clicker.mode='distributed' \
#     iterative_trainer=True \
#     dist_map_mode='disk' \
#     dataset.train='KITS23_2D_TUMOURS_FULL' \
#     dataset.val='KITS23_2D_TUMOURS_FULL_VAL' \
#     preprocessing/windowing=kits \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='ritm_standard' \
#     total_epochs=300 \
#     batch_size=80 \
#     iterative_evaluation_interval=-1 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

# # Dynamic SMALL LITS
# python train.py \
#     exp_name='Dynamic disks' \
#     model_type='FocalClick' \
#     clicker.mode='distributed' \
#     iterative_trainer=True \
#     dist_map_mode='disk' \
#     dataset.train='LITS_2D_FULL' \
#     dataset.val='LITS_2D_FULL_VAL' \
#     preprocessing/windowing=lits \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='ritm_standard' \
#     total_epochs=300 \
#     batch_size=80 \
#     iterative_evaluation_interval=-1 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

# # Dynamic SMALL MD_PANC
# python train.py \
#     exp_name='Dynamic disks' \
#     model_type='FocalClick' \
#     clicker.mode='distributed' \
#     iterative_trainer=True \
#     dist_map_mode='disk' \
#     dataset.train='MD_PANC_2D_FULL' \
#     dataset.val='MD_PANC_2D_FULL_VAL' \
#     preprocessing/windowing=md_panc \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='ritm_standard' \
#     total_epochs=300 \
#     batch_size=80 \
#     iterative_evaluation_interval=-1 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

# TRAIN Dynamic SMALL COMBINED
python train.py \
    exp_name='Dynamic disks' \
    model_type='FocalClick' \
    clicker.mode='distributed' \
    iterative_trainer=True \
    dist_map_mode='disk' \
    dataset.train='COMBINED_2D_FULL' \
    dataset.val='COMBINED_2D_FULL_VAL' \
    preprocessing/windowing=combined \
    preprocessing.num_input_channels=3 \
    augmentation_type='target_crop' \
    total_epochs=300 \
    batch_size=80 \
    iterative_evaluation_interval=-1 \
    early_stopping_patience=30 \
    lr_scheduling.patience=10 \
    hydra.job.chdir=False\