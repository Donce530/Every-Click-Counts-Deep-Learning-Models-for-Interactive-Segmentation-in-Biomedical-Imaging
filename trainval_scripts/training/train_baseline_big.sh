#!/bin/bash
#SBATCH -J iter_seg_train
#SBATCH -A revvity
#SBATCH --partition=gpu
#SBATCH -t 96:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH --gres=gpu:a100-80g
#SBATCH --exclude=falcon1,falcon2,falcon3

# TODO 2
# load your virtual environment here
module load any/python/3.8.3-conda cuda/11.7.0
conda activate bcv_clickseg

# STATIC BIG LIDC
python train.py \
    exp_name='Static disks' \
    model_type='UnetPlusPlus'  \
    clicker.mode='locked' \
    iterative_trainer=False \
    dist_map_mode='gaussian' \
    use_prev_mask=False \
    dataset.train='LIDC_2D_FULL' \
    dataset.val='LIDC_2D_FULL_VAL' \
    preprocessing/windowing=lidc \
    preprocessing.num_input_channels=3 \
    augmentation_type='ritm_standard' \
    input_size.height=512 \
    input_size.width=512 \
    total_epochs=300 \
    batch_size=40 \
    iterative_evaluation_interval=-1 \
    early_stopping_patience=30 \
    lr_scheduling.patience=10 \
    use_pretrained_weights=False \
    hydra.job.chdir=False\

# STATIC BIG KITS
python train.py \
    exp_name='Static disks' \
    model_type='UnetPlusPlus'  \
    clicker.mode='locked' \
    iterative_trainer=False \
    dist_map_mode='gaussian' \
    use_prev_mask=False \
    dataset.train='KITS23_2D_TUMOURS_FULL' \
    dataset.val='KITS23_2D_TUMOURS_FULL_VAL' \
    preprocessing/windowing=kits \
    preprocessing.num_input_channels=3 \
    augmentation_type='ritm_standard' \
    input_size.height=512 \
    input_size.width=512 \
    total_epochs=300 \
    batch_size=40 \
    iterative_evaluation_interval=-1 \
    early_stopping_patience=30 \
    lr_scheduling.patience=10 \
    use_pretrained_weights=False \
    hydra.job.chdir=False\

# STATIC BIG LITS
python train.py \
    exp_name='Static disks' \
    model_type='UnetPlusPlus'  \
    clicker.mode='locked' \
    iterative_trainer=False \
    dist_map_mode='gaussian' \
    use_prev_mask=False \
    dataset.train='LITS_2D_FULL' \
    dataset.val='LITS_2D_FULL_VAL' \
    preprocessing/windowing=lits \
    preprocessing.num_input_channels=3 \
    augmentation_type='ritm_standard' \
    input_size.height=512 \
    input_size.width=512 \
    total_epochs=300 \
    batch_size=40 \
    iterative_evaluation_interval=-1 \
    early_stopping_patience=30 \
    lr_scheduling.patience=10 \
    use_pretrained_weights=False \
    hydra.job.chdir=False\

# STATIC BIG MD_PANC
python train.py \
    exp_name='Static disks' \
    model_type='UnetPlusPlus'  \
    clicker.mode='locked' \
    iterative_trainer=False \
    dist_map_mode='gaussian' \
    use_prev_mask=False \
    dataset.train='MD_PANC_2D_FULL' \
    dataset.val='MD_PANC_2D_FULL_VAL' \
    preprocessing/windowing=md_panc \
    preprocessing.num_input_channels=3 \
    augmentation_type='ritm_standard' \
    input_size.height=512 \
    input_size.width=512 \
    total_epochs=300 \
    batch_size=40 \
    iterative_evaluation_interval=-1 \
    early_stopping_patience=30 \
    lr_scheduling.patience=10 \
    use_pretrained_weights=False \
    hydra.job.chdir=False\

# TRAIN STATIC BIG COMBINED
python train.py \
    exp_name='Static disks' \
    model_type='UnetPlusPlus'  \
    clicker.mode='locked' \
    iterative_trainer=False \
    dist_map_mode='gaussian' \
    use_prev_mask=False \
    dataset.train='COMBINED_2D_FULL' \
    dataset.val='COMBINED_2D_FULL_VAL' \
    preprocessing/windowing=combined \
    preprocessing.num_input_channels=3 \
    augmentation_type='ritm_standard' \
    input_size.height=512 \
    input_size.width=512 \
    total_epochs=300 \
    batch_size=40 \
    iterative_evaluation_interval=-1 \
    early_stopping_patience=30 \
    lr_scheduling.patience=10 \
    use_pretrained_weights=False \
    hydra.job.chdir=False\