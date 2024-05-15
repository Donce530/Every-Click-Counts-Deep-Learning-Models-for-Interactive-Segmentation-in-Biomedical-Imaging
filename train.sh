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

# python train.py \
#     exp_name='focalclick_target_crop_aug' \
#     model_type='FocalClick' \
#     target_crop_augmentation.enabled=True \
#     total_epochs=700 \
#     lr_scheduling.milestones=[500,600] \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='full_fc_target_crop_aug' \
#     model_type='FocalClick' \
#     dataset.train='LIDC_2D_FULL' \
#     dataset.val='LIDC_2D_FULL_VAL' \
#     iterative_evaluation_interval=-1 \
#     workers=128 \
#     target_crop_augmentation.enabled=True \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='focalclick_standard_aug' \
#     model_type='FocalClick' \
#     target_crop_augmentation.enabled=False \
#     standard_augmentations.enabled=True \
#     batch_size=40 \
#     total_epochs=700 \
#     lr_scheduling.milestones=[500,600] \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='focalclick_no_aug' \
#     model_type='FocalClick' \
#     target_crop_augmentation.enabled=False \
#     standard_augmentations.enabled=False \
#     batch_size=40 \
#     total_epochs=700 \
#     lr_scheduling.milestones=[500,600] \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='focalclick_standard_aug_ritm_crop' \
#     model_type='FocalClick' \
#     target_crop_augmentation.enabled=False \
#     standard_augmentations.enabled=True \
#     input_size.height=320 \
#     input_size.width=480 \
#     total_epochs=700 \
#     lr_scheduling.milestones=[500,600] \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='ritm_standard_aug' \
#     model_type='RITM' \
#     target_crop_augmentation.enabled=False \
#     standard_augmentations.enabled=True \
#     dataset.train='LIDC_2D_FULL' \
#     dataset.val='LIDC_2D_FULL_VAL' \
#     iterative_evaluation_interval=-1 \
#     input_size.height=320 \
#     input_size.width=480 \
#     total_epochs=700 \
#     lr_scheduling.milestones=[500,600] \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='full_ritm_standard_aug' \
#     model_type='RITM' \
#     target_crop_augmentation.enabled=False \
#     standard_augmentations.enabled=True \
#     dataset.train='LIDC_2D_FULL' \
#     dataset.val='LIDC_2D_FULL_VAL' \
#     iterative_evaluation_interval=-1 \
#     workers=128 \
#     input_size.height=320 \
#     input_size.width=480 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='ritm_standard_aug_no_crop' \
#     model_type='RITM' \
#     target_crop_augmentation.enabled=False \
#     standard_augmentations.enabled=True \
#     batch_size=40 \
#     total_epochs=700 \
#     lr_scheduling.milestones=[500,600] \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='ritm__1px' \
#     model_type='RITM' \
#     target_crop_augmentation.enabled=False \
#     standard_augmentations.enabled=True \
#     input_size.height=320 \
#     input_size.width=480 \
#     batch_size=40 \
#     total_epochs=700 \
#     use_prev_mask=False \
#     lr_scheduling.milestones=[500,600] \
#     hydra.job.chdir=False\
#     dist_map_radius=1 \

# python train.py \
#     exp_name='No clicks big dataset' \
#     model_type='RITM' \
#     dataset.train='LIDC_2D_FULL' \
#     dataset.val='LIDC_2D_FULL_VAL' \
#     iterative_trainer=True \
#     clicker.mode='distributed' \
#     dynamic_radius_points=True \
#     augmentation_type='ritm_standard' \
#     input_size.height=320 \
#     input_size.width=480 \
#     total_epochs=700\
#     batch_size=40 \
#     lr_scheduling.milestones=[500,600] \
#     iterative_evaluation_interval=-1 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='Dynamic one click a_0_75 unlocked backbone' \
#     model_type='RITM' \
#     iterative_trainer=True \
#     clicker.mode='distributed' \
#     dynamic_radius_points=True \
#     augmentation_type='ritm_standard' \
#     input_size.height=320 \
#     input_size.width=480 \
#     total_epochs=700\
#     batch_size=40 \
#     lr_scheduling.milestones=[500,600] \
#     iterative_evaluation_interval=-1 \
#     early_stopping_patience=75 \
#     lr_scheduling.patience=25 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='Pre-stage features loss generator unlocked backbone' \
#     model_type='RITM' \
#     backbone_lr_multiplier=1 \
#     iterative_trainer=True \
#     clicker.mode='distributed' \
#     dynamic_radius_points=True \
#     augmentation_type='ritm_standard' \
#     input_size.height=320 \
#     input_size.width=480 \
#     total_epochs=700\
#     batch_size=40 \
#     lr_scheduling.milestones=[500,600] \
#     early_stopping_patience=75 \
#     lr_scheduling.patience=25 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='Dynamic clicking one channel unlocked' \
#     model_type='RITM' \
#     backbone_lr_multiplier=1 \
#     preprocessing.num_input_channels=1 \
#     iterative_trainer=True \
#     clicker.mode='distributed' \
#     use_rgb_conv=False \
#     dynamic_radius_points=True \
#     augmentation_type='ritm_standard' \
#     input_size.height=320 \
#     input_size.width=480 \
#     total_epochs=300\
#     batch_size=40 \
#     early_stopping_patience=75 \
#     lr_scheduling.patience=25 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='UPP one click' \
#     model_type='RITMUPP' \
#     backbone_lr_multiplier=1 \
#     preprocessing.num_input_channels=1 \
#     iterative_trainer=True \
#     use_pretrained_weights=False \
#     clicker.mode='distributed' \
#     use_rgb_conv=False \
#     dynamic_radius_points=True \
#     augmentation_type='ritm_standard' \
#     input_size.height=320 \
#     input_size.width=480 \
#     total_epochs=300\
#     batch_size=24 \
#     early_stopping_patience=75 \
#     lr_scheduling.patience=25 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='dynamic clicks big dataset' \
#     model_type='RITM' \
#     dataset.train='LIDC_2D_FULL' \
#     dataset.val='LIDC_2D_FULL_VAL' \
#     backbone_lr_multiplier=0.1 \
#     preprocessing.num_input_channels=3 \
#     iterative_trainer=True \
#     use_pretrained_weights=True \
#     clicker.mode='distributed' \
#     one_click_only=False \
#     use_rgb_conv=False \
#     dynamic_radius_points=True \
#     augmentation_type='ritm_standard' \
#     input_size.height=320 \
#     input_size.width=480 \
#     total_epochs=300\
#     batch_size=80 \
#     iterative_evaluation_interval=-1 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='Dynamic disks big dataset' \
#     model_type='RITM' \
#     wandb.project_name='RITM Segmentation KITS23' \
#     dataset.train='KITS23_2D_TUMOURS_FULL' \
#     dataset.val='KITS23_2D_TUMOURS_FULL_VAL' \
#     preprocessing/windowing=kits \
#     preprocessing.num_input_channels=3 \
#     iterative_trainer=True \
#     use_pretrained_weights=True \
#     clicker.mode='distributed' \
#     one_click_only=False \
#     use_rgb_conv=False \
#     dynamic_radius_points=True \
#     augmentation_type='ritm_standard' \
#     input_size.height=320 \
#     input_size.width=480 \
#     total_epochs=300 \
#     batch_size=40 \
#     iterative_evaluation_interval=-1 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='Dynamic disks small dataset' \
#     model_type='RITM' \
#     clicker.mode='distributed' \
#     iterative_trainer=True \
#     dist_map_mode='disk' \
#     dataset.train='KITS23_2D_TUMOURS' \
#     dataset.val='KITS23_2D_TUMOURS_VAL' \
#     preprocessing/windowing=kits \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='ritm_standard' \
#     input_size.height=320 \
#     input_size.width=480 \
#     total_epochs=300 \
#     batch_size=40 \
#     iterative_evaluation_interval=10 \
#     early_stopping_patience=75 \
#     lr_scheduling.patience=25 \
#     hydra.job.chdir=False\

#DYNAMIC BIG LIDC

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
#     batch_size=40 \
#     iterative_evaluation_interval=-1 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='Dynamic disks' \
#     model_type='RITM' \
#     clicker.mode='distributed' \
#     iterative_trainer=True \
#     dist_map_mode='disk' \
#     dataset.train='LIDC_2D_FULL' \
#     dataset.val='LIDC_2D_FULL_VAL' \
#     preprocessing/windowing=lidc \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='ritm_standard' \
#     input_size.height=320 \
#     input_size.width=480 \
#     total_epochs=300 \
#     batch_size=40 \
#     iterative_evaluation_interval=-1 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

# # STATIC SMALL LIDC

# python train.py \
#     exp_name='Static disks' \
#     model_type='FocalClick' \
#     clicker.mode='locked' \
#     iterative_trainer=False \
#     dist_map_mode='disk' \
#     dataset.train='LIDC_2D' \
#     dataset.val='LIDC_2D_VAL' \
#     preprocessing/windowing=lidc \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='target_crop' \
#     total_epochs=300 \
#     batch_size=40 \
#     iterative_evaluation_interval=-1 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='Static disks' \
#     model_type='RITM' \
#     clicker.mode='locked' \
#     iterative_trainer=False \
#     dist_map_mode='disk' \
#     dataset.train='LIDC_2D_FULL' \
#     dataset.val='LIDC_2D_FULL_VAL' \
#     preprocessing/windowing=lidc \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='ritm_standard' \
#     input_size.height=320 \
#     input_size.width=480 \
#     total_epochs=300 \
#     batch_size=40 \
#     iterative_evaluation_interval=-1 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

# # STATIC SMALL LIDC
# python train.py \
#     exp_name='Static disks target crop aug' \
#     model_type='FocalClick' \
#     clicker.mode='locked' \
#     iterative_trainer=False \
#     dist_map_mode='disk' \
#     dataset.train='LIDC_2D' \
#     dataset.val='LIDC_2D_VAL' \
#     preprocessing/windowing=lidc \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='target_crop' \
#     total_epochs=300 \
#     batch_size=40 \
#     early_stopping_patience=75 \
#     lr_scheduling.patience=25 \
#     hydra.job.chdir=False\

# # STATIC SMALL KITS
# python train.py \
#     exp_name='Static disks ritm_standard aug' \
#     model_type='FocalClick' \
#     clicker.mode='locked' \
#     iterative_trainer=False \
#     dist_map_mode='disk' \
#     dataset.train='KITS23_2D_TUMOURS' \
#     dataset.val='KITS23_2D_TUMOURS_VAL' \
#     preprocessing/windowing=kits \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='ritm_standard' \
#     total_epochs=300 \
#     batch_size=40 \
#     early_stopping_patience=75 \
#     lr_scheduling.patience=25 \
#     hydra.job.chdir=False\
#
## # STATIC SMALL KITS
#python train.py \
#    exp_name='static ritm_standard aug' \
#    model_type='FocalClick' \
#    clicker.mode='locked' \
#    iterative_trainer=False \
#    dist_map_mode='disk' \
#    dataset.train='KITS23_2D_TUMOURS' \
#    dataset.val='KITS23_2D_TUMOURS_VAL' \
#    preprocessing/windowing=kits \
#    preprocessing.num_input_channels=3 \
#    augmentation_type='ritm_standard' \
#    total_epochs=300 \
#    batch_size=40 \
#    early_stopping_patience=75 \
#    lr_scheduling.patience=25 \
#    hydra.job.chdir=False\
#
#python train.py \
#    exp_name='static focalclick_standard aug' \
#    model_type='FocalClick' \
#    clicker.mode='locked' \
#    iterative_trainer=False \
#    dist_map_mode='disk' \
#    dataset.train='KITS23_2D_TUMOURS' \
#    dataset.val='KITS23_2D_TUMOURS_VAL' \
#    preprocessing/windowing=kits \
#    preprocessing.num_input_channels=3 \
#    augmentation_type='focalclick_standard' \
#    total_epochs=300 \
#    batch_size=40 \
#    early_stopping_patience=75 \
#    lr_scheduling.patience=25 \
#    hydra.job.chdir=False\
#
#python train.py \
#    exp_name='static target_crop aug' \
#    model_type='FocalClick' \
#    clicker.mode='locked' \
#    iterative_trainer=False \
#    dist_map_mode='disk' \
#    dataset.train='KITS23_2D_TUMOURS' \
#    dataset.val='KITS23_2D_TUMOURS_VAL' \
#    preprocessing/windowing=kits \
#    preprocessing.num_input_channels=3 \
#    augmentation_type='target_crop' \
#    total_epochs=300 \
#    batch_size=40 \
#    early_stopping_patience=75 \
#    lr_scheduling.patience=25 \
#    hydra.job.chdir=False\

# # STATIC SMALL LITS
# python train.py \
#     exp_name='static ritm_standard aug' \
#     model_type='FocalClick' \
#     clicker.mode='locked' \
#     iterative_trainer=False \
#     dist_map_mode='disk' \
#     dataset.train='MD_PANC_2D_FULL' \
#     dataset.val='MD_PANC_2D_FULL_VAL' \
#     preprocessing/windowing=md_panc \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='ritm_standard' \
#     total_epochs=300 \
#     batch_size=40 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='static focalclick_standard aug' \
#     model_type='FocalClick' \
#     clicker.mode='locked' \
#     iterative_trainer=False \
#     dist_map_mode='disk' \
#     dataset.train='MD_PANC_2D_FULL' \
#     dataset.val='MD_PANC_2D_FULL_VAL' \
#     preprocessing/windowing=md_panc \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='focalclick_standard' \
#     total_epochs=300 \
#     batch_size=40 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='static target_crop aug' \
#     model_type='FocalClick' \
#     clicker.mode='locked' \
#     iterative_trainer=False \
#     dist_map_mode='disk' \
#     dataset.train='MD_PANC_2D_FULL' \
#     dataset.val='MD_PANC_2D_FULL_VAL' \
#     preprocessing/windowing=md_panc \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='target_crop' \
#     total_epochs=300 \
#     batch_size=40 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

## # STATIC SMALL MD_PANC
#python train.py \
#    exp_name='static ritm_standard aug' \
#    model_type='FocalClick' \
#    clicker.mode='locked' \
#    iterative_trainer=False \
#    dist_map_mode='disk' \
#    dataset.train='MD_PANC_2D' \
#    dataset.val='MD_PANC_2D_VAL' \
#    preprocessing/windowing=md_panc \
#    preprocessing.num_input_channels=3 \
#    augmentation_type='ritm_standard' \
#    total_epochs=300 \
#    batch_size=40 \
#    early_stopping_patience=75 \
#    lr_scheduling.patience=25 \
#    hydra.job.chdir=False\
#
#python train.py \
#    exp_name='static focalclick_standard aug' \
#    model_type='FocalClick' \
#    clicker.mode='locked' \
#    iterative_trainer=False \
#    dist_map_mode='disk' \
#    dataset.train='MD_PANC_2D' \
#    dataset.val='MD_PANC_2D_VAL' \
#    preprocessing/windowing=md_panc \
#    preprocessing.num_input_channels=3 \
#    augmentation_type='focalclick_standard' \
#    total_epochs=300 \
#    batch_size=40 \
#    early_stopping_patience=75 \
#    lr_scheduling.patience=25 \
#    hydra.job.chdir=False\
#
#python train.py \
#    exp_name='static target_crop aug' \
#    model_type='FocalClick' \
#    clicker.mode='locked' \
#    iterative_trainer=False \
#    dist_map_mode='disk' \
#    dataset.train='MD_PANC_2D' \
#    dataset.val='MD_PANC_2D_VAL' \
#    preprocessing/windowing=md_panc \
#    preprocessing.num_input_channels=3 \
#    augmentation_type='target_crop' \
#    total_epochs=300 \
#    batch_size=40 \
#    early_stopping_patience=75 \
#    lr_scheduling.patience=25 \
#    hydra.job.chdir=False\

# # STATIC SMALL COMBINED
# python train.py \
#    exp_name='static ritm_standard aug' \
#    model_type='FocalClick' \
#    clicker.mode='locked' \
#    iterative_trainer=False \
#    dist_map_mode='disk' \
#    dataset.train='COMBINED_2D' \
#    dataset.val='COMBINED_2D_VAL' \
#    preprocessing/windowing=combined \
#    preprocessing.num_input_channels=3 \
#    augmentation_type='ritm_standard' \
#    total_epochs=300 \
#    batch_size=40 \
#    early_stopping_patience=75 \
#    lr_scheduling.patience=25 \
#    hydra.job.chdir=False\



# python train.py \
#    exp_name='static focalclick_standard aug' \
#    model_type='FocalClick' \
#    clicker.mode='locked' \
#    iterative_trainer=False \
#    dist_map_mode='disk' \
#    dataset.train='COMBINED_2D' \
#    dataset.val='COMBINED_2D_VAL' \
#    preprocessing/windowing=combined \
#    preprocessing.num_input_channels=3 \
#    augmentation_type='focalclick_standard' \
#    total_epochs=300 \
#    batch_size=40 \
#    early_stopping_patience=75 \
#    lr_scheduling.patience=25 \
#    hydra.job.chdir=False\

python train.py \
    exp_name='Baseline test' \
    model_type='UnetPlusPlus' \
    clicker.mode='locked' \
    iterative_trainer=False \
    dist_map_mode='gaussian' \
    use_prev_mask=False \
    dataset.train='LIDC_2D' \
    dataset.val='LIDC_2D_VAL' \
    preprocessing/windowing=lidc \
    preprocessing.num_input_channels=3 \
    augmentation_type='ritm_standard' \
    input_size.height=512 \
    input_size.width=512 \
    total_epochs=300 \
    batch_size=40 \
    early_stopping_patience=75 \
    lr_scheduling.patience=25 \
    use_pretrained_weights=False \
    hydra.job.chdir=False\

# python train.py \
#    exp_name='static ritm_standard aug' \
#    model_type='FocalClick' \
#    clicker.mode='locked' \
#    iterative_trainer=False \
#    dist_map_mode='disk' \
#    dataset.train='COMBINED_2D_FULL' \
#    dataset.val='COMBINED_2D_FULL_VAL' \
#    preprocessing/windowing=combined \
#    preprocessing.num_input_channels=3 \
#    augmentation_type='ritm_standard' \
#    total_epochs=300 \
#    batch_size=40 \
#    early_stopping_patience=30 \
#    lr_scheduling.patience=10 \
#    hydra.job.chdir=False \

# python train.py \
#    exp_name='static focalclick_standard aug' \
#    model_type='FocalClick' \
#    clicker.mode='locked' \
#    iterative_trainer=False \
#    dist_map_mode='disk' \
#    dataset.train='COMBINED_2D_FULL' \
#    dataset.val='COMBINED_2D_FULL_VAL' \
#    preprocessing/windowing=combined \
#    preprocessing.num_input_channels=3 \
#    augmentation_type='focalclick_standard' \
#    total_epochs=300 \
#    batch_size=40 \
#    early_stopping_patience=30 \
#    lr_scheduling.patience=10 \
#    hydra.job.chdir=False \

# python train.py \
#    exp_name='static target_crop aug' \
#    model_type='FocalClick' \
#    clicker.mode='locked' \
#    iterative_trainer=False \
#    dist_map_mode='disk' \
#    dataset.train='COMBINED_2D_FULL' \
#    dataset.val='COMBINED_2D_FULL_VAL' \
#    preprocessing/windowing=combined \
#    preprocessing.num_input_channels=3 \
#    augmentation_type='target_crop' \
#    total_epochs=300 \
#    batch_size=40 \
#    early_stopping_patience=30 \
#    lr_scheduling.patience=10 \
#    hydra.job.chdir=False \

## # STATIC SMALL LITS
## python train.py \
##     exp_name='Static disks alternative windowing' \
##     model_type='RITM' \
##     clicker.mode='locked' \
##     iterative_trainer=False \
##     dist_map_mode='disk' \
##     dataset.train='LITS_2D' \
##     dataset.val='LITS_2D_VAL' \
##     preprocessing/windowing=lits \
##     preprocessing.num_input_channels=3 \
##     augmentation_type='ritm_standard' \
##     input_size.height=320 \
##     input_size.width=480 \
##     total_epochs=300 \
##     batch_size=40 \
##     early_stopping_patience=75 \
##     lr_scheduling.patience=25 \
##     hydra.job.chdir=False\



# # STATIC SMALL MD_PANC
# python train.py \
#     exp_name='Static disks' \
#     model_type='RITM' \
#     clicker.mode='locked' \
#     iterative_trainer=False \
#     dist_map_mode='disk' \
#     dataset.train='MD_PANC_2D' \
#     dataset.val='MD_PANC_2D_VAL' \
#     preprocessing/windowing=md_panc \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='ritm_standard' \
#     input_size.height=320 \
#     input_size.width=480 \
#     total_epochs=300 \
#     batch_size=40 \
#     early_stopping_patience=75 \
#     lr_scheduling.patience=25 \
#     hydra.job.chdir=False\
    
# STATIC BIG KITS
# python train.py \
#     exp_name='Static disks' \
#     model_type='FocalClick' \
#     clicker.mode='locked' \
#     iterative_trainer=False \
#     dist_map_mode='disk' \
#     dataset.train='KITS23_2D_TUMOURS_FULL' \
#     dataset.val='KITS23_2D_TUMOURS_FULL_VAL' \
#     preprocessing/windowing=kits \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='target_crop' \
#     total_epochs=300 \
#     batch_size=40 \
#     iterative_evaluation_interval=-1 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='Static disks' \
#     model_type='FocalClick' \
#     clicker.mode='locked' \
#     iterative_trainer=False \
#     dist_map_mode='disk' \
#     dataset.train='LIDC_2D_FULL' \
#     dataset.val='LIDC_2D_FULL_VAL' \
#     preprocessing/windowing=lidc \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='target_crop' \
#     total_epochs=300 \
#     batch_size=40 \
#     iterative_evaluation_interval=-1 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='Static disks' \
#     model_type='RITM' \
#     clicker.mode='locked' \
#     iterative_trainer=False \
#     dist_map_mode='disk' \
#     dataset.train='LIDC_2D_FULL' \
#     dataset.val='LIDC_2D_FULL_VAL' \
#     preprocessing/windowing=lidc \
#     preprocessing.num_input_channels=3 \
#     augmentation_type='ritm_standard' \
#     input_size.height=320 \
#     input_size.width=480 \
#     total_epochs=300 \
#     batch_size=40 \
#     iterative_evaluation_interval=-1 \
#     early_stopping_patience=30 \
#     lr_scheduling.patience=10 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='iterative locked' \
#     model_type='RITM' \
#     backbone_lr_multiplier=0.1 \
#     preprocessing.num_input_channels=3 \
#     iterative_trainer=True \
#     use_pretrained_weights=True \
#     clicker.mode='locked' \
#     overwrite_click_maps=False \
#     one_click_only=False \
#     use_rgb_conv=False \
#     dynamic_radius_points=True \
#     augmentation_type='ritm_standard' \
#     input_size.height=320 \
#     input_size.width=480 \
#     total_epochs=300\
#     batch_size=40 \
#     early_stopping_patience=75 \
#     lr_scheduling.patience=25 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='baseline_one_click_1px' \
#     model_type='UnetPlusPlus' \
#     target_crop_augmentation.enabled=False \
#     standard_augmentations.enabled=True \
#     use_pretrained_weights=False \
#     use_prev_mask=False \
#     max_clicks=1 \
#     max_clicks_before_backprop=0 \
#     batch_size=40 \
#     total_epochs=700 \
#     lr=0.00005 \
#     lr_scheduling.milestones=[500,600] \
#     hydra.job.chdir=False\
#     iterative_evaluation_interval=-1 \
#     dist_map_radius=1 \

# python train.py \
#     exp_name='ritm_no_aug' \
#     model_type='RITM' \
#     target_crop_augmentation.enabled=False \
#     standard_augmentations.enabled=False \
#     batch_size=40 \
#     total_epochs=700 \
#     lr_scheduling.milestones=[500,600] \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='ritm_target_crop_aug' \
#     model_type='RITM' \
#     target_crop_augmentation.enabled=True \
#     standard_augmentations.enabled=False \
#     total_epochs=700 \
#     lr_scheduling.milestones=[500,600] \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='standard_aug_finetune_single_channel' \
#     model_type='SimpleClick_T' \
#     target_crop_augmentation.enabled=False \
#     standard_augmentations.enabled=True \
#     preprocessing.num_input_channels=1 \
#     input_size.channels=1 \
#     batch_size=160 \
#     lr=0.00005 \
#     total_epochs=6000 \
#     lr_scheduling.milestones=[4800,5400] \
#     iterative_evaluation_interval=50 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='standard_aug' \
#     model_type='SimpleClick_T' \
#     target_crop_augmentation.enabled=False \
#     standard_augmentations.enabled=True \
#     total_epochs=6000 \
#     lr_scheduling.milestones=[4800,5400] \
#     iterative_evaluation_interval=50 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='standard_aug_finetune' \
#     model_type='SimpleClick_B' \
#     target_crop_augmentation.enabled=False \
#     standard_augmentations.enabled=True \
#     use_pretrained_weights=True \
#     batch_size=60 \
#     lr=0.00005 \
#     total_epochs=6000 \
#     lr_scheduling.milestones=[4800,5400] \
#     iterative_evaluation_interval=50 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='standard_aug' \
#     model_type='SimpleClick_B' \
#     target_crop_augmentation.enabled=False \
#     standard_augmentations.enabled=True \
#     use_pretrained_weights=True \
#     batch_size=60 \
#     total_epochs=6000 \
#     lr_scheduling.milestones=[4800,5400] \
#     iterative_evaluation_interval=50 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='common_long_run_base_finetune' \
#     model_type='SimpleClick' \
#     standard_augmentations.enabled=True \
#     total_epochs=4000 \
#     batch_size=20\
#     lr=0.00005 \
#     lr_scheduling.milestones=[2800,3200,3600] \
#     iterative_evaluation_interval=50 \
#     hydra.job.chdir=False\

# python train.py \
#     exp_name='model_test' \
#     model_type='RITM' \
#     target_crop_augmentation.enabled=False \
#     standard_augmentations.enabled=True \
#     total_epochs=10 \
#     lr_scheduling.milestones=[5] \
#     hydra.job.chdir=False\
