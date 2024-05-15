#!/bin/bash
#SBATCH -J iter_eval
#SBATCH -A revvity
#SBATCH --partition=gpu
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=20000
#SBATCH --gres=gpu:tesla

# TODO 2
# load your virtual environment here
module load any/python/3.8.3-conda cuda/11.7.0
conda activate bcv_clickseg

export HYDRA_FULL_ERROR=1

# python scripts/evaluate_model.py \
#   datasets=['LIDC_2D_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-21/01-14-07-RITM_long_run','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-21/00-57-56-FocalClick_long_run','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-20/16-55-39-SimpleClick512_high_lr_long_run'] \
#   exp_name='preliminary_long_runs_simpleclick' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LIDC_2D_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-23/00-36-25-SimpleClick-common_long_run','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-23/00-48-25-SimpleClick-common_long_run_rgb','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-23/00-52-16-SimpleClick-common_long_run_from_scratch'] \
#   exp_name='simpleclick_tiny_pretrained_vs_rgb_vs_fresh_last_checkpoint' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LIDC_2D_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-29/13-39-34-FocalClick-focalclick_target_crop_aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-29/14-03-45-FocalClick-focalclick_standard_aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-29/14-15-37-FocalClick-focalclick_no_aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-29/14-31-37-FocalClick-focalclick_standard_aug_ritm_crop','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-29/14-47-43-RITM-ritm_standard_aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-29/14-57-41-RITM-ritm_standard_aug_no_crop','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-29/19-38-52-RITM-ritm_no_aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-12-03/16-39-19-RITM-ritm_target_crop_aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-12-03/18-03-17-SimpleClick_T-standard_aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-12-03/17-51-16-SimpleClick_T-standard_aug_finetune','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-12-03/18-25-19-SimpleClick_T-standard_aug_from_scratch'] \
#   exp_name='retesting_models_again' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LIDC_2D_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-29/13-39-34-FocalClick-focalclick_target_crop_aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-29/14-47-43-RITM-ritm_standard_aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-02/17-03-39-UnetPlusPlus-baseline_gaussian_standard_aug_no_crop','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-02/16-31-36-UnetPlusPlus-baseline_gaussian_standard_aug_no_crop_no_prev_mask','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-12-30/14-34-28-UnetPlusPlus-baseline_disk_standard_aug_no_crop','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-02/14-05-30-UnetPlusPlus-baseline_disk_standard_aug_no_crop_no_prev_mask'] \
#   exp_name='gaussian_disk_unetpp_vs_ritm_focalclick' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LIDC_2D_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-12-30/14-34-28-UnetPlusPlus-baseline_disk_standard_aug_no_crop','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-02/14-05-30-UnetPlusPlus-baseline_disk_standard_aug_no_crop_no_prev_mask','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-04/14-59-14-UnetPlusPlus-baseline_gaussian_5','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-04/14-57-14-UnetPlusPlus-baseline_gaussian_5_no_prev_mask','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-04/14-51-16-UnetPlusPlus-baseline_gaussian_16','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-04/14-53-13-UnetPlusPlus-baseline_gaussian_16_no_prev_mask','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-04/14-49-14-UnetPlusPlus-baseline_gaussian_32','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-04/14-47-25-UnetPlusPlus-baseline_gaussian_32_no_prev_mask'] \
#   exp_name='unetpp_gaussian_eval' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LIDC_2D_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-12-30/14-34-28-UnetPlusPlus-baseline_disk_standard_aug_no_crop','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-02/14-05-30-UnetPlusPlus-baseline_disk_standard_aug_no_crop_no_prev_mask','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-05/00-34-27-UnetPlusPlus-baseline_distance_5','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-08/20-50-59-UnetPlusPlus-baseline_distance_5_no_prev_mask','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-08/21-28-57-UnetPlusPlus-baseline_distance_16','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-08/20-58-44-UnetPlusPlus-baseline_distance_16_no_prev_mask','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-08/21-40-46-UnetPlusPlus-baseline_distance_32','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-09/02-59-20-UnetPlusPlus-baseline_distance_32_no_prev_mask'] \
#   exp_name='unetpp_distance_eval' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LIDC_2D_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-29/14-47-43-RITM-ritm_standard_aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-10/13-30-48-RITM-ritm_one_click','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-10/13-32-33-RITM-ritm_one_click_random_crop','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-11/12-00-53-RITM-ritm_one_click_random_crop_finetune','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-11/12-12-35-UnetPlusPlus-baseline_one_click'] \
#   exp_name='one_click_models' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LIDC_2D_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2023-11-29/14-47-43-RITM-ritm_standard_aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-01-11/14-29-17-RITM-ritm__1px'] \
#   exp_name='1px_ritm' \
#   hydra.job.chdir=False\










# python scripts/evaluate_model.py \
#   datasets=['KITS23_2D_TUMOURS_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-06/16-12-02-RITM-KITS23_2D_TUMOURS-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-06/15-31-57-RITM-KITS23_2D_TUMOURS-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-18/20-45-30-RITM-KITS23_2D_TUMOURS-Dynamic disks small dataset','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-19/01-31-34-RITM-KITS23_2D_TUMOURS-Static disks small dataset'] \
#   exp_name='ritm_static_vs_dynamic_kits_small_test_old_vs_new' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LIDC_2D_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-02-15/17-20-07-RITM-ritm_standard BceDice loss','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-02-15/16-42-10-RITM-dynamic clicks BceDice loss'] \
#   exp_name='ritm_static_vs_dynamic_lidc_small' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LIDC_2D_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-02-15/16-42-10-RITM-dynamic clicks BceDice loss','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-02-20/17-13-21-RITM-dynamic clicks overwrite maps','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-02-20/20-37-57-RITM-only positive dynamic clicks'] \
#   exp_name='dynamic_vs_overwrite_vs_pos_only' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LIDC_2D_FULL_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-02-20/02-34-57-RITM-ritm_standard big_dataset','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-02-20/12-20-46-RITM-dynamic_clicks big dataset'] \
#   exp_name='ritm_static_vs_dynamic_lidc_big' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LIDC_2D_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-02-15/17-20-07-RITM-ritm_standard BceDice loss','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-24/14-58-53-FocalClick-LIDC_2D-Static disks NormFocalLoss'] \
#   exp_name='static_ritm_vs_focalclick_lidc_small' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LIDC_2D_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-24/14-58-53-FocalClick-LIDC_2D-Static disks NormFocalLoss','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-24/16-11-30-FocalClick-LIDC_2D-Static disks BceDiceLoss','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-24/16-51-49-FocalClick-LIDC_2D-Static disks NormFocalLoss ritm_standard_aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-24/17-05-44-FocalClick-LIDC_2D-Static disks NormFocalLoss focalclick_standard_aug'] \
#   exp_name='static_focalclick_loss_augmentation_tests' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LIDC_2D_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-24/14-58-53-FocalClick-LIDC_2D-Static disks NormFocalLoss','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-25/15-01-09-FocalClick-LIDC_2D-Dynamic disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-25/15-21-13-RITM-LIDC_2D-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-25/15-17-29-RITM-LIDC_2D-Dynamic disks'] \
#   exp_name='ritm_vs_focalclick_lidc_small' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['KITS23_2D_TUMOURS_VAL'] \
#   predictor.ensure_minimum_focus_crop_size=False \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-13/18-44-52-FocalClick-KITS23_2D_TUMOURS-static ritm_standard aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-13/19-41-56-FocalClick-KITS23_2D_TUMOURS-static focalclick_standard aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-13/20-34-54-FocalClick-KITS23_2D_TUMOURS-static target_crop aug'] \
#   exp_name='focalclick_kits_augmentations' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LITS_2D_VAL'] \
#   predictor.ensure_minimum_focus_crop_size=False \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-13/21-16-37-FocalClick-LITS_2D-static ritm_standard aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-13/21-32-28-FocalClick-LITS_2D-static focalclick_standard aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-13/21-58-08-FocalClick-LITS_2D-static target_crop aug'] \
#   exp_name='focalclick_lits_augmentations' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['MD_PANC_2D_VAL'] \
#   predictor.ensure_minimum_focus_crop_size=False \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-13/22-29-02-FocalClick-MD_PANC_2D-static ritm_standard aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-13/23-19-05-FocalClick-MD_PANC_2D-static focalclick_standard aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-13/23-49-45-FocalClick-MD_PANC_2D-static target_crop aug'] \
#   exp_name='focalclick_md_panc_augmentations' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LITS_2D_FULL_VAL'] \
#   predictor.ensure_minimum_focus_crop_size=False \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-14/21-38-38-FocalClick-LITS_2D_FULL-static ritm_standard aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-15/03-45-49-FocalClick-LITS_2D_FULL-static focalclick_standard aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-15/09-57-42-FocalClick-LITS_2D_FULL-static target_crop aug'] \
#   exp_name='focalclick_lits_full_augmentations' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['MD_PANC_2D_FULL'] \
#   predictor.ensure_minimum_focus_crop_size=False \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-15/14-00-04-FocalClick-MD_PANC_2D_FULL-static ritm_standard aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-15/15-46-57-FocalClick-MD_PANC_2D_FULL-static focalclick_standard aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-15/17-45-37-FocalClick-MD_PANC_2D_FULL-static target_crop aug'] \
#   exp_name='focalclick_md_panc_full_augmentations' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['LIDC_2D_VAL'] \
#   predictor.ensure_minimum_focus_crop_size=False \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-15/15-14-51-FocalClick-LIDC_2D-static ritm_standard aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-15/16-17-14-FocalClick-LIDC_2D-static focalclick_standard aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-15/17-24-55-FocalClick-LIDC_2D-static target_crop aug'] \
#   exp_name='focalclick_lidc_augmentations' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['COMBINED_2D_VAL'] \
#   predictor.ensure_minimum_focus_crop_size=False \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-15/16-29-22-FocalClick-COMBINED_2D-static ritm_standard aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-15/17-06-57-FocalClick-COMBINED_2D-static focalclick_standard aug','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-04-16/12-31-41-FocalClick-COMBINED_2D-static target_crop aug'] \
#   exp_name='focalclick_combined_augmentations' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['KITS23_2D_TUMOURS_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-06/16-12-02-RITM-KITS23_2D_TUMOURS-Static disks','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-06/15-31-57-RITM-KITS23_2D_TUMOURS-Dynamic disks'] \
#   exp_name='ritm_static_vs_dynamic_kits_small' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['KITS23_2D_TUMOURS_FULL_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-06/16-38-20-RITM-KITS23_2D_TUMOURS-Static disks big dataset','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-07/10-41-43-RITM-Dynamic disks big dataset'] \
#   exp_name='ritm_static_vs_dynamic_kits_big' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['KITS23_2D_TUMOURS_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-02-15/17-20-07-RITM-ritm_standard BceDice loss','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-02-15/17-20-07-RITM-ritm_standard BceDice loss kits windowing','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-06/16-12-02-RITM-KITS23_2D_TUMOURS-Static disks'] \
#   exp_name='lidc_to_kits_transfer_static' \
#   hydra.job.chdir=False\

# python scripts/evaluate_model.py \
#   datasets=['KITS23_2D_TUMOURS_VAL'] \
#   model_training_paths=['/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-02-15/16-42-10-RITM-dynamic clicks BceDice loss','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-02-15/16-42-10-RITM-dynamic clicks BceDice loss kits windowing','/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training/2024-03-06/15-31-57-RITM-KITS23_2D_TUMOURS-Dynamic disks'] \
#   exp_name='lidc_to_kits_transfer_dynamic' \
#   hydra.job.chdir=False\




