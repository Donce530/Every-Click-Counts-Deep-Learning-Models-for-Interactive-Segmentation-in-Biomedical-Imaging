#!/bin/bash
#SBATCH -J bcv_donatas_training
#SBATCH --partition=gpu
#SBATCH -t 10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --gres=gpu:a100-40g

# TODO 2
# load your virtual environment here
module load any/python/3.8.3-conda cuda/11.7.0
conda activate bcv_clickseg

# /gpfs/space/home/donatasv/.conda/envs/bcv_ritm/bin/python3.11 train.py models/iter_mask/hrnet32_lidc_itermask_3c.py --gpus=0 --workers=4 --exp-name=hrnet_32-3-channel-lidc --total-epochs=1000 --batch-size=64
# python train.py models/focalclick/hrnet32_lidc_3c.py --gpus=0 --workers=4 --exp-name=hrnet_32-3-channel-lidc-focal --total-epochs=20 --batch-size=64
