import os
import random
import logging
from copy import deepcopy
from collections import defaultdict

import cv2
import torch
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points, add_tag
from isegm.utils.misc import save_checkpoint
from isegm.utils.serialization import get_config_repr
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from torch.cuda.amp import autocast as autocast, GradScaler

from isegm.utils.exp_imports.default import *

MODEL_NAME = "lidc_hrnet32"

from isegm.data.compose import ComposeDataset, ProportionalComposeDataset
import torch.nn as nn
from isegm.data.aligned_augmentation import AlignedAugmentator
from isegm.engine.focalclick_trainer import ISTrainer

from isegm.data.compose import ComposeDataset, ProportionalComposeDataset
import torch.nn as nn
from isegm.data.aligned_augmentation import AlignedAugmentator
from isegm.engine.focalclick_trainer import ISTrainer

LIDC_PATH = "/gpfs/space/projects/PerkinElmer/donatasv_experiments/datasets/processed_datasets/LIDC-2D/train"
LIDC_256_PATH = "/gpfs/space/projects/PerkinElmer/donatasv_experiments/datasets/processed_datasets/LIDC-2D-256/train"

crop_size = (256, 256)

points_sampler = MultiPointSampler(
        24,
        prob_gamma=0.80,
        merge_objects_prob=0.15,
        max_num_merged_objects=2,
        use_hierarchy=False,
        first_click_center=True,
    )
    
# train_augmentator = Compose(
#     [
#         # UniformRandomResize(scale_range=(0.75, 1.40)),
#         # HorizontalFlip(),
#         # PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
#         # RandomCrop(*crop_size),
#         # RandomBrightnessContrast(
#         #     brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75
#         # ),
#         # RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
#     ],
#     p=1.0,
# )
# val_augmentator = Compose(
#     [
#         # PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
#         # RandomCrop(*crop_size),
#     ],
#     p=1.0,
# )

a_set = LidcDataset(
    LIDC_256_PATH, points_sampler=points_sampler, #augmentator=train_augmentator
)
b_set = LidcDataset(
    LIDC_PATH, points_sampler=points_sampler, #augmentator=val_augmentator
)

distributed = 'WORLD_SIZE' in os.environ
print(f'distributed: {distributed}')

print('created datasets')

loader = DataLoader(
            a_set,
            64,
            sampler=get_sampler(a_set, shuffle=True, distributed=False),
            drop_last=True,
            pin_memory=True,
            num_workers=16,
        )

print('created_loader')

start = time.time()

for batch in loader:
    print(f'{time.time() - start}')
    print(len(batch))

end = time.time()

print(f'time: {end - start}')