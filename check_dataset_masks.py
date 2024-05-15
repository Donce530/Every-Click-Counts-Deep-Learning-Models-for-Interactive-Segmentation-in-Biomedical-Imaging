import os
import random
import logging
from copy import deepcopy
from collections import defaultdict
from matplotlib import pyplot as plt

import cv2
import torch
import numpy as np
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
# from isegm.engine.focalclick_trainer import ISTrainer

from isegm.data.compose import ComposeDataset, ProportionalComposeDataset
import torch.nn as nn
from isegm.data.aligned_augmentation import AlignedAugmentator
# from isegm.engine.focalclick_trainer import ISTrainer
from isegm.data.preprocess import Preprocessor
from isegm.utils import exp
from isegm.inference import utils

cfg = exp.load_config_file('config.yml', return_edict=True)


from tqdm import tqdm
dataset_names = [
    'LIDC_2D','LIDC_2D_VAL','LIDC_2D_TEST','LIDC_2D_FULL','LIDC_2D_FULL_VAL','LIDC_2D_FULL_TEST',
    'KITS23_2D_TUMOURS','KITS23_2D_TUMOURS_VAL','KITS23_2D_TUMOURS_TEST','KITS23_2D_TUMOURS_FULL','KITS23_2D_TUMOURS_FULL_VAL','KITS23_2D_TUMOURS_FULL_TEST',
    'LITS_2D','LITS_2D_VAL','LITS_2D_TEST','LITS_2D_FULL','LITS_2D_FULL_VAL','LITS_2D_FULL_TEST',
    'MD_PANC_2D','MD_PANC_2D_VAL','MD_PANC_2D_TEST','MD_PANC_2D_FULL','MD_PANC_2D_FULL_VAL','MD_PANC_2D_FULL_TEST',
    'COMBINED_2D','COMBINED_2D_VAL','COMBINED_2D_TEST','COMBINED_2D_FULL','COMBINED_2D_FULL_VAL','COMBINED_2D_FULL_TEST',
]

empty_mask_sample_paths = []

for d_name in dataset_names:
    dataset = utils.get_dataset(d_name, cfg)
    print(f'{d_name} loaded, has {len(dataset)} samples')
    for i in tqdm(range(len(dataset))):
        sample = dataset.get_sample(i)
        if sample.gt_mask.sum() == 0:
            print(f'{d_name} sample {i} has no mask')
            paths = dataset.get_sample_paths(i)
            print(f'Sample path = {paths}')
            empty_mask_sample_paths.append(paths)
            
import pickle
with open('empty_mask_sample_paths.pkl', 'wb') as f:
    pickle.dump(empty_mask_sample_paths, f)