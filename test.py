import matplotlib.pyplot as plt

import sys
import numpy as np
import torch
import cv2

sys.path.insert(0, '..')
from isegm.utils import vis, exp

from isegm.inference import utils
from isegm.inference.predictors import get_predictor
from isegm.simple_click.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss
from isegm.simple_click.model.modeling.pos_embed import interpolate_pos_embed_inference

device = torch.device('cuda:0')
cfg = exp.load_config_file('config.yml', return_edict=True)

EVAL_MAX_CLICKS = 20
MODEL_THRESH = 0.49

TRAINING_OUTPUT_PATH = '/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/outputs/training'
PRETRAINED_WEIGTHS_PATH = '/gpfs/space/projects/PerkinElmer/donatasv_experiments/repos/ClickSEG/weights'

checkpoint_path = f'{PRETRAINED_WEIGTHS_PATH}/sbd_vit_xtiny.pth'

data = torch.load(checkpoint_path)