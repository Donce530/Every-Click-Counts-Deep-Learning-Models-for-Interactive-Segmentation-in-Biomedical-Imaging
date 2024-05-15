import torch
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from isegm.simple_click.data.datasets import *
from isegm.simple_click.model.losses import *
from isegm.simple_click.data.transforms import *
from isegm.simple_click.engine.trainer import ISTrainer
from isegm.simple_click.model.metrics import AdaptiveIoU
from isegm.simple_click.data.points_sampler import MultiPointSampler
from isegm.simple_click.utils.log import logger
from isegm.simple_click.model import initializer

from isegm.simple_click.model.is_hrnet_model import HRNetModel
from isegm.simple_click.model.is_deeplab_model import DeeplabModel
from isegm.simple_click.model.is_segformer_model import SegformerModel
from isegm.simple_click.model.is_hrformer_model import HRFormerModel
from isegm.simple_click.model.is_swinformer_model import SwinformerModel
from isegm.simple_click.model.is_plainvit_model import PlainVitModel
