# from isegm.utils.exp_imports.default import *
import torch
from easydict import EasyDict as edict

from isegm.utils import common_utils
from isegm.utils.log import logger


def main(cfg):
    device = torch.device('cuda:0')
    model, model_cfg = init_model(cfg, device)
    train(model, cfg, model_cfg, device)


def init_model(cfg, device):
    model_cfg = edict()
    model_cfg.crop_size = (cfg.input_size.height, cfg.input_size.width)
    model_cfg.num_max_points = cfg.max_clicks
    model_cfg.name = f'{cfg.model_type}'
        
    model = common_utils.load_model(cfg, model_cfg, device, train=True)

    return model, model_cfg


def train(model, cfg, model_cfg, device):
    loss_cfg = common_utils.get_loss_configuration(cfg)

    train_set, val_set = common_utils.configure_datasets(cfg, model_cfg)

    trainer = common_utils.get_trainer(cfg, model_cfg, model,loss_cfg, train_set, val_set, device)

    trainer.run(num_epochs=cfg.total_epochs)

