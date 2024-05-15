# from isegm.simple_click.utils.exp_imports.default import *
import torch
from easydict import EasyDict as edict

from isegm.simple_click.model.is_plainvit_model import PlainVitModel
from isegm.simple_click.data.transforms import UniformRandomResize
from isegm.simple_click.data.points_sampler import MultiPointSampler
from isegm.simple_click.model.losses import NormalizedFocalLossSigmoid
from albumentations import *
from functools import partial
from isegm.simple_click.engine.trainer import ISTrainer
from isegm.simple_click.model.modeling.transformer_helper.cross_entropy_loss import (
    CrossEntropyLoss,
)
from omegaconf import OmegaConf
from isegm.data.preprocess import Preprocessor
from isegm.simple_click.inference.utils import get_dataset


from isegm.model.metrics import AdaptiveIoU, F1Score, IoU

MODEL_NAME = 'plainvit_xtiny512'


def main(cfg):
    device = torch.device('cuda:0')
    model, model_cfg = init_model(cfg, device)
    train(model, cfg, model_cfg, device)


def init_model(cfg, device):
    model_cfg = edict()
    model_cfg.crop_size = (512, 512)
    model_cfg.num_max_points = 24

    windowing_suffix = (
        f'_w_{cfg.preprocessing.windowing.min}_{cfg.preprocessing.windowing.max}'
        if cfg.preprocessing.windowing.enabled
        else ''
    )
    model_cfg.name = f'{MODEL_NAME}{windowing_suffix}'

    if cfg.use_pretrained_weights:
        print(f'xTiny simpleclick has no pretrained weights')

    backbone_params = dict(
        img_size=model_cfg.crop_size,
        patch_size=(16, 16),
        in_chans=3,
        embed_dim=160,
        depth=8,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim=160,
        out_dims=[96, 192, 288, 384],
    )

    head_params = dict(
        in_channels=[96, 192, 288, 384],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=1,
        loss_decode=CrossEntropyLoss(),
        align_corners=False,
        upsample=cfg.upsample,
        channels=128,
    )

    model = PlainVitModel(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=cfg.random_split,
    )

    model.to(device)

    return model, model_cfg


def train(model, cfg, model_cfg, device):
    train_set, val_set = configure_datasets(cfg, model_cfg)

    loss_cfg = configure_loss()

    optimizer_params = {'lr': cfg.lr, 'betas': (0.9, 0.999), 'eps': 1e-8}

    if cfg.lr_scheduling.enabled:
        lr_scheduler = partial(
            torch.optim.lr_scheduler.MultiStepLR,
            milestones=cfg.lr_scheduling.milestones,
            gamma=cfg.lr_scheduling.gamma,
        )
    else:
        lr_scheduler = None

    trainer = ISTrainer(
        model,
        cfg,
        model_cfg,
        loss_cfg,
        train_set,
        val_set,
        device,
        optimizer='adam',
        optimizer_params=optimizer_params,
        layerwise_decay=cfg.layerwise_decay,
        lr_scheduler=lr_scheduler,
        checkpoint_interval=[(0, 5), (200, 10)],
        image_dump_interval=3000,
        metrics=[AdaptiveIoU(), F1Score(), IoU()],
        max_interactive_points=model_cfg.num_max_points,
        max_num_next_clicks=3,
        iterative_evaluation_interval=cfg.iterative_evaluation_interval,
    )
    trainer.run(num_epochs=cfg.total_epochs, validation=True)


def configure_datasets(cfg, model_cfg):
    preprocessor = Preprocessor(OmegaConf.to_container(cfg.preprocessing, resolve=True))

    train_set = get_dataset(cfg.dataset.train, cfg.data_paths, preprocessor)
    train_set.epoch_len = cfg.epoch_length

    val_set = get_dataset(cfg.dataset.val, cfg.data_paths, preprocessor)
    val_epoch_len = (
        int(cfg.epoch_length / 10) if cfg.epoch_length > 0 else -1
    )  # 10% of train if limited, else all
    val_set.epoch_len = val_epoch_len

    if cfg.target_crop_augmentation.enabled:
        print(f'SimpleClick does not have target crops')

    if cfg.standard_augmentations.enabled:
        crop_size = model_cfg.crop_size

        train_augmentator = Compose(
            [
                UniformRandomResize(scale_range=(0.75, 1.40)),
                HorizontalFlip(),
                PadIfNeeded(
                    min_height=crop_size[0], min_width=crop_size[1], border_mode=0
                ),
                RandomCrop(*crop_size),
                # RandomBrightnessContrast(
                #     brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75
                # ),
                # RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
            ],
            p=1.0,
        )
        # val_augmentator = Compose(
        #     [
        #         PadIfNeeded(
        #             min_height=crop_size[0], min_width=crop_size[1], border_mode=0
        #         ),
        #         RandomCrop(*crop_size),
        #     ],
        #     p=1.0,
        # )

        train_set.augmentator = train_augmentator
        # val_set.augmentator = val_augmentator

    points_sampler = MultiPointSampler(
        model_cfg.num_max_points,
        prob_gamma=0.80,
        merge_objects_prob=0.15,
        max_num_merged_objects=2,
    )

    train_set.points_sampler = points_sampler
    val_set.points_sampler = points_sampler

    return train_set, val_set


def configure_loss():
    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0

    return loss_cfg
