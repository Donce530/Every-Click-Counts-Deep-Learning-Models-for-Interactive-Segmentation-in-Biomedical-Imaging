from isegm.utils.exp_imports.default import *
import torch.nn as nn
import torch
import numpy as np
from isegm.data.preprocess import Preprocessor
from isegm.engine.focalclick_trainer import ISTrainer
from isegm.inference.utils import find_checkpoint, load_is_model
from omegaconf import OmegaConf


MODEL_NAME = "lidc_hrnet32"


def main(cfg):
    device = torch.device('cuda:0')
    model, model_cfg = init_model(cfg, device)
    train(model, cfg, model_cfg, device)


def init_model(cfg, device):
    model_cfg = edict()
    model_cfg.num_max_points = 24

    windowing_suffix = (
        f'_w_{cfg.preprocessing.windowing.min}_{cfg.preprocessing.windowing.max}'
        if cfg.preprocessing.windowing.enabled
        else ''
    )
    # model_cfg.name = f'{MODEL_NAME}_c_{cfg.preprocessing.enhancements.contrast}_b_{cfg.preprocessing.enhancements.brightness}_s_{cfg.preprocessing.enhancements.sharpness}_g_{cfg.preprocessing.enhancements.gaussian_blur}{windowing_suffix}'
    # model_cfg.name = f'{MODEL_NAME}{windowing_suffix}'

    model_cfg.name = f'{MODEL_NAME}'

    if cfg.use_pretrained_weights:
        print(f'Using pretrained_weights')
        checkpoint_path = find_checkpoint(
            cfg.data_paths.INTERACTIVE_MODELS_PATH, 'author_pretrained_combined'
        )
        model = load_is_model(checkpoint_path, device)
        for param in model.parameters():
            param.requires_grad = True
        model = model.to(device)
        model.train()
    else:
        print('Training from imagenet')
        model = HRNetModel(
            pipeline_version='s2',
            width=32,
            ocr_width=128,
            small=False,
            with_aux_output=True,
            use_leaky_relu=True,
            use_rgb_conv=False,
            use_disks=True,
            norm_radius=5,
            with_prev_mask=True,
        )

        model.to(device)
        model.apply(initializer.XavierGluon(rnd_type="gaussian", magnitude=2.0))
        model.feature_extractor.load_pretrained_weights(
            cfg.data_paths.IMAGENET_PRETRAINED_MODELS.HRNETV2_W32
        )

    return model, model_cfg


def train(model, cfg, model_cfg, device):
    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    loss_cfg.instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.instance_aux_loss_weight = 0.4
    loss_cfg.instance_refine_loss = WFNL(alpha=0.5, gamma=2)
    loss_cfg.instance_refine_loss_weight = 1.0
    loss_cfg.trimap_loss = nn.BCEWithLogitsLoss()
    loss_cfg.trimap_loss_weight = 1.0

    points_sampler = MultiPointSampler(
        model_cfg.num_max_points,
        prob_gamma=0.80,
        merge_objects_prob=0.15,
        max_num_merged_objects=2,
        use_hierarchy=False,
        first_click_center=True,
    )

    preprocessor = Preprocessor(OmegaConf.to_container(cfg.preprocessing, resolve=True))
    print(
        f'initialized preprocessor with {preprocessor.enhancements}, {preprocessor.windowing}, normalize {preprocessor.normalize}'
    )

    train_set = LidcDataset(
        cfg.data_paths.LIDC_PATH,
        points_sampler=points_sampler,
        preprocessor=preprocessor,
    )
    val_set = LidcDataset(
        cfg.data_paths.LIDC_VAL_PATH,
        points_sampler=points_sampler,
        preprocessor=preprocessor,
    )

    # optimizer_params = {"lr": 5e-4, "betas": (0.9, 0.999), "eps": 1e-8}
    optimizer_params = {"lr": 5e-6, "betas": (0.9, 0.999), "eps": 1e-8}

    # lr_scheduler = partial(
    #     torch.optim.lr_scheduler.MultiStepLR, milestones=[190, 210], gamma=0.1
    # )
    # lr_scheduler = partial(
    #     torch.optim.lr_scheduler.MultiStepLR, milestones=[25, 150], gamma=0.1
    # )
    lr_scheduler = None

    trainer = ISTrainer(
        model,
        cfg,
        model_cfg,
        loss_cfg,
        train_set,
        val_set,
        device,
        optimizer="adam",
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        checkpoint_interval=[(0, 5), (200, 10)],
        # checkpoint_interval=None,
        image_dump_interval=500,
        metrics=[AdaptiveIoU(), F1Score(), IoU()],
        max_interactive_points=model_cfg.num_max_points,
        max_num_next_clicks=3,
    )

    trainer.run(num_epochs=cfg.total_epochs)
