from isegm.utils.exp_imports.default import *
import torch.nn as nn
import torch
from isegm.data.compose import ComposeDataset, ProportionalComposeDataset
from isegm.data.aligned_augmentation import AlignedAugmentator
from isegm.engine.focalclick_trainer import ISTrainer
from isegm.inference.utils import find_checkpoint, load_is_model

MODEL_NAME = 'segformerB3_S2_comb'


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (256, 256)
    model_cfg.num_max_points = 24
    model_cfg.name = MODEL_NAME

    device = torch.device('cuda:0')
    checkpoint_path = find_checkpoint(
        cfg.INTERACTIVE_MODELS_PATH, 'author_pretrained_segformerB3_combined'
    )
    model = load_is_model(checkpoint_path, device)
    for param in model.parameters():
        param.requires_grad = True
    model.to(device)
    model.train()

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 28 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0

    loss_cfg.instance_refine_loss = WFNL(alpha=0.5, gamma=2, w=0.5)
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

    train_set = LidcDataset(cfg.LIDC_PATH, points_sampler=points_sampler)
    val_set = LidcDataset(cfg.LIDC_VAL_PATH, points_sampler=points_sampler)

    optimizer_params = {'lr': 5e-3, 'betas': (0.9, 0.999), 'eps': 1e-8}

    lr_scheduler = partial(
        torch.optim.lr_scheduler.MultiStepLR, milestones=[190, 210], gamma=0.1
    )

    trainer = ISTrainer(
        model,
        cfg,
        model_cfg,
        loss_cfg,
        train_set,
        val_set,
        optimizer="adam",
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        checkpoint_interval=[(0, 50), (200, 5)],
        image_dump_interval=500,
        metrics=[AdaptiveIoU()],
        max_interactive_points=model_cfg.num_max_points,
        max_num_next_clicks=3,
    )

    trainer.run(num_epochs=cfg.total_epochs)
