from isegm.inference import utils
from easydict import EasyDict as edict
from omegaconf import OmegaConf
import torch.nn as nn
import torch
from functools import partial

from isegm.utils.log import logger

from isegm.model.metrics import AdaptiveIoU, F1Score, IoU


from isegm.model import initializer as FocalClickInitializer
from isegm.inference.utils import find_checkpoint as find_focalclick_checkpoint
from isegm.model.is_hrnet_model import HRNetModel as FocalClickModel

from isegm.ritm.model import initializer as RITMInitializer
from isegm.ritm.inference.utils import find_checkpoint as find_ritm_checkpoint
from isegm.ritm.model.is_hrnet_model import HRNetModel as RITMModel
from isegm.ritm.model.is_unet_plus_plus_model import UNetPlusPlusModel as RITMUPPModel

from isegm.simple_click.model.modeling.transformer_helper.cross_entropy_loss import (
    CrossEntropyLoss,
)
from isegm.simple_click.inference.utils import find_checkpoint as find_simpleclick_checkpoint
from isegm.simple_click.model.is_plainvit_model import PlainVitModel as SimpleClickModel
from isegm.simple_click.model.modeling.pos_embed import interpolate_pos_embed_inference

from isegm.model.is_unetplusplus_model import UnetPlusPlusModel


from isegm.data.preprocess import Preprocessor
from isegm.data.points_sampler import MultiPointSampler
from isegm.model.losses import ClickDiceLoss, NormalizedFocalLossSigmoid, SigmoidBinaryCrossEntropyLoss, WFNL, FocalLoss, BinaryCrossEntropyLossSigmoid, DiceLossSigmoid, BceDiceLoss

from isegm.data.augmentations import AugmentationsProvider

from isegm.engine.common_trainer.focalclick_trainer import FocalClickTrainer
from isegm.engine.common_trainer.ritm_trainer import RITMTrainer
from isegm.engine.common_trainer.iterative_ritm_trainer import IterativeRITMTrainer
from isegm.engine.common_trainer.iterative_focalclick_trainer import IterativeFocalClickTrainer
from isegm.engine.common_trainer.simpleclick_trainer import SimpleClickTrainer
from isegm.engine.common_trainer.unetplusplus_trainer import UnetPlusPlusTrainer

def load_model(cfg, model_cfg, device, train=False):
    if cfg.model_type == 'FocalClick':
        model = initialize_focalclick(cfg, device)
    elif cfg.model_type == 'RITM':
        model = initialize_ritm(cfg, device)
    elif cfg.model_type == 'RITMUPP':
        model = initialize_ritmupp(cfg, device)
    elif cfg.model_type.startswith('SimpleClick'):
        model = initialize_simpleclick(cfg, model_cfg, device)
    elif cfg.model_type == 'UnetPlusPlus':
        model = initialize_unetplusplus(cfg, device)
    else:
        raise Exception(f'Model type {type} unknown')
    
    if train:
        for param in model.parameters():
            param.requires_grad = True
        model.train()
        
    logger.info(f'Loaded {cfg.model_type} model (class: {model.__class__.__name__}), use_pretrained_weights: {cfg.use_pretrained_weights}')   
    
    return model

def initialize_focalclick(cfg, device):
    if cfg.use_pretrained_weights:
        checkpoint_path = find_focalclick_checkpoint(
                cfg.data_paths.INTERACTIVE_MODELS_PATH, 'author_pretrained_combined'
            )
        
        overrides = {
            'use_rgb_conv': cfg.use_rgb_conv,
            'dist_map_mode': cfg.dist_map_mode,
            'overwrite_click_maps': cfg.overwrite_click_maps
        }
        
        model = utils.load_is_model(checkpoint_path, device, model_type=cfg.model_type, training=True, overrides=overrides)
        
        logger.info(f'Loading pretrained weights from: {checkpoint_path}')
    else:
        model = FocalClickModel(
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
            
        model.apply(FocalClickInitializer.XavierGluon(rnd_type="gaussian", magnitude=2.0))
        model.feature_extractor.load_pretrained_weights(
                cfg.data_paths.IMAGENET_PRETRAINED_MODELS.HRNETV2_W32
            )
    return model

def initialize_ritm(cfg, device):
    if cfg.use_pretrained_weights:
        checkpoint_path = find_ritm_checkpoint(
            cfg.data_paths.INTERACTIVE_MODELS_PATH, cfg.pretrained_models.ritm
        )
        # checkpoint_path = find_ritm_checkpoint(
        #     cfg.data_paths.INTERACTIVE_MODELS_PATH, 'lidc_pretrained'
        # )
        
        logger.info(f'Loading pretrained weights from: {checkpoint_path}')
        
        overrides = {
            'use_rgb_conv': cfg.use_rgb_conv,
            'dist_map_mode': cfg.dist_map_mode,
            'overwrite_click_maps': cfg.overwrite_click_maps
        }
        model = utils.load_is_model(checkpoint_path, device, model_type=cfg.model_type, training=True, overrides=overrides)
        
        model.set_backbone_lr_multiplier(cfg.backbone_lr_multiplier)
    else:
        model = RITMModel(
            width=32,
            ocr_width=128,
            with_aux_output=True,
            use_leaky_relu=True,
            use_rgb_conv=False,
            norm_radius=5,
            with_prev_mask=True,
            dist_map_mode=cfg.dist_map_mode,
        )
        model.apply(RITMInitializer.XavierGluon(rnd_type="gaussian", magnitude=2.0))
        model.feature_extractor.load_pretrained_weights(
            cfg.data_paths.IMAGENET_PRETRAINED_MODELS.HRNETV2_W32
        )
        
        model.set_backbone_lr_multiplier(cfg.backbone_lr_multiplier)
            
    return model

def initialize_ritmupp(cfg, device):
    if cfg.use_pretrained_weights:
        # checkpoint_path = find_ritm_checkpoint(
        #     cfg.data_paths.INTERACTIVE_MODELS_PATH, cfg.pretrained_models.ritm
        # )
        # checkpoint_path = find_ritm_checkpoint(
        #     cfg.data_paths.INTERACTIVE_MODELS_PATH, 'lidc_pretrained'
        # )
        
        # logger.info(f'Loading pretrained weights from: {checkpoint_path}')
        
        # model = utils.load_is_model(checkpoint_path, device, model_type=cfg.model_type)
        # model.dynamic_radius_points=cfg.dynamic_radius_points
        
        # overrides = {}
        # if cfg.use_rgb_conv:
        #     overrides['use_rgb_conv'] = True
        # model = utils.load_is_model(checkpoint_path, device, model_type=cfg.model_type, training=True, overrides=overrides)
        
        # model.set_backbone_lr_multiplier(cfg.backbone_lr_multiplier)
        raise Exception(f'No pretrained weights for RITM-UPP available')
    else:
        model = RITMUPPModel()
        # model.dynamic_radius_points=cfg.dynamic_radius_points
        model.set_backbone_lr_multiplier(cfg.backbone_lr_multiplier)
            
    return model

def initialize_simpleclick(cfg, model_cfg, device):
    if cfg.use_pretrained_weights:
        model_size = cfg.model_type.split('_')[1]
        
        if model_size == 'T':
            checkpoint_path = find_simpleclick_checkpoint(
                cfg.data_paths.INTERACTIVE_MODELS_PATH, 'sbd_vit_xtiny'
            )
        elif model_size == 'B':
            checkpoint_path = find_simpleclick_checkpoint(
                cfg.data_paths.INTERACTIVE_MODELS_PATH, 'cocolvis_vit_base'
            )
            
        logger.info(f'Loading pretrained weights from: {checkpoint_path}')
        
        overrides = {}
        if cfg.input_size.channels != 3:
            overrides['backbone_params'] = {'in_chans': cfg.input_size.channels}
        model = utils.load_is_model(checkpoint_path, device, model_type=cfg.model_type, training=True, overrides=overrides)
    else:
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
        
        model = SimpleClickModel(
            use_disks=True,
            norm_radius=5,
            with_prev_mask=True,
            backbone_params=backbone_params,
            neck_params=neck_params,
            head_params=head_params,
            random_split=cfg.random_split,
        )
        
    return model

def initialize_unetplusplus(cfg, device):
    if cfg.use_pretrained_weights:
        pass
        # checkpoint_path = find_ritm_checkpoint(
        #     cfg.data_paths.INTERACTIVE_MODELS_PATH, 'author_pretrained_ritm_32'
        # )
        
        # logger.info(f'Loading pretrained weights from: {checkpoint_path}')
        
        # model = utils.load_is_model(checkpoint_path, device, model_type=cfg.model_type)
    else:
        model = UnetPlusPlusModel(
            with_prev_mask=cfg.use_prev_mask,
            dist_map_mode=cfg.dist_map_mode,
            dist_map_radius=cfg.dist_map_radius
        )
        model = model.to(device)
        
    return model

def get_loss_configuration(cfg):
    loss_cfg = edict()
    if cfg.model_type == 'FocalClick':
        loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
        # loss_cfg.instance_loss = BceDiceLoss()
        loss_cfg.instance_loss_weight = 1.0
        loss_cfg.instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
        loss_cfg.instance_aux_loss_weight = 0.4
        loss_cfg.instance_refine_loss = WFNL(alpha=0.5, gamma=2)
        loss_cfg.instance_refine_loss_weight = 1.0
        loss_cfg.trimap_loss = nn.BCEWithLogitsLoss()
        loss_cfg.trimap_loss_weight = 1.0
    elif cfg.model_type == 'RITM':
        loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
        loss_cfg.instance_loss_weight = 1.0
        # loss_cfg.instance_loss = BceDiceLoss()
        # loss_cfg.instance_loss_weight = 1.0
        
        # loss_cfg.instance_loss = FocalLoss(alpha=0.5, gamma=2)
        # loss_cfg.instance_loss_weight = 1.0
        
        # loss_cfg.instance_loss = DiceLoss()
        # loss_cfg.instance_loss_weight = 1.0
        
        loss_cfg.instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
        loss_cfg.instance_aux_loss_weight = 0.4
        
        # loss_cfg.oversegmenting_loss = ClickDiceLoss()
        # loss_cfg.oversegmenting_loss_weight = 0.5
        
        # loss_cfg.pre_stage_features_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
        # loss_cfg.pre_stage_features_loss_weight = 1.0
        
    elif cfg.model_type == 'RITMUPP':
        loss_cfg.instance_loss = DiceLossSigmoid()
        loss_cfg.instance_loss_weight = 0.5
        
        loss_cfg.instance_aux_loss = BinaryCrossEntropyLossSigmoid()
        loss_cfg.instance_aux_loss_weight = 0.5
        
    elif cfg.model_type.startswith('SimpleClick'):
        loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
        loss_cfg.instance_loss_weight = 1.0
    elif cfg.model_type == 'UnetPlusPlus':
        loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
        loss_cfg.instance_loss_weight = 1.0
    else:
        raise Exception(f'Model type {cfg.model_type} unknown')
    
    logger.info(f'Loss configuration for model: {cfg.model_type}: {loss_cfg}')
    
    return loss_cfg

def configure_datasets(cfg, model_cfg):
    preprocessor_config = OmegaConf.to_container(cfg.preprocessing, resolve=True)
    preprocessor = Preprocessor(preprocessor_config)
    
    logger.info(f'Preprocessing config: {preprocessor_config}')
    
    with_refiner = cfg.model_type == 'FocalClick'
    
    logger.info(f'{"No " if not with_refiner else ""}Refiner data for model {cfg.model_type}')
    
    train_set = utils.get_dataset(cfg.dataset.train, cfg.data_paths, preprocessor, with_refiner=with_refiner)
    train_set.epoch_len = cfg.epoch_length
    val_set = utils.get_dataset(cfg.dataset.val, cfg.data_paths, preprocessor, with_refiner=with_refiner)
    val_epoch_len = (
        int(cfg.epoch_length / 10) if cfg.epoch_length > 0 else -1
    )  # 10% of train if limited, else all
    val_set.epoch_len = val_epoch_len
    
    logger.info(f'Train set: len: {len(train_set)}, epoch_len: {train_set.epoch_len}')
    logger.info(f'Val set: len: {len(val_set)}, epoch_len: {val_set.epoch_len}')

    augmentations_provider = AugmentationsProvider()
    train_set.augmentator = augmentations_provider.get_augmentator(cfg, model_cfg)

    points_sampler = MultiPointSampler(
        model_cfg.num_max_points,
        prob_gamma=0.80,
        merge_objects_prob=0.15,
        max_num_merged_objects=2,
        use_hierarchy=False,
        first_click_center=True,
    )
    train_set.points_sampler = points_sampler
    val_set.points_sampler = points_sampler

    return train_set, val_set



def get_trainer(cfg, model_cfg, model, loss_cfg, train_set, val_set, device):
    
    if cfg.lr_scheduling.enabled:
        if cfg.lr_scheduling.patience > 0:
            lr_scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau,
                                   mode='max', 
                                   factor=0.1, 
                                   patience=cfg.lr_scheduling.patience, 
                                   verbose=True)
        else:
            lr_scheduler = partial(
                torch.optim.lr_scheduler.MultiStepLR,
                milestones=cfg.lr_scheduling.milestones,
                gamma=cfg.lr_scheduling.gamma,
            )
    else:
        lr_scheduler = None
    
    optimizer_params = {"lr": cfg.lr, "betas": (0.9, 0.999), "eps": 1e-8}
    
    if cfg.clicker.mode != 'locked' and not cfg.iterative_trainer:
        raise Exception(f'Dynamic radius points only works with iterative trainers')
    
    trainer_classes = {
        'FocalClick': IterativeFocalClickTrainer if cfg.iterative_trainer else FocalClickTrainer,
        'RITM': IterativeRITMTrainer if cfg.iterative_trainer else RITMTrainer,
        'RITMUPP': IterativeRITMTrainer if cfg.iterative_trainer else RITMTrainer,
        'UnetPlusPlus': UnetPlusPlusTrainer,
        'SimpleClick_T': SimpleClickTrainer,
        'SimpleClick_B': SimpleClickTrainer
    }

    common_params = {
        'model': model,
        'cfg': cfg,
        'model_cfg': model_cfg,
        'loss_cfg': loss_cfg,
        'trainset': train_set,
        'valset': val_set,
        'device': device,
        'clicker_config': {
            'mode': cfg.clicker.mode,
            'size_range_modifier': cfg.clicker.size_range_modifier
        },
        'optimizer': "adam",
        'optimizer_params': optimizer_params,
        'lr_scheduler': lr_scheduler,
        'checkpoint_interval': [(0, 5), (200, 10)],
        'image_dump_interval': 3000,
        'metrics': [AdaptiveIoU(), F1Score(), IoU()],
        'max_interactive_points': model_cfg.num_max_points,
        'max_num_next_clicks': cfg.max_clicks_before_backprop,
        'iterative_evaluation_interval': cfg.iterative_evaluation_interval,
        'early_stopping_patience': cfg.early_stopping_patience
    }

    # Add layerwise_decay for SimpleClickTrainer if needed
    if cfg.model_type == 'SimpleClickTrainer':
        layerwise_decay = True  # Set the appropriate value for layerwise_decay
        common_params['layerwise_decay'] = layerwise_decay

    # Initialize the desired trainer class
    if cfg.model_type in trainer_classes:
        trainer = trainer_classes[cfg.model_type](**common_params)
    else:
        raise ValueError(f"Unknown trainer class: {cfg.model_type}")
    
    logger.info(f'Using {trainer.__class__} for training of model: {model.__class__}')
    
    return trainer