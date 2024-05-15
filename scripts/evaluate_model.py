from __future__ import annotations
import logging

import sys
import os
from pathlib import Path
import yaml

import torch
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.inference.iterative_evaluation_training import evaluate_dataset
from isegm.data.preprocess import Preprocessor


def setup_config(cfg: DictConfig) -> DictConfig:
    cfg.data_paths.EXPS_PATH = Path(cfg.data_paths.EXPS_PATH)

    if cfg.logs_path == '':
        cfg.logs_path = cfg.data_paths.EXPS_PATH / 'evaluation_logs'
    else:
        cfg.logs_path = Path(cfg.logs_path)

    return cfg


@hydra.main(config_path="../configs", config_name="evaluation_config")
def main(cfg: DictConfig):
    cfg = setup_config(cfg)
    logger = logging.getLogger(__name__)

    logger.info(OmegaConf.to_yaml(cfg))

    device = torch.device('cpu' if cfg.cpu else f"cuda:{cfg.gpus.split(',')[0]}")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    summary_rows = []

    logger.info(f'Start progressive merge after: {cfg.start_progressive_merge} clicks')

    model_setups = load_model_setups(cfg)

    for dataset_name in cfg.datasets:
        dataset = utils.get_dataset(dataset_name, cfg.data_paths)

        for model_setup in model_setups:
            model = utils.load_is_model(
                model_setup['checkpoint_path'],
                device,
                model_type=model_setup['type'],
            )
            
            logger.info(f'Evaluating model {model_setup["checkpoint_path"]}')
            
            logger.info(f'Loaded {model.__class__} model for type {model_setup["type"]}')

            dataset.preprocessor = model_setup['preprocessor']
            
            if cfg.preprocessing.force_windowing:
                if 'COMBINED' in dataset_name:
                    dataset.preprocessor.windowing = False
                    logger.info(f'Forcing windowing off for dataset: {dataset_name}')
                else:
                    logger.info(f'Forcing windowing on dataset: min: {cfg.preprocessing.window_min}, max: {cfg.preprocessing.window_max}')
                    dataset.preprocessor.windowing = True
                    dataset.preprocessor.window_min = cfg.preprocessing.window_min
                    dataset.preprocessor.window_max = cfg.preprocessing.window_max

            logger.info(
                f'Evaluating {dataset.epoch_len} samples from {dataset_name}...'
            )
            
            if 'clicker' in model_setup['train_config'].keys():
                clicker_config = model_setup['train_config']['clicker']
                logger.info(f'Model clicker config loaded: {clicker_config}')
                
                # Backwards compatibility :(
                if 'dynamic_radius_points' in model_setup['train_config'] and not model_setup['train_config']['dynamic_radius_points']:
                    clicker_config['mode'] = 'locked'
                
                if clicker_config['mode'] == 'distributed':
                    clicker_config['mode'] = 'avg' # Deterministic
                if clicker_config['mode'] == 'distributed_only_pos':
                    clicker_config['mode'] = 'avg_only_pos' # Deterministic
                clicker_config['size_range_modifier'] = 0 # Deterministic
                logger.info(f'Model clicker config used: {clicker_config}')
                
                # logger.info(f'Model {"is" if model.dist_maps.overwrite_maps else "is not"} overwriting click maps!')
                
            else:
                clicker_config = {
                    'mode':'locked',
                    'size_range_modifier': 0
                }
                logger.info(f'Clicker config not found for model {model_setup["model_name"]}. Defaulting to locked')
                
            if cfg.force_clicker_config:
                clicker_config = cfg.clicker_config
                logger.info(f'Forcing clicker config: {clicker_config}')
                
            zoom_in_params = dict()
            # if cfg.use_init_mask:
            #     zoom_in_params['skip_clicks'] = 0
            # if cfg.zoom_in.enabled:
            #     zoom_in_params['optimistic_masking'] = cfg.zoom_in.optimistic_masking
            zoom_in_params['recompute_click_size_on_zoom'] = clicker_config['mode'] != 'locked'
            logger.info(f'Zoom in params: {zoom_in_params}')
            
            predictor_params = dict()
            if model_setup['type'] == 'FocalClick':
                predictor_params['ensure_minimum_focus_crop_size'] = cfg.predictor.ensure_minimum_focus_crop_size
                
            logger.info(f'Predictor params: {predictor_params}')
            
            predictor = utils.get_predictor(
                model_setup['type'],
                model,
                device,
                zoom_in_params,
                predictor_params
            )
            
            logger.info(f'Loaded {predictor.__class__} predictor for type {model_setup["type"]}')

            mean_ious, noc, nof, iou_error, all_ious = evaluate_dataset(
                dataset,
                predictor,
                logger=logger,
                clicker_config=clicker_config,
                pred_thr=cfg.thresh,
                max_iou_thr=cfg.target_iou,
                min_clicks=cfg.min_n_clicks,
                max_clicks=cfg.n_clicks,
                use_init_mask=cfg.use_init_mask,
                start_progressive_merge_on_click=cfg.start_progressive_merge,
            )

            save_all_ious(
                model_setup['model_name'],
                dataset_name,
                all_ious,
                output_dir,
            )

            summary_row = get_model_summary(
                model_setup['model_name'], dataset_name, mean_ious, noc, nof, iou_error
            )
            summary_rows.append(summary_row)

        iou_thrs = [0.85, 0.9, 0.95]
        summary_df = pd.DataFrame(
            summary_rows,
            columns=['model', 'dataset']
            + [f'iou_{i+1}_clicks' for i in range(all_ious.shape[1])]
            + [f'NoC@{iou}' for iou in iou_thrs]
            + [f'NoF@{iou}' for iou in iou_thrs]
            + [f'IoU_Error']
        )
        logger.info(f'\nSummary:\n{summary_df}')
        summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)


def load_model_setups(cfg):
    if 'model_training_paths' in cfg and cfg.model_training_paths != None:
        training_paths = [Path(x) for x in cfg.model_training_paths]
        checkpoints_list = []
        preprocessor_list = []
        model_names = []
        train_configs = []
        for path in training_paths:
            run_config = get_run_config(checkpoints_list, path)
            preprocessor_list.append(Preprocessor(run_config['preprocessing']))
            model_name = run_config['exp_name']
            if 'multirun' in str(path):
                model_name += f'__{path.stem}'
            model_names.append(model_name)
            train_configs.append(run_config)
    elif 'model_training_parent_dir' in cfg and cfg.model_training_parent_dir != None:
        training_paths = [Path(os.path.join(cfg.model_training_parent_dir, x)) for x in cfg.model_training_parent_dir]
        checkpoints_list = []
        preprocessor_list = []
        model_names = []
        train_configs = []
        for path in training_paths:
            run_config = get_run_config(checkpoints_list, path)
            preprocessor_list.append(Preprocessor(run_config['preprocessing']))
            model_name = run_config['exp_name']
            if 'multirun' in str(path):
                model_name += f'__{path.stem}'
            model_name = f'{run_config["model_type"]}_{model_name}'
            model_names.append(model_name)
            train_configs.append(run_config)
    else:
        checkpoints_list = [
            os.path.join(cfg.models_path, x) for x in os.listdir(cfg.models_path)
        ]
        checkpoints_list = [Path(x) for x in checkpoints_list]
        model_names = [x.stem for x in checkpoints_list]
        preprocessor_list = [Preprocessor() for _ in checkpoints_list]
        train_configs = [None for _ in checkpoints_list]

    model_setups = [
        {
            'model_name': f'{train_config["model_type"]} {model_name} {train_config["dataset"]["train"]}',
            'checkpoint_path': checkpoint_path,
            'preprocessor': preprocessor,
            'train_config': train_config,
            'type': train_config['model_type']
        }
        for model_name, checkpoint_path, preprocessor, train_config in zip(
            model_names, checkpoints_list, preprocessor_list, train_configs
        )
    ]

    return model_setups


def get_run_config(checkpoints_list, path):
    # checkpoints_list.append(path / 'checkpoints' / 'iterative_best_model.pth')
    checkpoints_list.append(path / 'checkpoints' / 'best_model.pth')
    # checkpoints_list.append(path / 'checkpoints' / 'last_checkpoint.pth')

    hydra_config_dir = path / '.hydra'

    base_config = OmegaConf.load(str(hydra_config_dir / "config.yaml"))
    base_config = OmegaConf.to_container(base_config, resolve=True)

    overrides = OmegaConf.load(str(hydra_config_dir / "overrides.yaml"))
    overrides = [
        (override.split('=')[0], override.split('=')[1]) for override in overrides
    ]

    for key, value in overrides:
        keys = key.split('.')
        current_dict = base_config

        for k in keys[:-1]:
            if k in current_dict:
                current_dict = current_dict[k]
            else:
                current_dict[k] = {}
                current_dict = current_dict[k]

        current_dict[keys[-1]] = yaml.safe_load(value)

    return base_config

def get_model_summary(model_name, dataset_name, mean_ious, noc, nof, iou_error):
    summary_row = [model_name, dataset_name]
    summary_row.extend(mean_ious)
    summary_row.extend(noc)
    summary_row.extend(nof)
    summary_row.extend([iou_error])
    return summary_row


def save_all_ious(
    model_name, dataset_name, all_ious, output_dir
):
    ious_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(ious_dir, exist_ok=True)
    
    column_names = [f'iou_{i+1}_clicks' for i in range(all_ious.shape[1])]
    sample_ids = [i for i in range(all_ious.shape[0])]

    ious_df = pd.DataFrame(all_ious, index=sample_ids, columns=column_names)
    
    ious_path = os.path.join(ious_dir, f'{model_name}-{dataset_name}.csv')
    ious_df.to_csv(ious_path)


if __name__ == '__main__':
    main()
