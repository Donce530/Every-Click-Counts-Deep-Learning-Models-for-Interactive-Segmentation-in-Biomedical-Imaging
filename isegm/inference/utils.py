from datetime import timedelta
from pathlib import Path

from isegm.model.is_hrnet_model import HRNetModel as FocalClickHRNetModel
from isegm.model.is_segformer_model import SegFormerModel as FocalClickSegFormer
from isegm.ritm.model.is_hrnet_model import HRNetModel as RITMModel
from isegm.simple_click.model.is_plainvit_model import PlainVitModel as SimpleClickModel
from isegm.model.is_unetplusplus_model import UnetPlusPlusModel

from isegm.inference.predictors import get_predictor as get_focalclick_predictor
from isegm.ritm.inference.predictors import get_predictor as get_ritm_predictor
from isegm.simple_click.inference.predictors import get_predictor as get_simpleclick_predictor
from isegm.inference.predictors.unetplusplus_predictor import get_predictor as get_unetplusplus_predictor

from isegm.simple_click.model.modeling.pos_embed import interpolate_pos_embed_inference

from isegm.utils.log import logger

import torch
import numpy as np
import os

from isegm.data.datasets import (
    GrabCutDataset,
    BerkeleyDataset,
    DavisDataset,
    Davis2017Dataset,
    SBDEvaluationDataset,
    PascalVocDataset,
    Davis585Dataset,
    COCOMValDataset,
    LidcDataset,
    LidcCropsDataset,
    BratsSimpleClickDataset,
    LidcOneSampleDataset,
    Kits23Dataset,
    LitsDataset,
    MdPancDataset,
    CombinedDataset,
)

from isegm.data.preprocess import Preprocessor

from isegm.utils.serialization import load_model


def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def load_is_model(checkpoint, device, **kwargs):
    if isinstance(checkpoint, (str, Path)):
        state_dict = torch.load(checkpoint, map_location='cpu')
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0], device, **kwargs)
        models = [load_single_is_model(x, device, **kwargs) for x in state_dict]

        return model, models
    else:
        return load_single_is_model(state_dict, device, **kwargs)


def load_single_is_model(state_dict, device, **kwargs):
    model_type = kwargs.get('model_type', None)
    for_training = kwargs.get('training', False)
    model_class = (
        get_class_from_model_type(model_type) if model_type is not None else None
    )

    model = load_model(state_dict['config'], model_class=model_class, **kwargs)

    if model_type.startswith('SimpleClick'):
        matched_state_dict = {}
        mismatched_weights = []

        for name, param in state_dict['state_dict'].items():
            if name in model.state_dict() and model.state_dict()[name].size() == param.size():
                matched_state_dict[name] = param
            else:
                mismatched_weights.append(name)
        if mismatched_weights:
            logger.warning(f"Mismatched weights that were not loaded: {mismatched_weights}")
        
        if for_training:
            # when training, load weights with 448*448 dimensions, then interpolate
            model.load_state_dict(matched_state_dict, strict=False)
            interpolate_pos_embed_inference(model.backbone, (512,512), 'cpu')
        else: 
            # when inference, prepare dimensions for 512x512, then load weights
            interpolate_pos_embed_inference(model.backbone, (512,512), 'cpu')
            model.load_state_dict(matched_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict['state_dict'], strict=True)

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model


def get_class_from_model_type(model_type):
    if model_type == 'FocalClick':
        return FocalClickHRNetModel
    if model_type == 'FocalClickSegFormer':
        return FocalClickSegFormer
    if model_type == 'RITM':
        return RITMModel
    if model_type.startswith('SimpleClick'):
        return SimpleClickModel
    if model_type == 'UnetPlusPlus':
        return UnetPlusPlusModel
    raise Exception(f'No model class found for {model_type}')

def get_predictor(
    type, model, device, zoom_in_params, predictor_params
):
    if type == 'FocalClick' or type == 'FocalClickSegFormer':
        return get_focalclick_predictor(
            model,
            'FocalClick',
            device,
            infer_size=512,
            prob_thresh=0.5,
            focus_crop_r=1.40,
            zoom_in_params=zoom_in_params,
            predictor_params=predictor_params
        )
    elif type == 'RITM':
        return get_ritm_predictor(
            model,
            'NoBRS',
            device,
            zoom_in_params=zoom_in_params,
            predictor_params=predictor_params
        )
    elif type == 'RITMUPP':
        return get_ritm_predictor(
            model,
            'NoBRS',
            device,
            zoom_in_params=None, # does not support zoom_in as of yet.
            predictor_params=predictor_params
        )
    elif type.startswith('SimpleClick'):
        return get_simpleclick_predictor(
            model,
            'NoBRS',
            device,
            zoom_in_params=None,
            predictor_params=predictor_params,
            prob_thresh=0.5,
        )
    elif type == 'UnetPlusPlus':
        return get_unetplusplus_predictor(
            model,
            'UnetPlusPlus',
            device
        )
    else:
        raise Exception('Predictor error')


def get_dataset(dataset_name, cfg, preprocessor=Preprocessor(), with_refiner=True):
    if dataset_name == 'GrabCut':
        dataset = GrabCutDataset(cfg.GRABCUT_PATH)
    elif dataset_name == 'Berkeley':
        dataset = BerkeleyDataset(cfg.BERKELEY_PATH)
    elif dataset_name == 'DAVIS':
        dataset = DavisDataset(cfg.DAVIS_PATH)
    elif dataset_name == 'SBD':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH)
    elif dataset_name == 'SBD_Train':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH, split='train')
    elif dataset_name == 'PascalVOC':
        dataset = PascalVocDataset(cfg.PASCALVOC_PATH, split='val')
    elif dataset_name == 'COCO_MVal':
        dataset = COCOMValDataset(cfg.COCO_MVAL_PATH)
    elif dataset_name == 'D585_SP':
        dataset = Davis585Dataset(cfg.DAVIS585_PATH, init_mask_mode='sp')
    elif dataset_name == 'D585_STM':
        dataset = Davis585Dataset(cfg.DAVIS585_PATH, init_mask_mode='stm')
    elif dataset_name == 'D585_ZERO':
        dataset = Davis585Dataset(cfg.DAVIS585_PATH, init_mask_mode='zero')
    elif dataset_name == 'LIDC_2D':
        dataset = LidcDataset(cfg.LIDC_PATH, preprocessor=preprocessor)
    elif dataset_name == 'LIDC_2D_VAL':
        dataset = LidcDataset(cfg.LIDC_VAL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'LIDC_2D_VAL_ONE_SAMPLE':
        dataset = LidcOneSampleDataset(cfg.LIDC_VAL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'LIDC_2D_TEST':
        dataset = LidcDataset(cfg.LIDC_TEST_PATH, preprocessor=preprocessor)
    elif dataset_name == 'LIDC_2D_256_VAL':
        dataset = LidcDataset(cfg.LIDC_256_VAL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'DAVIS_2017_VAL':
        dataset = Davis2017Dataset(cfg.DAVIS_2017_VAL_PATH)
    elif dataset_name == 'BRATS_SIMPLECLICK':
        dataset = BratsSimpleClickDataset(cfg.BRATS_SIMPLECLICK_PATH)
    elif dataset_name == 'LIDC_2D_FULL':
        dataset = LidcDataset(cfg.LIDC_FULL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'LIDC_2D_FULL_VAL':
        dataset = LidcDataset(cfg.LIDC_FULL_VAL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'LIDC_2D_FULL_TEST':
        dataset = LidcDataset(cfg.LIDC_FULL_TEST_PATH, preprocessor=preprocessor)
        
    elif dataset_name == 'KITS23_2D_TUMOURS':
        dataset = Kits23Dataset(cfg.KITS23_PATH, preprocessor=preprocessor)
    elif dataset_name == 'KITS23_2D_TUMOURS_VAL':
        dataset = Kits23Dataset(cfg.KITS23_VAL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'KITS23_2D_TUMOURS_TEST':
        dataset = Kits23Dataset(cfg.KITS23_TEST_PATH, preprocessor=preprocessor)
    elif dataset_name == 'KITS23_2D_TUMOURS_FULL':
        dataset = Kits23Dataset(cfg.KITS23_FULL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'KITS23_2D_TUMOURS_FULL_VAL':
        dataset = Kits23Dataset(cfg.KITS23_FULL_VAL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'KITS23_2D_TUMOURS_FULL_TEST':
        dataset = Kits23Dataset(cfg.KITS23_FULL_TEST_PATH, preprocessor=preprocessor)
    
    elif dataset_name == 'LITS_2D':
        dataset = LitsDataset(cfg.LITS_PATH, preprocessor=preprocessor)
    elif dataset_name == 'LITS_2D_VAL':
        dataset = LitsDataset(cfg.LITS_VAL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'LITS_2D_TEST':
        dataset = LitsDataset(cfg.LITS_TEST_PATH, preprocessor=preprocessor)
    elif dataset_name == 'LITS_2D_FULL':
        dataset = LitsDataset(cfg.LITS_FULL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'LITS_2D_FULL_VAL':
        dataset = LitsDataset(cfg.LITS_FULL_VAL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'LITS_2D_FULL_TEST':
        dataset = LitsDataset(cfg.LITS_FULL_TEST_PATH, preprocessor=preprocessor)
        
    elif dataset_name == 'MD_PANC_2D':
        dataset = MdPancDataset(cfg.MD_PANC_PATH, preprocessor=preprocessor)
    elif dataset_name == 'MD_PANC_2D_VAL':
        dataset = MdPancDataset(cfg.MD_PANC_VAL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'MD_PANC_2D_TEST':
        dataset = MdPancDataset(cfg.MD_PANC_TEST_PATH, preprocessor=preprocessor)
    elif dataset_name == 'MD_PANC_2D_FULL':
        dataset = MdPancDataset(cfg.MD_PANC_FULL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'MD_PANC_2D_FULL_VAL':
        dataset = MdPancDataset(cfg.MD_PANC_FULL_VAL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'MD_PANC_2D_FULL_TEST':
        dataset = MdPancDataset(cfg.MD_PANC_FULL_TEST_PATH, preprocessor=preprocessor)
        
    elif dataset_name == 'COMBINED_2D':
        dataset = CombinedDataset(cfg.COMBINED_PATH, preprocessor=preprocessor)
    elif dataset_name == 'COMBINED_2D_VAL':
        dataset = CombinedDataset(cfg.COMBINED_VAL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'COMBINED_2D_TEST':
        dataset = CombinedDataset(cfg.COMBINED_TEST_PATH, preprocessor=preprocessor)
    elif dataset_name == 'COMBINED_2D_FULL':
        dataset = CombinedDataset(cfg.COMBINED_FULL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'COMBINED_2D_FULL_VAL':
        dataset = CombinedDataset(cfg.COMBINED_FULL_VAL_PATH, preprocessor=preprocessor)
    elif dataset_name == 'COMBINED_2D_FULL_TEST':
        dataset = CombinedDataset(cfg.COMBINED_FULL_TEST_PATH, preprocessor=preprocessor)
        
    else:
        assert False, f'Unknown dataset: {dataset_name}'
        
    dataset.with_refiner = with_refiner
    
    return dataset


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(
        np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv
    ).sum()
    union = np.logical_and(
        np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv
    ).sum()

    return intersection / union


def get_f1_score(gt, pred):
    # tp = np.logical_and(pred_mask, gt_mask).sum()
    # fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    # fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()

    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)

    # f1_score = 2 * precision * recall / (precision + recall)
    pred_mask = (pred > 0.5).astype(np.float32)
    gt_mask = (gt > 0.5).astype(np.float32)
    true_positive = np.sum(pred_mask * gt_mask)
    false_positive = np.sum(pred_mask * (1 - gt_mask))
    false_negative = np.sum((1 - pred_mask) * gt_mask)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1 = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return f1


def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array(
            [_get_noc(iou_arr, iou_thr) for iou_arr in all_ious], dtype=np.int
        )

        score = scores_arr.mean()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        over_max_list.append(over_max)

    return noc_list, over_max_list

def compute_scaled_iou_error(all_ious, max_clicks=20):
    average_ious = [np.nanmean(all_ious[:, i]) for i in range(max_clicks)]
    scaled_iou_error = np.sum([(1 - iou) * (1 + 0.1 * i) for i, iou in enumerate(average_ious)])
    
    max_error = np.sum([1 + 0.1 * i for i in range(max_clicks)])
    normalized_error = scaled_iou_error / max_error
    
    return normalized_error


def find_checkpoint(weights_folder, checkpoint_name):
    weights_folder = Path(weights_folder)
    if ':' in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(':')
        models_candidates = [
            x for x in weights_folder.glob(f'{model_name}*') if x.is_dir()
        ]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith('.pth'):
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f'{checkpoint_name}*.pth'))
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]

    return str(checkpoint_path)


def find_checkpoint_from_training(training_output_path, training_run_name, checkpoint_name='best_model.pth'):
    date_folders = os.listdir(training_output_path)
    date_folders.sort()
    date_folders.reverse()
    for day_folder in date_folders:
        run_names = os.listdir(os.path.join(training_output_path, day_folder))
        run_names.sort()
        run_names.reverse()
        for run_name in run_names:
            if run_name.endswith(f'-{training_run_name}'):
                checkpoint_path = os.path.join(
                    training_output_path,
                    day_folder,
                    run_name,
                    'checkpoints',
                    checkpoint_name,
                )

                return checkpoint_path

    assert False, f'Could not find checkpoint for {training_run_name}'


def get_results_table(
    noc_list,
    over_max_list,
    brs_type,
    dataset_name,
    mean_spc,
    elapsed_time,
    n_clicks=20,
    model_name=None,
):
    table_header = (
        f'|{"Pipeline":^13}|{"Dataset":^11}|'
        f'{"NoC@80%":^9}|{"NoC@85%":^9}|{"NoC@90%":^9}|'
        f'{">="+str(n_clicks)+"@85%":^9}|{">="+str(n_clicks)+"@90%":^9}|'
        f'{"SPC,s":^7}|{"Time":^9}|'
    )
    row_width = len(table_header)

    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f'|{brs_type:^13}|{dataset_name:^11}|'
    table_row += f'{noc_list[0]:^9.2f}|'
    table_row += f'{noc_list[1]:^9.2f}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{noc_list[2]:^9.2f}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{over_max_list[1]:^9}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{over_max_list[2]:^9}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{mean_spc:^7.3f}|{eval_time:^9}|'

    return header, table_row
