from time import time

import numpy as np
import torch
import os
from isegm.inference import utils
from isegm.inference.clicker import DynamicClicker
import shutil
import cv2
from isegm.utils.vis import add_tag


try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(
    dataset, predictor, vis=True, vis_path='./experiments/vis_val/', **kwargs
):
    all_ious = []
    if vis:
        save_dir = vis_path + dataset.name + '/'
        # save_dir = '/home/admin/workspace/project/data/logs/'+ dataset.name + '/'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        save_dir = None

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        _, sample_ious, _ = evaluate_sample(
            sample.image,
            sample.gt_mask,
            sample.init_mask,
            predictor,
            sample_id=index,
            vis=vis,
            save_dir=save_dir,
            index=index,
            **kwargs
        )
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def Progressive_Merge(pred_mask, previous_mask, y, x):
    info = {}

    info['predicted_mask'] = pred_mask
    info['previous_mask'] = previous_mask
    info['click_coords'] = (y, x)

    diff_regions = np.logical_xor(previous_mask, pred_mask)

    info['diff_regions'] = diff_regions

    num, labels = cv2.connectedComponents(diff_regions.astype(np.uint8))
    info['connected_labels'] = labels

    label = labels[y, x]
    info['point_label'] = label

    corr_mask = labels == label

    info['corr_mask'] = corr_mask

    info['prev_mask_positive'] = previous_mask[y, x] == 1

    if previous_mask[y, x] == 1:
        progressive_mask = np.logical_and(previous_mask, np.logical_not(corr_mask))
        info['progressive_mask_construction'] = {
            'prev_mask': previous_mask,
            'corr_mask': np.logical_not(corr_mask),
            'progressive_mask': progressive_mask,
        }
    else:
        progressive_mask = np.logical_or(previous_mask, corr_mask)
        info['progressive_mask_construction'] = {
            'prev_mask': previous_mask,
            'corr_mask': corr_mask,
            'progressive_mask': progressive_mask,
        }

    return progressive_mask, info


def Progressive_Merge_V2(pred_mask, previous_mask, y, x, is_positive_click):
    info = {}

    info['predicted_mask'] = pred_mask
    info['previous_mask'] = previous_mask
    info['click_coords'] = (y, x)

    diff_regions = np.logical_xor(previous_mask, pred_mask)

    info['diff_regions'] = diff_regions

    num, labels = cv2.connectedComponents(diff_regions.astype(np.uint8))
    info['connected_labels'] = labels

    label = labels[y, x]
    info['point_label'] = label

    corr_mask = labels == label

    info['corr_mask'] = corr_mask

    prev_mask_positive = previous_mask[y, x] == 1
    info['prev_mask_positive'] = prev_mask_positive
    new_mask_positive = pred_mask[y, x] == 1
    info['new_mask_positive'] = new_mask_positive

    # if the click doesn't match the new region, then the segmentation is useless
    if is_positive_click != new_mask_positive:
        info['progressive_mask_construction'] = {
            'progressive_mask': previous_mask,
        }
        return previous_mask, info

    if previous_mask[y, x] == 1:
        progressive_mask = np.logical_and(previous_mask, np.logical_not(corr_mask))
        info['progressive_mask_construction'] = {
            'prev_mask': previous_mask,
            'corr_mask': np.logical_not(corr_mask),
            'progressive_mask': progressive_mask,
        }
    else:
        progressive_mask = np.logical_or(previous_mask, corr_mask)
        info['progressive_mask_construction'] = {
            'prev_mask': previous_mask,
            'corr_mask': corr_mask,
            'progressive_mask': progressive_mask,
        }

    return progressive_mask, info


def evaluate_sample(
    image,
    gt_mask,
    init_mask,
    predictor,
    clicker,
    max_iou_thr,
    pred_thr=0.5,
    min_clicks=1,
    max_clicks=20,
    sample_id=None,
    callback=None,
    start_progressive_merge_on_click=-1,
):
    pred_mask = np.zeros_like(gt_mask)
    prev_mask = pred_mask
    ious_list = np.repeat(np.nan, max_clicks)
    progressive_mode = start_progressive_merge_on_click > 0
    pred_mask_list = []
    info_list = []

    with torch.no_grad():
        predictor.set_input_image(image)
        predictor.set_prev_mask(init_mask)
        pred_mask = init_mask

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs, info = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            info['progressive_merge_activated'] = False

            if progressive_mode:
                clicks = clicker.get_clicks()
                if len(clicks) >= start_progressive_merge_on_click:
                    last_click = clicks[-1]
                    last_y, last_x = last_click.coords[0], last_click.coords[1]
                    pred_mask, prog_info = Progressive_Merge(
                        pred_mask, prev_mask, last_y, last_x
                    )
                    # pred_mask, prog_info = Progressive_Merge_V2(
                    #     pred_mask, prev_mask, last_y, last_x, last_click.is_positive
                    # )

                    predictor.transforms[0]._prev_probs = np.expand_dims(
                        np.expand_dims(pred_mask, 0), 0
                    )
                    prog_info['click_is_positive'] = last_click.is_positive
                    info['progressive_info'] = prog_info
                    info['progressive_merge_activated'] = True
            if callback is not None:
                callback(
                    image,
                    gt_mask,
                    pred_probs,
                    sample_id,
                    click_indx,
                    clicker.clicks_list,
                )

            info['final_mask'] = pred_mask

            info_list.append(info)
            pred_mask_list.append(pred_mask)
            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list[click_indx] = iou
            prev_mask = pred_mask

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break
            else:
                clicks_list = clicker.get_clicks()
                last_y, last_x = predictor.last_y, predictor.last_x
            
        return (
            clicker.clicks_list,
            np.array(ious_list, dtype=np.float32),
            pred_probs,
            pred_mask_list,
            info_list,
        )
